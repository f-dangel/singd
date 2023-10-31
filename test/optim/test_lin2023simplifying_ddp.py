"""Compare our implementation with the original work by Lin et al. (2023)."""

import sys
from copy import deepcopy
from os import environ, path

import torch
import torch.distributed as dist
from pytest import skip as pytest_skip
from torch import allclose, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD


def test_compare_lin2023simplifying_ddp():  # noqa: C901
    """Compare our implementation with the original one on MNIST in the DDP setting.

    Raises:
        RuntimeError: If the test is not run via torchrun.
    """
    N_GPUS = torch.cuda.device_count()
    if not torch.cuda.is_available() or N_GPUS < 2:
        # This test requires multiple GPUs.
        pytest_skip()
    try:
        RANK = int(environ["LOCAL_RANK"])
    except KeyError as e:
        raise RuntimeError("This test has to be run via torchrun.") from e
    DEVICE = torch.device(f"cuda:{RANK}")
    torch.cuda.set_device(RANK)
    # Initialize the process group.
    dist.init_process_group("nccl")
    manual_seed(0)
    MAX_STEPS = 30

    # make original work importable
    HERE = path.abspath(__file__)
    HEREDIR = path.dirname(HERE)
    LIN2023_REPO_DIR = path.join(
        path.dirname(path.dirname(HEREDIR)), "lin2023simplifying"
    )
    LIN2023_OPTIM_DIR = path.join(LIN2023_REPO_DIR, "optimizers")
    sys.path.append(LIN2023_OPTIM_DIR)
    sys.path.append(LIN2023_REPO_DIR)  # for importing utils
    from local_cov import LocalOptimizer

    batch_size = 32
    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # _original indicates the original implementation by Lin et al. (2023)
    # _ours indicates using our own implementation

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved
    model_original = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        ReLU(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_ours = deepcopy(model_original)
    model_original.to(DEVICE)
    model_ours.to(DEVICE)

    loss_func_original = CrossEntropyLoss()
    loss_func_ours = deepcopy(loss_func_original)

    lr = 5e-4
    damping = 1e-4
    momentum = 0.9
    weight_decay = 1e-2
    lr_cov = 1e-2
    T = 1
    alpha1_beta2 = 0.5

    optim_original = LocalOptimizer(
        model_original,
        lr=lr,
        momentum=momentum,
        damping=damping,
        beta2=alpha1_beta2,
        weight_decay=weight_decay,
        TCov=T,
        TInv=T,
        faster=True,
        lr_cov=lr_cov,
        batch_averaged=True,
    )
    optim_ours = SINGD(
        model_ours,
        params=None,
        lr=lr,
        momentum=momentum,
        damping=damping,
        alpha1=alpha1_beta2,
        weight_decay=weight_decay,
        loss_average="batch",
        T=T,
        lr_cov=lr_cov,
        structures=("dense", "dense"),
    )

    model_original.train()
    model_ours.train()

    tolerances = {"rtol": 1e-5, "atol": 5e-7}

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        if RANK == 0:
            print(f"Step {optim_original.steps}")
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)

        # Zero gradient buffers
        optim_original.zero_grad()
        optim_ours.zero_grad()

        # The original optimizer has a hard-coded schedule for ``lr_cov`` which
        # we immitate here manually
        step_lr_cov = optim_original.lr_cov
        if optim_ours.steps <= 100:
            step_lr_cov = 2e-4
        elif optim_ours.steps < 500:
            step_lr_cov = 2e-3

        for group in optim_ours.param_groups:
            group["lr_cov"] = step_lr_cov

        # The original optimizer has a hard-coded schedule for ``weight_decay`` which
        # we immitate here manually
        step_weight_decay = weight_decay
        if optim_ours.steps < 20 * optim_ours.param_groups[0]["T"]:
            step_weight_decay = 0.0

        for group in optim_ours.param_groups:
            group["weight_decay"] = step_weight_decay

        # Get local batch for each device.
        local_inputs = inputs.view(N_GPUS, batch_size // N_GPUS, *inputs.shape[1:])[
            RANK
        ]
        local_target = target.view(N_GPUS, batch_size // N_GPUS, *target.shape[1:])[
            RANK
        ]

        output_original = model_original(inputs)
        output_ours = model_ours(local_inputs)
        global_output_ours = torch.empty(batch_size, 10, device=DEVICE)
        dist.all_gather_into_tensor(global_output_ours, output_ours)
        assert allclose(output_original, global_output_ours, **tolerances)

        loss_original = loss_func_original(output_original, target)
        loss_ours = loss_func_ours(output_ours, local_target)
        global_loss_ours = loss_ours.clone().detach()
        dist.all_reduce(global_loss_ours, op=dist.ReduceOp.AVG)
        assert allclose(loss_original, global_loss_ours, **tolerances)

        optim_original.acc_stats = True
        loss_original.backward()
        loss_ours.backward()

        optim_original.step()
        optim_ours.step()

        # compare K, C, m_K, m_C
        assert len(optim_original.modules) == len(optim_ours.modules)
        for m_original, m_ours in zip(optim_original.modules, optim_ours.modules):
            K_original = optim_original.A[m_original]
            K_ours = optim_ours.Ks[m_ours].to_dense()
            assert allclose(K_original, K_ours, **tolerances)

            m_K_original = optim_original.m_A[m_original]
            m_K_ours = optim_ours.m_Ks[m_ours].to_dense()
            assert allclose(m_K_original, m_K_ours, **tolerances)

            C_original = optim_original.B[m_original]
            C_ours = optim_ours.Cs[m_ours].to_dense()
            assert allclose(C_original, C_ours, **tolerances)

            m_C_original = optim_original.m_B[m_original]
            m_C_ours = optim_ours.m_Cs[m_ours].to_dense()
            assert allclose(m_C_original, m_C_ours, **tolerances)

        # compare momentum buffers
        for p_original, p_ours in zip(
            model_original.parameters(), model_ours.parameters()
        ):
            mom_original = optim_original.state[p_original]["momentum_buffer"]
            mom_ours = optim_ours.state[p_ours]["momentum_buffer"]
            assert allclose(mom_original, mom_ours, **tolerances)

        # compare model parameters
        for p_original, p_ours in zip(
            model_original.parameters(), model_ours.parameters()
        ):
            assert allclose(p_original, p_ours, **tolerances)

        if batch_idx >= MAX_STEPS:
            break


if __name__ == "__main__":
    """This test has to be run manually with multiple GPUs via torchrun."""
    test_compare_lin2023simplifying_ddp()
