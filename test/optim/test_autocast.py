"""Test mixed-precision training with float16."""

from copy import deepcopy
from test.utils import report_nonclose

from torch import autocast, bfloat16, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD


def test_autocast():
    """Compare optimizer in float32 and optimizer with mixed-precision (float16)."""
    manual_seed(0)
    MAX_STEPS = 8  # NOTE small as the optimizers diverge quickly

    mini_batch_size = 32

    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=mini_batch_size, shuffle=True
    )

    # _single indicates the float32 version
    # _mixed indicates the mixed precision version

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved
    model_single = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_mixed = deepcopy(model_single)

    loss_func_single = CrossEntropyLoss()
    loss_func_mixed = deepcopy(loss_func_single)

    optim_hyperparams = {
        "lr": 5e-4,
        "damping": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_cov": 1e-2,
        "batch_averaged": True,
        "T": 1,
        "alpha1": 0.5,
        "structures": ("dense", "dense"),
    }

    optim_single = SNGD(model_single, **optim_hyperparams)
    optim_mixed = SNGD(model_mixed, **optim_hyperparams)

    model_single.train()
    model_mixed.train()

    tolerances = {"rtol": 1e-2, "atol": 5e-5}
    # momentum requires larger tolerance
    tolerances_momentum = {"rtol": 1e-1, "atol": 5e-3}

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim_single.steps}")

        # Zero gradient buffers
        optim_single.zero_grad()
        optim_mixed.zero_grad()

        # Take a step
        loss_func_single(model_single(inputs), target).backward()
        optim_single.step()

        # NOTE This is NOT how you would use gradient scaling.
        # It serves for testing purposes because ``GradientScaler`` only
        # works with CUDA and we want the test to run on CPU.
        GRAD_SCALE = 10_000
        optim_mixed.grad_scale = GRAD_SCALE

        with autocast(device_type="cpu", dtype=bfloat16):
            output_mixed = model_mixed(inputs)
            assert output_mixed.dtype == bfloat16  # due to linear layers
            loss_mixed = loss_func_mixed(output_mixed, target)
            (GRAD_SCALE * loss_mixed).backward()
        optim_mixed.step()

        # compare K, C, m_K, m_C
        assert len(optim_single.modules) == len(optim_mixed.modules)
        for m_single, m_mixed in zip(optim_single.modules, optim_mixed.modules):
            K_single = optim_single.Ks[m_single].to_dense()
            K_mixed = optim_mixed.Ks[m_mixed].to_dense()
            report_nonclose(K_single, K_mixed, **tolerances, name="K")

            m_K_single = optim_single.m_Ks[m_single].to_dense()
            m_K_mixed = optim_mixed.m_Ks[m_mixed].to_dense()
            report_nonclose(m_K_single, m_K_mixed, **tolerances, name="m_K")

            C_single = optim_single.Cs[m_single].to_dense()
            C_mixed = optim_mixed.Cs[m_mixed].to_dense()
            report_nonclose(C_single, C_mixed, **tolerances, name="C")

            m_C_single = optim_single.m_Cs[m_single].to_dense()
            m_C_mixed = optim_mixed.m_Cs[m_mixed].to_dense()
            report_nonclose(m_C_single, m_C_mixed, **tolerances, name="m_C")

        # compare momentum buffers
        if optim_hyperparams["momentum"] != 0:
            for p_single, p_mixed in zip(
                model_single.parameters(), model_mixed.parameters()
            ):
                mom_single = optim_single.state[p_single]["momentum_buffer"]
                mom_mixed = optim_mixed.state[p_mixed]["momentum_buffer"]
                report_nonclose(
                    mom_single, mom_mixed, **tolerances_momentum, name="momentum"
                )

        # compare model parameters
        for p_single, p_mixed in zip(
            model_single.parameters(), model_mixed.parameters()
        ):
            report_nonclose(p_single, p_mixed, **tolerances, name="parameters")

        if batch_idx >= MAX_STEPS:
            break
