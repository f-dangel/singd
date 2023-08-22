"""Compare our implementation with the original work by Lin et al. (2023)."""

import sys
from copy import deepcopy
from os import path

from torch import allclose, manual_seed
from torch.nn import CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD


def test_compare_lin2023simplifying():
    """Compare our implementation with the original one on MNIST."""
    manual_seed(0)
    MAX_STEPS = 20

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
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # _original indicates the original implementation by Lin et al. (2023)
    # _ours indicates using our own implementation

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved
    model_original = Sequential(
        Flatten(),
        Linear(784, 50),
        ReLU(),
        Linear(50, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_ours = deepcopy(model_original)

    loss_func_original = CrossEntropyLoss()
    loss_func_ours = deepcopy(loss_func_original)

    optim_original = LocalOptimizer(
        model_original,
        lr=5e-4,
        momentum=0.9,
        damping=1e-4,
        beta2=0.5,
        # weight_decay=1e-2, not identical if turned on
        weight_decay=0,
        TCov=1,
        TInv=1,
        faster=True,
        lr_cov=1e-2,
        batch_averaged=True,
    )
    optim_ours = SNGD(
        model_ours,
        lr=5e-4,
        momentum=0.9,
        damping=1e-4,
        alpha1=0.5,
        # weight_decay=1e-2, not identical if turned on
        weight_decay=0,
        batch_averaged=True,
        T=1,
        model_params=None,
        lr_cov=1e-2,
        structures=("dense", "dense"),
    )

    model_original.train()
    model_ours.train()

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim_original.steps}")

        # Zero gradient buffers
        optim_original.zero_grad()
        optim_ours.zero_grad()

        # The original optimizer has a hard-coded schedule for ``lr_cov`` which
        # we immitate here manually
        if optim_ours.steps <= 100:
            optim_ours.lr_cov = 2e-4
        elif optim_ours.steps < 500:
            optim_ours.lr_cov = 2e-3
        else:
            optim_ours.lr_cov = optim_original.lr_cov

        output_original = model_original(inputs)
        output_ours = model_ours(inputs)
        assert allclose(output_original, output_ours)

        loss_original = loss_func_original(output_original, target)
        loss_ours = loss_func_ours(output_ours, target)
        assert allclose(loss_original, loss_ours)

        optim_original.acc_stats = True
        loss_original.backward()
        loss_ours.backward()

        optim_original.step()
        optim_ours.step()

        # compare K, C, m_K, m_C
        assert len(optim_original.modules) == len(optim_ours.modules)
        tolerances = {"rtol": 1e-5, "atol": 1e-7}
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

        # compare model parameters
        for p_original, p_ours in zip(
            model_original.parameters(), model_ours.parameters()
        ):
            assert allclose(p_original, p_ours, **tolerances)

        # manually sync model parameters for p_original, p_ours in
        # zip(model_original.parameters(), model_ours.parameters()):
        # p_ours.data = p_original.clone().detach().data

        if batch_idx >= MAX_STEPS:
            break
