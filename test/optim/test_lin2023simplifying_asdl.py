"""Compare our ASDL implementation with the original work by Lin et al. (2023)."""

import sys
from copy import deepcopy
from os import path

import asdl
import torch
from asdl import SHAPE_KRON
from torch import allclose, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def test_compare_lin2023simplifying():  # noqa: C901
    """Compare our ASDL implementation with the original one on MNIST."""
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
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # _original indicates the original implementation by Lin et al. (2023)

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved. We set ``bias=False`` for all layers because ASDL
    # handles the bias term separately.
    model_original = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2, bias=False),
        ReLU(),
        Flatten(),
        ReLU(),
        Linear(432, 50, bias=False),
        ReLU(),
        Linear(50, 10, bias=False),
    )
    model_asdl = deepcopy(model_original)

    loss_func_original = CrossEntropyLoss()
    loss_func_asdl = deepcopy(loss_func_original)

    lr = 5e-4
    damping = 1e-4
    momentum = 0.9
    weight_decay = 1e-2
    lr_cov = 1e-2
    batch_averaged = True
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
        batch_averaged=batch_averaged,
    )
    asdl_config = asdl.PreconditioningConfig(
        data_size=1,
        damping=damping,
        lr_cov=lr_cov,
        momentum_cov=alpha1_beta2,
        preconditioner_upd_interval=T,
        curvature_upd_interval=T,
    )
    asdl_grad_maker = asdl.InvFreeGradientMaker(
        model_asdl, asdl_config, fisher_type=asdl.FISHER_EMP
    )
    optim_asdl = torch.optim.SGD(
        model_asdl.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model_original.train()
    model_asdl.train()

    tolerances = {"rtol": 1e-5, "atol": 5e-7}

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim_original.steps}")

        # Zero gradient buffers
        optim_original.zero_grad()
        optim_asdl.zero_grad()

        # The original optimizer has a hard-coded schedule for ``lr_cov`` which
        # we immitate here manually
        if optim_original.steps <= 100:
            asdl_grad_maker.config.lr_cov = 2e-4
        elif optim_original.steps < 500:
            asdl_grad_maker.config.lr_cov = 2e-3
        else:
            asdl_grad_maker.config.lr_cov = optim_original.lr_cov

        # The original optimizer has a hard-coded schedule for ``weight_decay``
        # which we immitate here manually
        if optim_original.steps < 20 * T:
            for param_group in optim_asdl.param_groups:
                param_group["weight_decay"] = 0.0
        else:
            for param_group in optim_asdl.param_groups:
                param_group["weight_decay"] = weight_decay

        output_original = model_original(inputs)
        loss_original = loss_func_original(output_original, target)

        optim_original.acc_stats = True
        loss_original.backward()

        def _loss_fn(batch):
            X, y = batch
            logits = model_asdl(X)
            loss = loss_func_asdl(logits, y)
            return logits, loss

        dummy_y = asdl_grad_maker.setup_model_call(_loss_fn, (inputs, target))
        asdl_grad_maker.setup_loss_repr(dummy_y[1])
        (output_asdl, loss_asdl), _ = asdl_grad_maker.forward_and_backward()
        assert allclose(output_original, output_asdl, **tolerances)
        assert allclose(loss_original, loss_asdl, **tolerances)

        optim_original.step()
        optim_asdl.step()

        # compare K, C, m_K, m_C
        asdl_modules = [
            m
            for m in model_asdl.modules()
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
        ]
        assert len(optim_original.modules) == len(asdl_modules)
        for m_original, m_asdl in zip(optim_original.modules, asdl_modules):
            K_original = optim_original.A[m_original]
            K_asdl = asdl_grad_maker._get_module_symmatrix(
                m_asdl, SHAPE_KRON
            ).K.to_dense()
            assert allclose(K_original, K_asdl, **tolerances)

            m_K_original = optim_original.m_A[m_original]
            m_K_asdl = asdl_grad_maker._get_module_symmatrix(
                m_asdl, SHAPE_KRON
            ).m_K.to_dense()
            assert allclose(m_K_original, m_K_asdl, **tolerances)

            C_original = optim_original.B[m_original]
            C_asdl = asdl_grad_maker._get_module_symmatrix(
                m_asdl, SHAPE_KRON
            ).C.to_dense()
            assert allclose(C_original, C_asdl, **tolerances)

            m_C_original = optim_original.m_B[m_original]
            m_C_asdl = asdl_grad_maker._get_module_symmatrix(
                m_asdl, SHAPE_KRON
            ).m_C.to_dense()
            assert allclose(m_C_original, m_C_asdl, **tolerances)

        # compare momentum buffers
        for p_original, p_asdl in zip(
            model_original.parameters(),
            model_asdl.parameters(),
        ):
            mom_original = optim_original.state[p_original]["momentum_buffer"]
            mom_asdl = optim_asdl.state[p_asdl]["momentum_buffer"]
            assert allclose(mom_original, mom_asdl, **tolerances)

        # compare model parameters
        for p_original, p_asdl in zip(
            model_original.parameters(),
            model_asdl.parameters(),
        ):
            assert allclose(p_original, p_asdl, **tolerances)

        if batch_idx >= MAX_STEPS:
            break
