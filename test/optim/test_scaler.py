"""Test training with gradient scaling."""

from copy import deepcopy
from test.utils import report_nonclose

from torch import manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD


def test_scaler():
    """Compare optimizer with optimizer w/ gradient scaling."""
    manual_seed(0)
    MAX_STEPS = 150

    mini_batch_size = 32

    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=mini_batch_size, shuffle=True
    )

    # _scale indicates the mixed precision version

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_scale = deepcopy(model)

    loss_func = CrossEntropyLoss()
    loss_func_scale = deepcopy(loss_func)

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

    optim = SNGD(model, **optim_hyperparams)
    optim_scale = SNGD(model_scale, **optim_hyperparams)

    model.train()
    model_scale.train()

    tolerances = {"rtol": 1e-2, "atol": 5e-5}

    losses = []

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim.steps}")

        # Zero gradient buffers
        optim.zero_grad()
        optim_scale.zero_grad()

        # Take a step
        loss = loss_func(model(inputs), target)
        loss.backward()
        losses.append(loss.item())
        optim.step()

        # NOTE This is NOT how you would use gradient scaling.
        # It serves for testing purposes because ``GradientScaler`` only
        # works with CUDA and we want the test to run on CPU.
        GRAD_SCALE = 10_000
        optim_scale.grad_scale = GRAD_SCALE

        output_scale = model_scale(inputs)
        loss_scale = loss_func_scale(output_scale, target)
        (GRAD_SCALE * loss_scale).backward()
        optim_scale.step()

        # compare K, C, m_K, m_C
        assert len(optim.modules) == len(optim_scale.modules)
        for m, m_scale in zip(optim.modules, optim_scale.modules):
            K = optim.Ks[m].to_dense()
            K_scale = optim_scale.Ks[m_scale].to_dense()
            report_nonclose(K, K_scale, **tolerances, name="K")

            m_K = optim.m_Ks[m].to_dense()
            m_K_scale = optim_scale.m_Ks[m_scale].to_dense()
            report_nonclose(m_K, m_K_scale, **tolerances, name="m_K")

            C = optim.Cs[m].to_dense()
            C_scale = optim_scale.Cs[m_scale].to_dense()
            report_nonclose(C, C_scale, **tolerances, name="C")

            m_C = optim.m_Cs[m].to_dense()
            m_C_scale = optim_scale.m_Cs[m_scale].to_dense()
            report_nonclose(m_C, m_C_scale, **tolerances, name="m_C")

        # compare momentum buffers
        if optim_hyperparams["momentum"] != 0:
            for p, p_scale in zip(model.parameters(), model_scale.parameters()):
                mom = optim.state[p]["momentum_buffer"]
                mom_scale = optim_scale.state[p_scale]["momentum_buffer"]
                report_nonclose(mom, mom_scale, **tolerances, name="momentum")

        # compare model parameters
        for p, p_scale in zip(model.parameters(), model_scale.parameters()):
            report_nonclose(p, p_scale, **tolerances, name="parameters")

        if batch_idx >= MAX_STEPS:
            break
