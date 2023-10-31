"""Test training with parameter groups (layer-dependent hyper-parameters)."""

from copy import deepcopy
from test.utils import report_nonclose

from torch import manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD


def test_param_groups():
    """Compare optimizer w/ layer dependent hyper-parameters."""
    manual_seed(0)
    MAX_STEPS = 200

    mini_batch_size = 32
    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=mini_batch_size, shuffle=True
    )

    # _ indicates that a single optimizer with layer-dependent
    # hyper-parameters is used, _sep indicates that two separate optimizers are used

    # NOTE We will update the convolution layer differently than the linear layers
    # for illustration
    conv_hyperparams = {
        "lr": 1e-3,
        "damping": 1e-2,
        "momentum": 0.5,
        "weight_decay": 5e-2,
        "lr_cov": 5e-2,
        "loss_average": "batch",
        "T": 1,
        "alpha1": 0.2,
        "structures": ("diagonal", "diagonal"),
    }
    linear_hyperparams = {
        "lr": 5e-4,
        "damping": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_cov": 1e-2,
        "loss_average": "batch",
        "T": 3,
        "alpha1": 0.5,
        "structures": ("dense", "dense"),
    }
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model.train()
    conv, linear1, linear2 = model[0], model[3], model[5]

    model_sep = deepcopy(model)
    model_sep.train()
    conv_sep, linear1_sep, linear2_sep = model_sep[0], model_sep[3], model_sep[5]

    loss_func = CrossEntropyLoss()
    loss_func_sep = deepcopy(loss_func)

    param_groups = [
        {"params": conv.parameters(), **conv_hyperparams},
        {
            "params": list(linear1.parameters()) + list(linear2.parameters()),
            **linear_hyperparams,
        },
    ]
    optim = SINGD(model, params=param_groups)

    optim_sep_conv = SINGD(model_sep, params=conv_sep.parameters(), **conv_hyperparams)
    optim_sep_linear = SINGD(
        model_sep,
        params=list(linear1_sep.parameters()) + list(linear2_sep.parameters()),
        **linear_hyperparams,
    )

    losses = []

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim.steps}")

        # Zero gradient buffers
        optim.zero_grad()
        optim_sep_conv.zero_grad()
        optim_sep_linear.zero_grad()

        # Take a step
        loss = loss_func(model(inputs), target)
        loss.backward()
        losses.append(loss.item())
        optim.step()

        loss_sep = loss_func_sep(model_sep(inputs), target)
        loss_sep.backward()
        optim_sep_conv.step()
        optim_sep_linear.step()

        assert len(optim.module_names) == len(optim_sep_conv.module_names) + len(
            optim_sep_linear.module_names
        )

        atol = 1e-7
        rtol = 1e-5

        # compare K, C, m_K, m_C on convolution layer
        for name in optim_sep_conv.module_names.values():
            K = optim.Ks[name].to_dense()
            K_scale = optim_sep_conv.Ks[name].to_dense()
            report_nonclose(K, K_scale, atol=atol, rtol=rtol, name="K")

            m_K = optim.m_Ks[name].to_dense()
            m_K_sep = optim_sep_conv.m_Ks[name].to_dense()
            report_nonclose(m_K, m_K_sep, atol=atol, rtol=rtol, name="m_K")

            C = optim.Cs[name].to_dense()
            C_sep = optim_sep_conv.Cs[name].to_dense()
            report_nonclose(C, C_sep, atol=atol, rtol=rtol, name="C")

            m_C = optim.m_Cs[name].to_dense()
            m_C_sep = optim_sep_conv.m_Cs[name].to_dense()
            report_nonclose(m_C, m_C_sep, atol=atol, rtol=rtol, name="m_C")

        # compare K, C, m_K, m_C on linear layer
        for name in optim_sep_linear.module_names.values():
            K = optim.Ks[name].to_dense()
            K_scale = optim_sep_linear.Ks[name].to_dense()
            report_nonclose(K, K_scale, atol=atol, rtol=rtol, name="K")

            m_K = optim.m_Ks[name].to_dense()
            m_K_sep = optim_sep_linear.m_Ks[name].to_dense()
            report_nonclose(m_K, m_K_sep, atol=atol, rtol=rtol, name="m_K")

            C = optim.Cs[name].to_dense()
            C_sep = optim_sep_linear.Cs[name].to_dense()
            report_nonclose(C, C_sep, atol=atol, rtol=rtol, name="C")

            m_C = optim.m_Cs[name].to_dense()
            m_C_sep = optim_sep_linear.m_Cs[name].to_dense()
            report_nonclose(m_C, m_C_sep, atol=atol, rtol=rtol, name="m_C")

        # compare parameter values
        for p1, p2 in zip(model.parameters(), model_sep.parameters()):
            report_nonclose(p1, p2, atol=atol, rtol=rtol, name="parameters")

        if batch_idx >= MAX_STEPS:
            break
