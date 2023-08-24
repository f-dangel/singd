"""Check micro-batch support (optimizer can be used with gradient accumulation)."""

from copy import deepcopy

from torch import allclose, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD


def test_micro_batches():
    """Compare optimizer on mini-batch with optimizer operating on micro-batches."""
    manual_seed(0)
    MAX_STEPS = 30

    mini_batch_size = 32
    micro_batch_size = 8

    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=mini_batch_size, shuffle=True
    )

    # _mini indicates the mini-batch version
    # _micro indicates the micro-batch version

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved
    model_mini = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_micro = deepcopy(model_mini)

    loss_func_mini = CrossEntropyLoss()
    loss_func_micro = deepcopy(loss_func_mini)

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

    optim_mini = SNGD(model_mini, **optim_hyperparams)
    optim_micro = SNGD(model_micro, **optim_hyperparams)

    model_mini.train()
    model_micro.train()

    tolerances = {"rtol": 1e-5, "atol": 5e-7}

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim_mini.steps}")

        # Zero gradient buffers
        optim_mini.zero_grad()
        optim_micro.zero_grad()

        # Take a step
        loss_func_mini(model_mini(inputs), target).backward()
        optim_mini.step()

        inputs_split = inputs.split(micro_batch_size)
        target_split = target.split(micro_batch_size)
        for input_micro, target_micro in zip(inputs_split, target_split):
            loss_func_micro(model_micro(input_micro), target_micro).backward()

        # un-scale the accumulated gradients
        for p in model_micro.parameters():
            p.grad *= micro_batch_size / mini_batch_size
        optim_micro.step()

        # compare K, C, m_K, m_C
        assert len(optim_mini.modules) == len(optim_micro.modules)
        for m_mini, m_ours in zip(optim_mini.modules, optim_micro.modules):
            K_mini = optim_mini.Ks[m_mini].to_dense()
            K_ours = optim_micro.Ks[m_ours].to_dense()
            assert allclose(K_mini, K_ours, **tolerances)

            m_K_mini = optim_mini.m_Ks[m_mini].to_dense()
            m_K_ours = optim_micro.m_Ks[m_ours].to_dense()
            assert allclose(m_K_mini, m_K_ours, **tolerances)

            C_mini = optim_mini.Cs[m_mini].to_dense()
            C_ours = optim_micro.Cs[m_ours].to_dense()
            assert allclose(C_mini, C_ours, **tolerances)

            m_C_mini = optim_mini.m_Cs[m_mini].to_dense()
            m_C_ours = optim_micro.m_Cs[m_ours].to_dense()
            assert allclose(m_C_mini, m_C_ours, **tolerances)

        # compare momentum buffers
        for p_mini, p_ours in zip(model_mini.parameters(), model_micro.parameters()):
            mom_mini = optim_mini.state[p_mini]["momentum_buffer"]
            mom_ours = optim_micro.state[p_ours]["momentum_buffer"]
            assert allclose(mom_mini, mom_ours, **tolerances)

        # compare model parameters
        for p_mini, p_ours in zip(model_mini.parameters(), model_micro.parameters()):
            assert allclose(p_mini, p_ours, **tolerances)

        if batch_idx >= MAX_STEPS:
            break


if __name__ == "__main__":
    test_micro_batches()
