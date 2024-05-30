"""Integration test: Train on MNIST using KFAC-like updates."""

from torch import manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD


def test_integration_kfac_like():
    """Integration test that uses ``kfac_like=True`` in the optimizer."""
    manual_seed(0)
    MAX_STEPS = 100

    batch_size = 32
    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        ReLU(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    loss_func = CrossEntropyLoss()

    optim = SINGD(model, alpha1=0.0, T=1, kfac_like=True)
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim.steps}")

        optim.zero_grad()
        output = model(inputs)
        loss = loss_func(output, target)
        loss.backward()
        optim.step()

        # check that buffers for ``m_K`` and ``m_C`` are empty
        assert len(optim.m_Ks) == 0
        assert len(optim.m_Cs) == 0

        if batch_idx >= MAX_STEPS:
            break
