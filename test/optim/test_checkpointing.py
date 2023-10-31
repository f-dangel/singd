"""Test saving and loading the optimizer for checkpointing."""

from test.utils import compare_optimizers
from typing import Tuple

from torch import load, manual_seed, save
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD


def setup() -> Tuple[Sequential, Module, SINGD]:
    """Set up the model, loss function, and optimizer.

    Returns:
        A tuple containing the model, loss function, and optimizer.
    """
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 200),
        Linear(200, 50),
        ReLU(),
        Linear(50, 10),
    )
    loss_func = CrossEntropyLoss()

    params = [
        p
        for m in model.modules()
        if isinstance(m, (Linear, Conv2d))
        for p in m.parameters()
        if p.requires_grad
    ]
    singd_hyperparams = {
        "lr": 5e-4,
        "damping": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_cov": 1e-2,
        "loss_average": "batch",
        "T": 1,
        "alpha1": 0.5,
        "structures": ("dense", "dense"),
    }
    group = {"params": params, **singd_hyperparams}
    param_groups = [group]
    singd = SINGD(model, params=param_groups)

    return model, loss_func, singd


def test_checkpointing():
    """Check whether optimizer is saved/restored correctly while training."""
    manual_seed(0)  # make deterministic
    MAX_STEPS = 100  # quit training after this many steps

    BATCH_SIZE = 32
    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model, loss_func, singd = setup()
    checkpoints = [10, 33, 50]

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {singd.steps}")

        # Save model and optimizer, then restore and compare with original ones
        if batch_idx in checkpoints:
            # keep a reference to compare with restored optimizer
            original_singd = singd

            print("Saving checkpoint")
            save(singd.state_dict(), f"singd_checkpoint_{batch_idx}.pt")
            save(model.state_dict(), f"model_checkpoint_{batch_idx}.pt")
            print("Deleting model and optimizer")
            del model, singd

            print("Loading checkpoint")
            model, _, singd = setup()
            singd.load_state_dict(load(f"singd_checkpoint_{batch_idx}.pt"))
            model.load_state_dict(load(f"model_checkpoint_{batch_idx}.pt"))

            # compare restored and copied optimizer
            compare_optimizers(singd, original_singd)

        # Zero gradient buffers
        singd.zero_grad()

        # Backward pass
        loss = loss_func(model(inputs), target)
        loss.backward()

        # Update parameters
        singd.step()

        if batch_idx >= MAX_STEPS:
            break
