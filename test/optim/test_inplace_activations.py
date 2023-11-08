"""SINGD with a model that uses in-place activations."""

from copy import deepcopy
from test.utils import REDUCTION_IDS, REDUCTIONS, compare_optimizers

from pytest import mark, skip
from torch import manual_seed, rand
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD


@mark.parametrize("inplace", [True, False], ids=["inplace=True", "inplace=False"])
def test_hooks_support_inplace_activations(inplace: bool):
    """Test that SINGD's hooks support in in-place activations.

    See https://github.com/f-dangel/singd/issues/56.

    Args:
        inplace: Whether to use in-place activations.
    """
    manual_seed(0)

    X = rand(2, 1, 5, 5)
    model = Sequential(Conv2d(1, 1, 3), ReLU(inplace=inplace))
    SINGD(model)  # install hooks

    model(X)


@mark.parametrize("reduction", REDUCTIONS, ids=REDUCTION_IDS)
def test_compare_training_using_inplace_activations(reduction: str):
    """Compare training w/o in-place activations.

    Args:
        reduction: Reduction used for the loss function.
    """
    if reduction == "sum":
        skip("Need to fix https://github.com/f-dangel/singd/issues/43 first.")

    manual_seed(0)
    MAX_STEPS = 100
    batch_size = 32

    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # _inplace indicates that the trained net has in-place activations

    # NOTE All parameters of this net are supported by SINGD
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_inplace = deepcopy(model)
    # activate in-place option
    for mod in model_inplace.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = True

    loss_func = CrossEntropyLoss(reduction=reduction)
    loss_func_inplace = deepcopy(loss_func)

    loss_average = {"mean": "batch", "sum": None}[reduction]
    optim_hyperparams = {
        "lr": 5e-4,
        "damping": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_cov": 1e-2,
        "loss_average": loss_average,
        "T": 1,
        "alpha1": 0.5,
        "structures": ("dense", "dense"),
    }

    optim = SINGD(model, **optim_hyperparams)
    optim_inplace = SINGD(model_inplace, **optim_hyperparams)

    model.train()
    model_inplace.train()

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim.steps}")

        # Zero gradient buffers
        optim.zero_grad()
        optim_inplace.zero_grad()

        # Take a step
        loss_func(model(inputs), target).backward()
        optim.step()

        loss_func_inplace(model_inplace(inputs), target).backward()
        optim_inplace.step()

        compare_optimizers(optim, optim_inplace, rtol=1e-5, atol=1e-7)

        if batch_idx >= MAX_STEPS:
            break
