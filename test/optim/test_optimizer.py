"""Tests ``sparse_ngd.optim.optimizer`` functionality."""


from test.optim.utils import check_preconditioner_dtypes
from test.utils import DEVICE_IDS, DEVICES
from typing import Tuple, Union

from pytest import mark, raises, warns
from torch import autocast, bfloat16, device, dtype, float16, float32, manual_seed
from torch.nn import (
    BatchNorm1d,
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    ReLU,
    Sequential,
)
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD


def test_SNGD_get_trainable_params_warning():
    """Test emission of warning if a model contains non-trainable parameters."""
    # NOTE This module contains a nested ``Sequential`` to make it 'challenging'
    # to iterate correctly over all layers and find the un-trainable batch
    # normalization layer in the inner sequence.
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Sequential(
            ReLU(),
            Linear(432, 50),
            ReLU(),
            # NOTE: BatchNorm cannot be trained
            BatchNorm1d(50),
            Linear(50, 10),
        ),
    )
    dummy_hyperparams = {
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
    with warns(UserWarning):
        SNGD(model, **dummy_hyperparams, warn_unsupported=True)


def test_SNGD_check_param_groups():
    """Test conflict detection in parameter groups."""
    layer1 = Linear(10, 9)
    layer2 = Linear(9, 8)
    model = Sequential(layer1, layer2)
    dummy_hyperparams = {
        "lr": 5e-4,
        "damping": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-2,
        "lr_cov": 1e-2,
        "batch_averaged": True,
        "T": 1,
        "alpha1": 0.5,
        "structures": ("dense", "dense"),
        "warn_unsupported": True,
    }

    # parameters in same layer are in different groups
    param_groups = [
        {"params": [layer1.weight]},
        {"params": [layer1.bias]},
    ]

    with raises(ValueError):
        SNGD(model, params=param_groups, **dummy_hyperparams)

    # per-layer groups with different structures
    group1 = {"params": [layer1.weight, layer1.bias], "structures": ("dense", "dense")}
    group2 = {
        "params": [layer2.weight, layer2.bias],
        "structures": ("diagonal", "diagonal"),
    }
    param_groups = [group1, group2]
    optim = SNGD(model, params=param_groups, **dummy_hyperparams)

    assert optim.param_groups[0]["structures"] == ("dense", "dense")
    assert optim.param_groups[1]["structures"] == ("diagonal", "diagonal")

    # warn user if kfac_like is turned on and alpha1≠0
    with warns(UserWarning):
        optim = SNGD(model, **dummy_hyperparams, kfac_like=True)


PRECONDITIONER_DTYPES = [
    (None, None),
    (float16, float16),
    (bfloat16, bfloat16),
    (float32, float32),
]
PRECONDITIONER_DTYPE_IDS = [
    f"{'-'.join([str(d) for d in dtypes])}".replace("torch.", "")
    for dtypes in PRECONDITIONER_DTYPES
]
STRUCTURES = [("dense", "dense"), ("diagonal", "diagonal")]
STRUCTURE_IDS = [f"{'-'.join(structures)}" for structures in STRUCTURES]


@mark.parametrize("amp", [False, True], ids=["no-amp", "amp"])
@mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
@mark.parametrize("structures", STRUCTURES, ids=STRUCTURE_IDS)
@mark.parametrize(
    "preconditioner_dtype", PRECONDITIONER_DTYPES, ids=PRECONDITIONER_DTYPE_IDS
)
def test_SNGD_preconditioner_dtype(
    structures: Tuple[str, str],
    preconditioner_dtype: Tuple[Union[None, dtype], Union[None, dtype]],
    dev: device,
    amp: bool,
):
    """Check preconditioner data types remain identical during training.

    Args:
        structures: The structures to use.
        preconditioner_dtype: The preconditioner data types to use.
        dev: The device to use.
        amp: Whether to use ``autocast``.
    """
    manual_seed(0)
    MAX_STEPS = 5

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
    ).to(dev)
    loss_func = CrossEntropyLoss().to(dev)

    optim = SNGD(
        model, T=1, structures=structures, preconditioner_dtype=preconditioner_dtype
    )
    # check that pre-conditioner matrices have correct data type at init
    check_preconditioner_dtypes(optim)

    model.train()

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(dev), target.to(dev)
        print(f"Step {optim.steps}")

        optim.zero_grad()
        if amp:
            if str(dev) == "cpu":
                amp_ctx = autocast(device_type="cpu", dtype=bfloat16)
            else:
                amp_ctx = autocast(device_type="cuda", dtype=float16)
            with amp_ctx:
                output = model(inputs)
        else:
            output = model(inputs)

        loss = loss_func(output, target)
        loss.backward()
        optim.step()

        # check that dtype of pre-conditioner remains the same
        check_preconditioner_dtypes(optim)

        if batch_idx >= MAX_STEPS:
            break
