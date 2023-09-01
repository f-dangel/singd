"""Tests ``sparse_ngd.optim.optimizer`` functionality."""


from pytest import warns
from torch.nn import BatchNorm1d, Conv2d, Flatten, Linear, ReLU, Sequential

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
