"""Test mixed-precision training with float16."""

from copy import deepcopy
from test.utils import compare_optimizers, report_nonclose

from torch import autocast, bfloat16, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD


def test_autocast():
    """Compare optimizer in float32 and optimizer with mixed-precision (float16)."""
    manual_seed(0)
    MAX_STEPS = 8  # NOTE small as the optimizers diverge quickly

    mini_batch_size = 32

    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=mini_batch_size, shuffle=True
    )

    # _single indicates the float32 version
    # _mixed indicates the mixed precision version

    # NOTE All parameters of this network will be updated with KFAC, no other
    # optimizer involved
    model_single = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    )
    model_mixed = deepcopy(model_single)

    loss_func_single = CrossEntropyLoss()
    loss_func_mixed = deepcopy(loss_func_single)

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

    optim_single = SNGD(model_single, **optim_hyperparams)
    optim_mixed = SNGD(model_mixed, **optim_hyperparams)

    model_single.train()
    model_mixed.train()

    # Loop over each batch from the training set
    for batch_idx, (inputs, target) in enumerate(train_loader):
        print(f"Step {optim_single.steps}")

        # Zero gradient buffers
        optim_single.zero_grad()
        optim_mixed.zero_grad()

        # Take a step
        loss_func_single(model_single(inputs), target).backward()
        optim_single.step()

        # NOTE This is NOT how you would use gradient scaling.
        # It serves for testing purposes because ``GradientScaler`` only
        # works with CUDA and we want the test to run on CPU.
        GRAD_SCALE = 10_000
        optim_mixed.grad_scale = GRAD_SCALE

        with autocast(device_type="cpu", dtype=bfloat16):
            output_mixed = model_mixed(inputs)
            assert output_mixed.dtype == bfloat16  # due to linear layers
            loss_mixed = loss_func_mixed(output_mixed, target)
            (GRAD_SCALE * loss_mixed).backward()
        optim_mixed.step()

        compare_optimizers(
            optim_single,
            optim_mixed,
            atol=5e-5,
            rtol=1e-2,
            # momentum requires larger tolerance
            atol_momentum=5e-3,
            rtol_momentum=1e-1,
        )

        if batch_idx >= MAX_STEPS:
            break
