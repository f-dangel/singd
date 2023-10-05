"""Demonstrate training MNIST with SINGD and AdamW."""

from singd.optim.optimizer import SINGD
from torch import cuda, device, manual_seed
from torch.nn import (
    BatchNorm1d,
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    ReLU,
    Sequential,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

manual_seed(0)  # make deterministic
MAX_STEPS = 100  # quit training after this many steps
DEV = device("cuda" if cuda.is_available() else "cpu")

BATCH_SIZE = 32
train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Sequential(
    Conv2d(1, 3, kernel_size=5, stride=2),
    ReLU(),
    Flatten(),
    Linear(432, 200),
    BatchNorm1d(200),
    Linear(200, 50),
    ReLU(),
    Linear(50, 10),
).to(DEV)
loss_func = CrossEntropyLoss().to(DEV)

# We will train parameters of convolutions, linear layers, and batch
# normalizations differently. Convolutions will be trained with ``SINGD`` and
# dense structures. Linear layers will also be trained with ``SINGD``, but using
# diagonal structures. BN layers are not supported by ``SINGD``, so we will
# train them with ``AdamW``.
conv_params = [
    p
    for m in model.modules()
    if isinstance(m, Conv2d)
    for p in m.parameters()
    if p.requires_grad
]
linear_params = [
    p
    for m in model.modules()
    if isinstance(m, Linear)
    for p in m.parameters()
    if p.requires_grad
]
ptrs = [p.data_ptr() for p in conv_params + linear_params]
other_params = [
    p for p in model.parameters() if p.data_ptr() not in ptrs and p.requires_grad
]

singd_hyperparams = {
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

# To demonstrate using multiple parameter groups, we define separate groups for
# the parameters in convolution and linear layers. For simplicity, we use the
# same hyperparameters in each group, but they could be different in practise.
conv_group = {"params": conv_params, **singd_hyperparams}
linear_group = {"params": linear_params, **singd_hyperparams}
linear_group["structures"] = ("diagonal", "diagonal")  # structure of K, C

param_groups = [conv_group, linear_group]
singd = SINGD(model, params=param_groups)

# For the other parameters, we use ``AdamW``
adamw = AdamW(
    other_params,
    eps=1e-8,
    betas=(0.9, 0.999),
    lr=5e-4,
    weight_decay=1e-2,
)

# Loop over each batch from the training set
for batch_idx, (inputs, target) in enumerate(train_loader):
    print(f"Step {singd.steps}")
    inputs, target = inputs.to(DEV), target.to(DEV)

    # Zero gradient buffers
    singd.zero_grad()
    adamw.zero_grad()

    # Backward pass
    loss = loss_func(model(inputs), target)
    loss.backward()

    # Update parameters
    singd.step()
    adamw.step()

    if batch_idx >= MAX_STEPS:
        break
