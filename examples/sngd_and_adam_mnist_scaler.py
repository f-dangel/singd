"""Demonstrate training MNIST with SNGD and AdamW + gradient scaling."""

from torch import manual_seed
from torch.cuda.amp import GradScaler
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

from sparse_ngd.optim.optimizer import SNGD

manual_seed(0)
MAX_STEPS = 150
BATCH_SIZE = 32

###############################################################################
#                                 BASIC SETUP                                 #
###############################################################################
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
)
loss_func = CrossEntropyLoss()

# We will train parameters of convolutions, linear layers, and batch
# normalizations differently. Convolutions will be trained with ``SNGD`` and
# dense structures. Linear layers will also be trained with ``SNGD``, but using
# diagonal structures. BN layers are not supported by ``SNGD``, so we will
# train them with ``AdamW``.
conv_params = [
    p for m in model.modules() if isinstance(m, Conv2d) for p in m.parameters()
]
linear_params = [
    p for m in model.modules() if isinstance(m, Linear) for p in m.parameters()
]
bn_params = [
    p for m in model.modules() if isinstance(m, BatchNorm1d) for p in m.parameters()
]

sngd_hyperparams = {
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

conv_group = {"params": conv_params, **sngd_hyperparams}
linear_group = {"params": linear_params, **sngd_hyperparams}
linear_group["structures"] = ("diagonal", "diagonal")

param_groups = [conv_group, linear_group]
sngd = SNGD(model, params=param_groups)

adamw = AdamW(
    bn_params,
    eps=1e-8,
    betas=(0.9, 0.999),
    lr=5e-4,
    weight_decay=1e-2,
)

losses = []

scaler_sngd = GradScaler()
scaler_adamw = GradScaler()

# Loop over each batch from the training set
for batch_idx, (inputs, target) in enumerate(train_loader):
    print(f"Step {sngd.steps}")

    # Zero gradient buffers
    sngd.zero_grad()
    adamw.zero_grad()

    # Take a step
    loss = loss_func(model(inputs), target)
    losses.append(loss.item())
    scaler_sngd.scale(loss.backward())

    scaler_sngd.step(sngd)
    scaler_adamw.step(adamw)

    scaler_sngd.update()
    scaler_adamw.update()

    if batch_idx >= MAX_STEPS:
        print(losses)
        break
