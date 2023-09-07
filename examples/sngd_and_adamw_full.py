"""Demonstrate training MNIST with SNGD and AdamW + bells and whistles.

Uses the following bells and whistles (relevant parts in the code are tagged):

- [SCL] gradient scaling
- [AMP] mixed-precision operations with autocast
- [LR] learning rate scheduler
- [ACC] micro batches (gradient accumulation)
"""

from torch import autocast, bfloat16, cuda, device, manual_seed
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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from sparse_ngd.optim.optimizer import SNGD

manual_seed(0)  # make deterministic
MAX_STEPS = 100  # quit training after this many steps
DEV = device("cuda" if cuda.is_available() else "cpu")

BATCH_SIZE = 32

MICRO_BATCH_SIZE = 8  # [ACC]
assert BATCH_SIZE % MICRO_BATCH_SIZE == 0  # [ACC]

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
# normalizations differently. Convolutions will be trained with ``SNGD`` and
# dense structures. Linear layers will also be trained with ``SNGD``, but using
# diagonal structures. BN layers are not supported by ``SNGD``, so we will
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

# To demonstrate using multiple parameter groups, we define separate groups for
# the parameters in convolution and linear layers. For simplicity, we use the
# same hyperparameters in each group, but they could be different in practise.
conv_group = {"params": conv_params, **sngd_hyperparams}
linear_group = {"params": linear_params, **sngd_hyperparams}
linear_group["structures"] = ("diagonal", "diagonal")

param_groups = [conv_group, linear_group]
sngd = SNGD(model, params=param_groups)

adamw = AdamW(
    other_params,
    eps=1e-8,
    betas=(0.9, 0.999),
    lr=5e-4,
    weight_decay=1e-2,
)

# [SCL] We need one scaler per optimizer, as each will handle the ``.grad``s of
# the parameters in its optimizer (THEY NEED TO BE IDENTICAL!)
scaler_sngd = GradScaler()  # [SCL]
scaler_adamw = GradScaler()  # [SCL]

# [LR] We need one learning rate scheduler per optimizer (THEY CAN BE DIFFERENT)
scheduler_sngd = ExponentialLR(sngd, gamma=0.999)  # [LR]
scheduler_adamw = ExponentialLR(sngd, gamma=0.999)  # [LR]

# [AMP] Determine device and data type for autocast
amp_device_type = "cuda" if "cuda" in str(DEV) else "cpu"  # [AMP]
amp_dtype = bfloat16  # [AMP]

# Loop over each batch from the training set
for batch_idx, (inputs, target) in enumerate(train_loader):
    print(f"Step {sngd.steps}")

    inputs, target = inputs.to(DEV), target.to(DEV)

    # [ACC] split mini-batch into micro-batches
    inputs_split = inputs.split(MICRO_BATCH_SIZE)  # [ACC]
    target_split = target.split(MICRO_BATCH_SIZE)  # [ACC]

    # Zero gradient buffers
    sngd.zero_grad()
    adamw.zero_grad()

    # Backward pass
    # [ACC] Loop over micro-batches
    for inputs_micro, target_micro in zip(inputs_split, target_split):  # [ACC]
        with autocast(device_type=amp_device_type, dtype=amp_dtype):  # [AMP]
            loss = loss_func(model(inputs_micro), target_micro)
            loss = scaler_sngd.scale(loss)  # [SCL]
            loss.backward()

    # [SCL] Re-scale gradients and update parameters
    scaler_sngd.step(sngd)  # [SCL]
    scaler_adamw.step(adamw)  # [SCL]

    # [SCL] Update gradient scale for next iteration
    scaler_sngd.update()  # [SCL]
    scaler_adamw.update()  # [SCL]

    # [LR] Update learning rate schedule
    scheduler_sngd.step()  # [LR]
    scheduler_adamw.step()  # [LR]

    if batch_idx >= MAX_STEPS:
        break
