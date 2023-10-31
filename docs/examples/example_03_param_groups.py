"""# Per-parameter Options.

Here we demonstrate how to initialize `SINGD` from [parameter
groups](https://pytorch.org/docs/stable/optim.html#per-parameter-options) which
allow training parameters of a neural network differently. We will demonstrate
this by taking a CNN and training the parameters in the linear layers
differently than those of the convolutional layers.

First, the imports.
"""

from torch import cuda, device, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD
from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix

manual_seed(0)  # make deterministic
MAX_STEPS = 200  # quit training after this many steps (or one epoch)
DEV = device("cuda" if cuda.is_available() else "cpu")

# %%
# ## Problem Setup
#
# Next, we load the data set, define the neural network, and the loss
# function:

BATCH_SIZE = 32
train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Sequential(
    Conv2d(1, 3, kernel_size=5, stride=2),
    ReLU(),
    Flatten(),
    Linear(432, 50),
    ReLU(),
    Linear(50, 10),
).to(DEV)
loss_func = CrossEntropyLoss().to(DEV)

# %%
# ## Optimizer Setup
#
# As mentioned above, we will train parameters of convolutions different than
# those in linear layers. We will do so by specifying two groups, and passing
# them to the optimizer via `param_groups`.
#
# Specifically, we will use a dense pre-conditioner for convolutions, and a
# diagonal pre-conditioner for linear layers. All other hyper-parameters are
# identical, but they could be different:

singd_hyperparams = {
    "lr": 5e-4,
    "damping": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-2,
    "lr_cov": 1e-2,
    "loss_average": "batch",
    "T": 1,
    "alpha1": 0.5,
}

# %%
#
# As a first step, we identify each group's parameters:

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

# %%
#
# We are now ready to set up the two groups:
conv_group = {
    "params": conv_params,
    "structures": ("dense", "dense"),
    **singd_hyperparams,
}
linear_group = {
    "params": linear_params,
    "structures": ("diagonal", "diagonal"),
    **singd_hyperparams,
}

# %%
#
# The `param_groups` are just a list containing the groups. We can pass it to
# the optimizer's `params` argument:

param_groups = [conv_group, linear_group]
singd = SINGD(model, params=param_groups)

# %%
#
# That's everything. What follows is just a canonical training loop.
#
# ## Training
#
# Let's train for a couple of steps and print the loss. SINGD works like most
# other PyTorch optimizers:

PRINT_LOSS_EVERY = 25  # logging interval

for step, (inputs, target) in enumerate(train_loader):
    singd.zero_grad()  # clear gradients from previous iterations

    # regular forward-backward pass
    loss = loss_func(model(inputs.to(DEV)), target.to(DEV))
    loss.backward()
    if step % PRINT_LOSS_EVERY == 0:
        print(f"Step {step}, Loss {loss.item():.3f}")

    singd.step()  # update neural network parameters

    if step >= MAX_STEPS:  # don't train a full epoch to keep the example light-weight
        break

# %%
#
# ## Pre-conditioner Inspection
#
# We can also verify that convolutions and linear layers indeed use different
# pre-conditioner structures:

for name, module in model.named_modules():
    # print the pre-conditioner matrix types of all trained layers
    if name in singd.module_names.values():
        module_cls = module.__class__
        preconditioner_cls = singd.Ks[name].__class__
        print(f"{name} {module_cls.__name__}: {preconditioner_cls.__name__}")

    # make sure convolutions use `DenseMatrix`s, linear layers use `DiagonalMatrix`s
    if isinstance(module, Linear):
        assert isinstance(singd.Ks[name], DiagonalMatrix)
    elif isinstance(module, Conv2d):
        assert isinstance(singd.Ks[name], DenseMatrix)

# %%
#
# ## Conclusion
#
# Congratulations! You now know how to train each layer of a neural network
# differently with `SINGD`.
#
# For example, this may be useful when the network has layers with large
# pre-conditioner dimensions. One way to reduce cost would be to use a more
# light-weight pre-conditioner type (like `DiagonalMatrix`) for such layers.
# But of course you can also use this to tweak learning rates, momenta, etc.
# per layer.
