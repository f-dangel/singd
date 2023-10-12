"""# Basic usage.

This example demonstrates the simplest usage of `SINGD`. The algorithm works
pretty much like any other
[`torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html#algorithms);
but there are some additional aspects that are good to know.

First, the imports.
"""

from torch import cuda, device, manual_seed
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from singd.optim.optimizer import SINGD

manual_seed(0)  # make deterministic
MAX_STEPS = 200  # quit training after this many steps (or one epoch)
DEV = device("cuda" if cuda.is_available() else "cpu")

# %%
# ## Problem Setup
#
# We will train a simple neural network on MNIST using cross-entropy loss:

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
# One difference to many built-in PyTorch optimizers is that `SINGD` requires
# access to the neural network (a `torch.nn.Module`):

singd = SINGD(model)

# %%
#
# This is because `SINGD` needs to install hooks onto some of the neural
# network's layers to carry out the additional computations for its pre-conditioner.
#
# Of course, you can also tweak `SINGD`'s other arguments, such as learning
# rates and momenta. See [here](https://readthedocs.org/projects/singd/api/)
# for a complete overview.
#
# ## Training
#
# When it comes to training, `SINGD` can be used in **exactly** the same way as
# other optimizers (see
# [here](https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step)
# for an introduction). Let's train for a couple of steps and print the loss.

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
# ## Conclusion
#
# You now know the most basic way to train a neural network with `SINGD`. From
# here, you might be interested in
#
# - checking out the [more advanced
# examples](https://readthedocs.org/projects/singd/generated/gallery/)
#
# - taking a closer look at [`SINGD`s
# hyper-parameters](https://readthedocs.org/projects/singd/api/).
