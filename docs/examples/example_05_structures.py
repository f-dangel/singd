"""# Overview of Structures.

This example visualizes the available pre-conditioner structures that can be used
in `SINGD` through the `structures` argument.

First, the imports.
"""

from math import ceil

from matplotlib import animation
from matplotlib import pyplot as plt
from torch import Tensor, manual_seed, rand
from torch.nn.functional import interpolate

from singd.optim.optimizer import SINGD

manual_seed(0)  # make deterministic

# %%
#
# ## Available Structures
#
# When constructing a `SINGD` optimizer, one can specify the pre-conditioner's
# structures through the 2-tuple
# [`structures`](https://readthedocs.org/projects/singd/api/). The first entry
# specifies the structure of $\mathbf{K}$ and its momentum
# $\mathbf{m}_\mathbf{K}$, while the second entry specifies the structure of
# $\mathbf{C}$ and its momentum $\mathbf{m}_\mathbf{C}$ (see the
# [paper](http://arxiv.org/abs/2312.05705) for details). It is even possible to specify
# structures on a per-layer basis (see
# [this](https://singd.readthedocs.io/en/latest/generated/gallery/example_03_param_groups/)
# example).
#
# The following structures are available:

available_structures = list(SINGD.SUPPORTED_STRUCTURES.keys())
print(available_structures)

# %%
#
# ## Basic Visualization
#
# The structured matrices used by `SINGD` represent structural approximations
# of dense symmetric matrices that are closed under addition and matrix
# multiplication. Let's create a dense symmetric matrix and generate its
# structural approximation with a diagonal matrix:


def rand_symmetric(dim: int) -> Tensor:
    """Create a random symmetric matrix.

    Args:
        dim: Dimension of the matrix.

    Returns:
        Random symmetric matrix of specified dimension.
    """
    dense = rand(dim, dim)
    return (dense + dense.T) / 2  # make symmetric


dim = 10
dense = rand_symmetric(dim)

name = "diagonal"
structured = SINGD.SUPPORTED_STRUCTURES[name].from_dense(dense).to_dense()

# %%
#
# Here is what they look like:

# shared limits
vmin = min(dense.min(), structured.min())
vmax = max(dense.max(), structured.max())

fig, axes = plt.subplots(1, 2)
plt.tight_layout()

for ax, structure_name, mat in zip(axes, ["original", name], [dense, structured]):
    ax.set_title(structure_name.capitalize())
    ax.set(xticks=[], yticks=[])  # turn of ticks
    ax.imshow(mat, vmin=vmin, vmax=vmax)

# %%
#
# ## Animation
#
# In the above example, we saw the diagonal structure, which is straightforward
# to understand. Other structures are more complicated and contain
# sub-structures that only emerge for large enough matrix dimensions. For
# instance, a block diagonal matrix looks exactly like the original matrix as
# long as its dimension is smaller than the block size. The block structure
# only becomes visible for larger dimensions.
#
# Here, we will thus visualize the structures for different matrix dimensions.
# Let's pre-compute the matrices and their shared axis limits:

dims = [2, 4, 8, 16, 32, 64, 128]  # dimensions to visualize
matrices = {dim: {} for dim in dims}  # stores pre-computed matrices
vmins, vmaxs = {}, {}  # limits

for dim in dims:
    # store original matrix
    dense = rand_symmetric(dim)
    matrices[dim]["original"] = dense

    # store structured approximations
    for name in available_structures:
        matrices[dim][name] = (
            SINGD.SUPPORTED_STRUCTURES[name].from_dense(dense).to_dense()
        )

    # store shared limits
    vmins[dim] = min(mat.min() for mat in matrices[dim].values())
    vmaxs[dim] = max(mat.max() for mat in matrices[dim].values())

# %%
#
# Next, we will create animations using the `ArtistAnimation` class from
# `matplotlib.animation`. Because the matrices have different dimensions, we
# need a utility function that up-samples them to the maximum dimension:


def upsample(mat: Tensor) -> Tensor:
    """Resize a matrix to the maximum dimension.

    Args:
        mat: Matrix to rescale.

    Returns:
        Resized matrix.
    """
    upsample_shape = (max(dims), max(dims))
    as_image = mat.unsqueeze(0).unsqueeze(0)
    image_upsampled = interpolate(as_image, size=upsample_shape)
    return image_upsampled.squeeze(0).squeeze(0)


# %%
#
# We will arrange the matrices on a grid with three columns. Each dimension
# will be a separate frame that plots the matrices into their respective
# sub-plot. While doing that, we need to collect a list of `Artist`s for each
# frame. Finally, we can use this nested list to create our animation
# (**Note:** You need to click onto the right triangle to play the animation):

# BASIC SETUP
img_width = 3  # size of each matrix plot
columns = 3
rows = ceil((len(available_structures) + 1) / 3)

fig, axes = plt.subplots(
    nrows=rows, ncols=columns, figsize=(columns * img_width, rows * img_width)
)
plt.tight_layout()
# turn off ticks for all axes
for ax in axes.flat:
    ax.set(xticks=[], yticks=[])

# collect all artists for animations, each sub-list is a frame
artists = []

# FRAME GENERATION
for dim in dims:
    vmin, vmax = vmins[dim], vmaxs[dim]
    artists_this_dim = []

    for name, ax in zip(["original"] + available_structures, axes.flat):
        mat_upsampled = upsample(matrices[dim][name])
        im = ax.imshow(mat_upsampled, vmin=vmin, vmax=vmax, animated=True)
        # workaround for animated title: https://stackoverflow.com/a/47421938
        title = plt.text(
            0.5,
            1.01,
            f"{name.capitalize()} (D = {dim})",
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )
        artists_this_dim.extend([im, title])

    artists.append(artists_this_dim)

# ANIMATION
ani = animation.ArtistAnimation(
    fig, artists, interval=1000, blit=True, repeat_delay=1000, repeat=True
)

# %%
#
# ## Conclusion
#
# You now know the different structural matrices that can be used in `SINGD`'s
# `structures` argument to specify the pre-conditioner's structure and have a
# visual impression how they look like.
