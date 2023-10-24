"""Utility functions for the optimizers."""

from math import sqrt
from typing import Tuple, Union

import numpy as np
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, cat
from torch.nn import Conv2d, Linear, Module


def _extract_patches(
    x: Tensor,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int],
    padding: Union[Tuple[int, int], int],
    groups: int,
) -> Tensor:
    """Extract patches from the input of a 2d-convolution.

    The patches are averaged over channel groups.

    Args:
        x: Input to a 2d-convolution. Has shape `[batch_size, C_in, I1, I2]`.
        kernel_size: The convolution's kernel size supplied as 2-tuple or integer.
        stride: The convolution's stride supplied as 2-tuple or integer.
        padding: The convolution's padding supplied as 2-tuple or integer.
        groups: The number of channel groups.

    Returns:
        A tensor of shape `[batch_size, O1 * O2, C_in * K1 * K2]` where each column
        `[b, o1_o2, :]` contains the flattened patch of sample `b` used for output
        location `(o1, o2)`.
    """
    x_unfold = F.unfold(x, kernel_size, dilation=1, padding=padding, stride=stride)
    # separate the channel groups
    x_unfold = rearrange(
        x_unfold, "b (g c_in_k1_k2) o1_o2 -> b g c_in_k1_k2 o1_o2", g=groups
    )
    return reduce(x_unfold, "b g c_in_k1_k2 o1_o2 -> b o1_o2 c_in_k1_k2", "mean")


def process_input(input: Tensor, module: Module, kfac_approx: str) -> Tensor:
    """Unfold the input for convolutions, append ones if biases are present.

    Args:
        input: The input to the module.
        module: The module.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `'expand'` and `'reduce'`.

    Returns:
        The processed input.

    Raises:
        AssertionError: If `kfac_approx` is neither `'expand'` nor `'reduce'`.
        NotImplementedError: If the module is not supported.
    """
    assert kfac_approx in {"expand", "reduce"}
    if isinstance(module, Conv2d):
        return conv2d_process_input(input, module, kfac_approx)
    elif isinstance(module, Linear):
        return linear_process_input(input, module, kfac_approx)
    else:
        raise NotImplementedError(f"Can't process input for {module}.")


def conv2d_process_input(a: Tensor, layer: Conv2d, kfac_approx: str) -> Tensor:
    """Process the input of a convolution before the self-inner product.

    Args:
        a: Input to the convolution.
        layer: The convolution layer.
        kfac_approx: The KFAC approximation to use. Possible values are
            `'expand'` and `'reduce'`.

    Returns:
        The processed input.

    Raises:
        NotImplementedError: If the convolution uses dilation or string-valued
            padding.
    """
    # TODO Add support for dilation in `_extract_patches`
    if layer.dilation != (1, 1):
        raise NotImplementedError("Dilated convolutions are not yet supported.")
    # TODO Add support for string-valued padding in `_extract_patches`
    if isinstance(layer.padding, str):
        raise NotImplementedError(
            "String-valued padding is not yet supported (only 2-tuples)."
        )

    a = _extract_patches(
        a, layer.kernel_size, layer.stride, layer.padding, layer.groups
    )

    if kfac_approx == "expand":
        # KFAC-expand approximation
        a = rearrange(a, "b o1_o2 c_in_k1_k2 -> (b o1_o2) c_in_k1_k2")
    else:
        # KFAC-reduce approximation
        a = reduce(a, "b o1_o2 c_in_k1_k2 -> b c_in_k1_k2", "mean")

    if layer.bias is not None:
        a = cat([a, a.new_ones(a.shape[0], 1)], dim=1)

    scale = sqrt(a.shape[0])
    return a.div_(scale)


def linear_process_input(x: Tensor, layer: Linear, kfac_approx: str) -> Tensor:
    """Process the input of a linear layer before the self-inner product.

    Args:
        x: Input to the linear layer. Has shape `[batch_size, ..., d_in]` where
            `...` is an arbitrary number of weight-shared dimensions.
        layer: The linear layer.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `'expand'` and `'reduce'`.

    Returns:
        The processed input.
    """
    # NOTE Use in-place unless the below operations would alter the data of the
    # passed layer input
    in_place_okay = (x.ndim > 2 and kfac_approx == "reduce") or layer.bias is not None

    if kfac_approx == "expand":
        # KFAC-expand approximation
        x = rearrange(x, "b ... d_in -> (b ...) d_in")
    else:
        # KFAC-reduce approximation
        x = reduce(x, "b ... d_in -> b d_in", "mean")

    if layer.bias is not None:
        x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

    scale = sqrt(x.shape[0])
    return x.div_(scale) if in_place_okay else x / scale


def process_grad_output(
    grad_output: Tensor, module: Module, batch_averaged: bool, kfac_approx: str
) -> Tensor:
    """Reshape output gradients into matrices and apply scaling.

    Args:
        grad_output: The gradient w.r.t. the output of the module.
        module: The module.
        batch_averaged: Whether the loss is a mean over per-sample losses.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `'expand'` and `'reduce'`.

    Returns:
        The processed output gradient.

    Raises:
        AssertionError: If `kfac_approx` is neither `'expand'` nor `'reduce'`.
        NotImplementedError: If the module is not supported.
    """
    assert kfac_approx in {"expand", "reduce"}
    grad_scaling = 1.0
    if isinstance(module, Conv2d):
        return conv2d_process_grad_output(
            grad_output, batch_averaged, grad_scaling, kfac_approx
        )
    elif isinstance(module, Linear):
        return linear_process_grad_output(
            grad_output, batch_averaged, grad_scaling, kfac_approx
        )
    else:
        raise NotImplementedError(f"Can't process grad_output for {module}.")


def conv2d_process_grad_output(
    grad_output: Tensor, batch_averaged: bool, scaling: float, kfac_approx: str
) -> Tensor:
    """Process the output gradient of a convolution before the self-inner product.

    Args:
        grad_output: Gradient w.r.t. the output of a convolution.
        batch_averaged: Whether to multiply with the batch size.
        scaling: An additional scaling that will be applied to the gradient.
        kfac_approx: The KFAC approximation to use. Possible values are
            `'expand'` and `'reduce'`.

    Returns:
        The processed gradient.
    """
    g = grad_output
    # g: (batch_size, n_filters, out_h, out_w)
    # n_filters is actually the output dimension (analogous to Linear layer)

    batch_size = g.size(0)
    spatial_size = g.size(2) * g.size(3)  # out_h * out_w
    g = (
        g.transpose(1, 2).transpose(2, 3).reshape(batch_size, spatial_size, -1)
    )  # (batch_size, out_h * out_w, n_filters)

    if kfac_approx == "expand":
        # KFAC-expand approximation
        g = g.reshape(-1, g.size(-1))  # (batch_size * out_h * out_w, n_filters)
    else:
        # KFAC-reduce approximation
        g = g.sum(1)  # (batch_size, n_filters)

    # The scaling by `np.sqrt(batch_size)` when `batch_averaged=True` assumes
    # that we are in the reduce setting, i.e. the number of loss terms equals
    # the batch size.
    scaling = scaling * np.sqrt(batch_size) if batch_averaged else scaling
    return g * scaling


def linear_process_grad_output(
    grad_output: Tensor, batch_averaged: bool, scaling: float, kfac_approx: str
) -> Tensor:
    """Process the output gradient of a linear layer before the self-inner product.

    Args:
        grad_output: Gradient w.r.t. the output of a linear layer.
        batch_averaged: Whether to multiply with the batch size.
        scaling: An additional scaling that will be applied to the gradient.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `'expand'` and `'reduce'`.

    Returns:
        The processed gradient.
    """
    g = grad_output

    if g.ndim > 2:
        # Assumes that the first dimension is the mini-batch dimension.
        # g: (batch_size,  R_1, ...,  out_dim)
        if kfac_approx == "expand":
            # KFAC-expand approximation
            g = g.reshape(-1, g.size(-1))  # (batch_size * R_1 * ..., out_dim)
        else:
            # KFAC-reduce approximation
            weight_sharing_dims = tuple(range(1, g.ndim - 1))
            g = g.sum(weight_sharing_dims)  # (batch_size, out_dim)

    # The use of `g.size(0)` assumes that the setting of the loss, i.e. the
    # number of loss terms, matches the `kfac_approx` that is used.
    scaling = scaling * np.sqrt(g.size(0)) if batch_averaged else scaling
    return g * scaling
