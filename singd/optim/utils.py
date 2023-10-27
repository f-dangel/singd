"""Utility functions for the optimizers."""

from math import sqrt
from typing import Tuple, Union

import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, cat
from torch.nn import Conv2d, Linear, Module


def _extract_patches(
    x: Tensor,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int],
    padding: Union[Tuple[int, int], int],
    dilation: Union[Tuple[int, int], int],
    groups: int,
) -> Tensor:
    """Extract patches from the input of a 2d-convolution.

    The patches are averaged over channel groups.

    Args:
        x: Input to a 2d-convolution. Has shape `[batch_size, C_in, I1, I2]`.
        kernel_size: The convolution's kernel size supplied as 2-tuple or integer.
        stride: The convolution's stride supplied as 2-tuple or integer.
        padding: The convolution's padding supplied as 2-tuple or integer.
        dilation: The convolution's dilation supplied as 2-tuple or integer.
        groups: The number of channel groups.

    Returns:
        A tensor of shape `[batch_size, O1 * O2, C_in // groups * K1 * K2]` where
        each column `[b, o1_o2, :]` contains the flattened patch of sample `b` used
        for output location `(o1, o2)`, averaged over channel groups.
    """
    x_unfold = F.unfold(
        x, kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    # separate the channel groups
    x_unfold = rearrange(
        x_unfold, "b (g c_in_k1_k2) o1_o2 -> b g c_in_k1_k2 o1_o2", g=groups
    )
    return reduce(x_unfold, "b g c_in_k1_k2 o1_o2 -> b o1_o2 c_in_k1_k2", "mean")


def process_input(x: Tensor, module: Module, kfac_approx: str) -> Tensor:
    """Unfold the input for convolutions, append ones if biases are present.

    Args:
        x: The input to the module.
        module: The module.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `"expand"` and `"reduce"`.

    Returns:
        The processed input.

    Raises:
        AssertionError: If `kfac_approx` is neither `"expand"` nor `"reduce"`.
        NotImplementedError: If the module is not supported.
    """
    assert kfac_approx in {"expand", "reduce"}
    if isinstance(module, Conv2d):
        return conv2d_process_input(x, module, kfac_approx)
    elif isinstance(module, Linear):
        return linear_process_input(x, module, kfac_approx)
    else:
        raise NotImplementedError(f"Can't process input for {module}.")


def conv2d_process_input(x: Tensor, layer: Conv2d, kfac_approx: str) -> Tensor:
    """Process the input of a convolution before the self-inner product.

    Args:
        x: Input to the convolution. Has shape `[batch_size, C_in, I1, I2]`.
        layer: The convolution layer.
        kfac_approx: The KFAC approximation to use. Possible values are
            `"expand"` and `"reduce"`.

    Returns:
        The processed input. Has shape
        `[batch_size, O1 * O2, C_in // groups * K1 * K2 (+ 1)]` for `"reduce"` and
        `[batch_size * O1 * O2, C_in // groups * K1 * K2 (+ 1)]` for `"expand"`.
        The `+1` is active if the layer has a bias.

    Raises:
        NotImplementedError: If the convolution uses string-valued padding.
    """
    # TODO Add support for string-valued padding in `_extract_patches`
    if isinstance(layer.padding, str):
        raise NotImplementedError(
            "String-valued padding is not yet supported (only 2-tuples)."
        )

    x = _extract_patches(
        x, layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.groups
    )

    if kfac_approx == "expand":
        # KFAC-expand approximation
        x = rearrange(x, "b o1_o2 c_in_k1_k2 -> (b o1_o2) c_in_k1_k2")
    else:
        # KFAC-reduce approximation
        x = reduce(x, "b o1_o2 c_in_k1_k2 -> b c_in_k1_k2", "mean")

    if layer.bias is not None:
        x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

    scale = sqrt(x.shape[0])
    return x.div_(scale)


def linear_process_input(x: Tensor, layer: Linear, kfac_approx: str) -> Tensor:
    """Process the input of a linear layer before the self-inner product.

    Args:
        x: Input to the linear layer. Has shape `[batch_size, ..., d_in]` where
            `...` is an arbitrary number of weight-shared dimensions.
        layer: The linear layer.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `"expand"` and `"reduce"`.

    Returns:
        The processed input. Has shape `[batch_size, d_in (+ 1)]` for
        `"reduce"` and `[batch_size * ..., d_in (+ 1)]` for `"reduce"`.
        The `+1` is active if the layer has a bias.
    """
    if kfac_approx == "expand":
        # KFAC-expand approximation
        x = rearrange(x, "b ... d_in -> (b ...) d_in")
    else:
        # KFAC-reduce approximation
        x = reduce(x, "b ... d_in -> b d_in", "mean")

    if layer.bias is not None:
        x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

    scale = sqrt(x.shape[0])
    return x / scale


def process_grad_output(
    grad_output: Tensor, module: Module, batch_averaged: bool, kfac_approx: str
) -> Tensor:
    """Reshape output gradients into matrices and apply scaling.

    Args:
        grad_output: The gradient w.r.t. the output of the module.
        module: The module.
        batch_averaged: Whether the loss is a mean over per-sample losses.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `"expand"` and `"reduce"`.

    Returns:
        The processed output gradient.

    Raises:
        AssertionError: If `kfac_approx` is neither `"expand"` nor `"reduce"`.
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
    g: Tensor, batch_averaged: bool, scaling: float, kfac_approx: str
) -> Tensor:
    """Process the output gradient of a convolution before the self-inner product.

    Args:
        g: Gradient w.r.t. the output of a convolution. Has shape
            `[batch_size, C_out, O1, O2]`.
        batch_averaged: Whether to multiply with the batch size.
        scaling: An additional scaling that will be applied to the gradient.
        kfac_approx: The KFAC approximation to use. Possible values are
            `"expand"` and `"reduce"`.

    Returns:
        The processed scaled gradient. Has shape `[batch_size, C_out]` for
        `"reduce"` and `[batch_size * O1 * O2, C_out]` for `"expand"`.
    """
    # The scaling by `sqrt(batch_size)` when `batch_averaged=True` assumes
    # that we are in the reduce setting, i.e. the number of loss terms equals
    # the batch size.
    batch_size = g.shape[0]
    scaling = scaling * sqrt(batch_size) if batch_averaged else scaling

    if kfac_approx == "expand":
        # KFAC-expand approximation
        g = rearrange(g, "b c o1 o2 -> (b o1 o2) c")
    else:
        # KFAC-reduce approximation
        g = reduce(g, "b c o1 o2 -> b c", "sum")

    return g * scaling


def linear_process_grad_output(
    g: Tensor, batch_averaged: bool, scaling: float, kfac_approx: str
) -> Tensor:
    """Process the output gradient of a linear layer before the self-inner product.

    Args:
        g: Gradient w.r.t. the output of a linear layer. Has shape
            `[batch_size, ..., d_out]` where `...` is an arbitrary number of
            weight-shared dimensions.
        batch_averaged: Whether to multiply with the batch size.
        scaling: An additional scaling that will be applied to the gradient.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `"expand"` and `"reduce"`.

    Returns:
        The processed gradient. Has shape `[batch_size, d_out]` for `"reduce"`
        and `[batch_size * ..., d_out]` for `"expand"`.
    """
    if kfac_approx == "expand":
        # KFAC-expand approximation
        g = rearrange(g, "b ... d_out -> (b ...) d_out")
    else:
        # KFAC-reduce approximation
        g = reduce(g, "b ... d_out -> b d_out", "sum")

    # The use of `g.shape[0]` assumes that the setting of the loss, i.e. the
    # number of loss terms, matches the `kfac_approx` that is used.
    scaling = scaling * sqrt(g.shape[0]) if batch_averaged else scaling
    return g * scaling
