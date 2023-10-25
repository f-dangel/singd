"""Utility functions for the optimizers."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Linear, Module


def _extract_patches(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    groups: int,
) -> Tensor:
    """Extract patches from the input of a 2d-convolution.

    Args:
        x: The input feature maps. (batch_size, in_c, h, w)
        kernel_size: the kernel sizes of the conv filter (tuple of two elements).
        stride: the stride of conv operation (tuple of two elements).
        padding: number of paddings. be a tuple of two elements..
        groups: number of groups.

    Returns:
        (batch_size, out_h, out_w, in_c*kh*kw)
    """
    # TODO Make human-readable
    if padding[0] + padding[1] > 0:
        x = F.pad(
            x, (padding[1], padding[1], padding[0], padding[0])
        ).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3)
    return torch.mean(
        x.reshape((x.size(0), x.size(1), x.size(2), groups, -1, x.size(4), x.size(5))),
        3,
    ).view(x.size(0), x.size(1), x.size(2), -1)


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
    assert kfac_approx in ["expand", "reduce"]
    if isinstance(module, Conv2d):
        return conv2d_process_input(input, module, kfac_approx)
    elif isinstance(module, Linear):
        return linear_process_input(input, module, kfac_approx)
    else:
        raise NotImplementedError(f"Can't process input for {module}.")


def conv2d_process_input(input: Tensor, layer: Conv2d, kfac_approx: str) -> Tensor:
    """Process the input of a convolution before the self-inner product.

    Args:
        input: Input to the convolution.
        layer: The convolution layer.
        kfac_approx: The KFAC approximation to use. Possible values are
            `'expand'` and `'reduce'`.

    Returns:
        The processed input.
    """
    a = input
    a = _extract_patches(
        a,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        layer.groups,
    )

    batch_size = a.size(0)
    spatial_size = a.size(1) * a.size(2)
    scale = np.sqrt(batch_size)
    a = a.view(batch_size, spatial_size, -1)  # (batch_size, out_h * out_w, nfilters)

    if kfac_approx == "expand":
        # KFAC-expand approximation
        scale *= np.sqrt(spatial_size)
        a = a.view(-1, a.size(-1))  # (batch_size * out_h * out_w, nfilters)
    else:
        # KFAC-reduce approximation
        a = a.mean(1)  # (batch_size, nfilters)

    if layer.bias is not None:
        a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)

    return a / scale


def linear_process_input(input: Tensor, layer: Linear, kfac_approx: str) -> Tensor:
    """Process the input of a linear layer before the self-inner product.

    Args:
        input: Input to the linear layer.
        layer: The linear layer.
        kfac_approx: The KFAC approximation to use for linear weight-sharing
            layers. Possible values are `'expand'` and `'reduce'`.

    Returns:
        The processed input.
    """
    a = input
    # Assumes that the first dimension is the mini-batch dimension.
    scale = np.sqrt(a.size(0))  # sqrt(batch_size)

    if a.ndim > 2:
        # a: (batch_size,  R_1, ..., in_dim)
        weight_sharing_scale = np.prod(a.shape[1:-1])
        if kfac_approx == "expand":
            # KFAC-expand approximation
            scale *= np.sqrt(weight_sharing_scale)
            a = a.reshape(-1, a.size(-1))  # (batch_size * R_1 * ..., in_dim)
        else:
            # KFAC-reduce approximation
            weight_sharing_dims = tuple(range(1, a.ndim - 1))
            a = a.mean(weight_sharing_dims)  # (batch_size, in_dim)

    if layer.bias is not None:
        a = torch.cat([a, a.new_ones(a.size(0), 1)], 1)

    return a / scale


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
    assert kfac_approx in ["expand", "reduce"]
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
