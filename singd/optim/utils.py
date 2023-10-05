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


def process_input(input: Tensor, module: Module) -> Tensor:
    """Unfold the input for convolutions, append ones if biases are present.

    Args:
        input: The input to the module.
        module: The module.

    Returns:
        The processed input.

    Raises:
        NotImplementedError: If the module is not supported.
    """
    if isinstance(module, Conv2d):
        return conv2d_process_input(input, module)
    elif isinstance(module, Linear):
        return linear_process_input(input, module)
    else:
        raise NotImplementedError(f"Can't process input for {module}")


def conv2d_process_input(input: Tensor, layer: Conv2d) -> Tensor:
    """Process the input of a convolution before the self-inner product.

    Args:
        input: Input to the convolution.
        layer: The convolution layer.

    Returns:
        The processed input.
    """
    # TODO Improve readability
    a = input

    batch_size = a.size(0)
    a = _extract_patches(
        a,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        layer.groups,
    )
    spatial_size = a.size(1) * a.size(2)
    a = a.reshape(-1, a.size(-1))  # (batch_size*out_h*out_w, nfilters)
    if layer.bias is not None:
        shape = list(a.shape[:-1]) + [1]  # new try
        a = torch.cat([a, a.new_ones(shape)], dim=-1)  # new try

    a = a / spatial_size
    a = a / np.sqrt(batch_size)  # new try2

    return a


def linear_process_input(input: Tensor, layer: Linear) -> Tensor:
    """Process the input of a linear layer before the self-inner product.

    Args:
        input: Input to the linear layer.
        layer: The linear layer.

    Returns:
        The processed input.
    """
    # TODO Improve readability
    a = input

    # mode = 'reduce'
    mode = "expand"

    wt = 1.0
    batch_size = a.size(0)
    if a.ndim == 2:
        # a: batch_size * in_dim
        # b = a.reshape(-1, a.size(-1)) #new try
        b = a
    elif a.ndim == 3:
        # print( 'a shape:', a.ndim, a.size() )
        # a: batch_size *  R  * in_dim
        if mode == "reduce":
            b = a.mean(dim=1)  # reduce case
        else:
            wt = 1.0 / np.sqrt(a.size(1))  # expand case
            b = a.reshape(-1, a.size(-1)) / np.sqrt(a.size(1))  # expand case
    else:
        # print( 'a shape:', a.ndim, a.size() )
        wt = 1.0 / np.sqrt(np.prod(a.shape[1:-1]))  # expand case
        b = a.reshape(-1, a.size(-1)) / np.sqrt(np.prod(a.shape[1:-1]))  # expand case
        # raise NotImplementedError

    if layer.bias is not None:
        b = torch.cat([b, b.new(b.size(0), 1).fill_(wt)], 1)

        # shape = list(b.shape[:-1]) + [1] #new try
        # b = torch.cat([b, b.new_ones(shape)], dim=-1)#new try

    # return b.t() @ (b / batch_size) #new try
    b = b / np.sqrt(batch_size)

    return b


def process_grad_output(
    grad_output: Tensor, module: Module, batch_averaged: bool
) -> Tensor:
    """Reshape output gradients into matrices and apply scaling.

    Args:
        grad_output: The gradient w.r.t. the output of the module.
        module: The module.
        batch_averaged: Whether the loss is a mean over per-sample losses.

    Returns:
        The processed output gradient.

    Raises:
        NotImplementedError: If the module is not supported.
    """
    grad_scaling = 1.0
    if isinstance(module, Conv2d):
        return conv2d_process_grad_output(grad_output, batch_averaged, grad_scaling)
    elif isinstance(module, Linear):
        return linear_process_grad_output(grad_output, batch_averaged, grad_scaling)
    else:
        raise NotImplementedError(f"Can't process grad_output for {module}")


def conv2d_process_grad_output(
    grad_output: Tensor, batch_averaged: bool, scaling: float
) -> Tensor:
    """Process the output gradient of a convolution before the self-inner product.

    Args:
        grad_output: Gradient w.r.t. the output of a convolution.
        batch_averaged: Whether to multiply with the batch size.
        scaling: An additional scaling that will be applied to the gradient.

    Returns:
        The processed gradient.
    """
    # TODO Improve readability
    g = grad_output
    # g: batch_size * n_filters * out_h * out_w
    # n_filters is actually the output dimension (analogous to Linear layer)

    spatial_size = g.size(2) * g.size(3)  # out_h*out_w
    batch_size = g.shape[0]
    g = g.transpose(1, 2).transpose(2, 3)  # batch_size, out_h, out_w, n_filters
    g = g.reshape(-1, g.size(-1))  # (batch_size*out_h*out_w, n_filters)

    if batch_averaged:
        g = g * batch_size
    g = g * spatial_size

    return g * (scaling / np.sqrt(g.size(0)))


def linear_process_grad_output(
    grad_output: Tensor, batch_averaged: bool, scaling: float
) -> Tensor:
    """Process the output gradient of a linear layer before the self-inner product.

    Args:
        grad_output: Gradient w.r.t. the output of a linear layer.
        batch_averaged: Whether to multiply with the batch size.
        scaling: An additional scaling that will be applied to the gradient.

    Returns:
        The processed gradient.
    """
    # TODO Improve readability
    g = grad_output

    # mode='reduce'
    mode = "expand"
    batch_size = g.size(0)

    if g.ndim == 2:
        # g = g.reshape(-1, g.size(-1)) #new try
        b = g
    elif g.ndim == 3:
        # print( 'g shape:', g.ndim, g.size() )
        # g: batch_size *  R  * in_dim

        if mode == "reduce":
            b = g.sum(dim=1)  # reduce case
            # b = g.sum(dim=1)/np.sqrt(g.size(1)) #modified reduce case
        else:
            b = g.reshape(-1, g.size(-1))  # expand
    else:
        # print( 'g shape:', g.ndim, g.size() )
        # raise NotImplementedError
        b = g.reshape(-1, g.size(-1))  # expand

    if batch_averaged:
        b = b * (scaling * np.sqrt(batch_size))
    else:
        b = b * (scaling / np.sqrt(batch_size))

    return b
