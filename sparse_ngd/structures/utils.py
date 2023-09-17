"""Utility functions for the structured matrices."""

from typing import Any

import torch
from torch import (
    Tensor,
    arange,
    bfloat16,
    device,
    einsum,
    eye,
    float16,
    float32,
    get_default_dtype,
    zeros,
)


def is_half_precision(dtype: torch.dtype) -> bool:
    """Check if the given dtype is half precision.

    Args:
        dtype: The dtype to check.

    Returns:
        Whether the given dtype is half precision.
    """
    return dtype in [float16, bfloat16]


def supported_matmul(*matrices: Tensor) -> Tensor:
    """Multiply matrices with the same or higher numerical precision.

    If the matrix multiplication is not supported on the hardware,
    carry out the multiplication in single precision.

    Args:
        matrices: The matrices to multiply.

    Returns:
        The result of the matrix chain multiplication in the original precision.

    Raises:
        RuntimeError: If the matrices are not on the same device.
    """
    devices = {m.device for m in matrices}
    if len(devices) > 1:
        raise RuntimeError("Matrices must be on the same device.")
    dev = devices.pop()

    # Use the first matrix's data type as the result's data type.
    # The matrices may have different data types if ``autocast`` was used.
    dtype = matrices[0].dtype

    # @ not supported on CPU for float16 (bfloat16 is supported)
    convert = dtype == float16 and str(dev) == "cpu"

    result = matrices[0].to(float32) if convert else matrices[0]
    for mat in matrices[1:]:
        result = result @ mat.to(result.dtype)

    return result.to(dtype) if convert else result


def supported_eye(n: int, **kwargs: Any) -> Tensor:
    """Same as PyTorch's ``eye``, but uses higher precision if unsupported.

    Args:
        n: The number of rows.
        kwargs: Keyword arguments to ``torch.eye``.

    Returns:
        A 2-D tensor with ones on the diagonal and zeros elsewhere.
    """
    dtype = kwargs.pop("dtype", get_default_dtype())
    # TODO Figure out how to obtain the default device
    default_device = device("cpu")
    dev = kwargs.get("device", default_device)

    # eye not supported on CPU for bfloat16 (float16 is supported)
    if dtype == bfloat16 and str(dev) == "cpu":
        return eye(n, **kwargs, dtype=float32).to(dtype)
    else:
        return eye(n, **kwargs, dtype=dtype)


def supported_trace(input: Tensor) -> Tensor:
    """Same as PyTorch's ``trace``, but uses higher precision if unsupported.

    Args:
        input: The input tensor.

    Returns:
        The sum of the tensor's diagonal elements.
    """
    # eye not supported on CPU for half precision
    if is_half_precision(input.dtype) and str(input.device) == "cpu":
        original = input.dtype
        return input.to(float32).trace().to(original)
    else:
        return input.trace()


def supported_einsum(equation, *operands: Tensor) -> Tensor:
    """Compute an ``einsum`` with the same or higher numerical precision.

    If the ``einsum`` is not supported on the hardware,
    carry out the multiplication in single precision.

    Args:
        equation: The ``einsum`` equation.
        operands: The operands to the ``einsum``.

    Returns:
        The result of the ``einsum`` in the original precision.

    Raises:
        RuntimeError: If the operands are not on the same device.
    """
    devices = {m.device for m in operands}
    if len(devices) > 1:
        raise RuntimeError("Operands must be on the same device.")
    dev = devices.pop()

    # Use the first tensor's data type as the result's data type.
    # The tensors may have different data types if ``autocast`` was used.
    dtype = operands[0].dtype

    # @ not supported on CPU for float16 (bfloat16 is supported)
    convert = dtype == float16 and str(dev) == "cpu"

    operands = tuple(m.to(float32) if convert else m for m in operands)
    result = einsum(equation, *operands)

    return result.to(dtype) if convert else result


def all_traces(mat: Tensor) -> Tensor:
    """Compute the traces of a matrix across all diagonals.

    A matrix of shape ``[N, M]`` has ``N + M - 1`` diagonals.

    Args:
        mat: A matrix of shape ``[N, M]``.

    Returns:
        A tensor of shape ``[N + M - 1]`` containing the traces of the matrix. Element
        ``[N - 1]`` contains the main diagonal's trace. Elements to the left contain
        the traces of the negative off-diagonals, and elements to the right contain the
        traces of the positive off-diagonals.
    """
    num_rows, num_cols = mat.shape
    num_diags = 1 + (num_rows - 1) + (num_cols - 1)

    row_idxs = arange(num_rows, device=mat.device).unsqueeze(-1).expand(-1, num_cols)
    col_idxs = arange(num_cols, device=mat.device).unsqueeze(0).expand(num_rows, -1)
    idxs = col_idxs - row_idxs
    shift = num_rows - 1  # bottom left entry of idxs
    idxs = idxs.add_(shift).flatten()

    traces = zeros(num_diags, dtype=mat.dtype, device=mat.device)
    traces.scatter_add_(0, idxs, mat.flatten())

    return traces
