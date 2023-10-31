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
from torch.nn.functional import conv1d


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
    # The matrices may have different data types if `autocast` was used.
    dtype = matrices[0].dtype

    # @ not supported on CPU for float16 (bfloat16 is supported)
    convert = dtype == float16 and str(dev) == "cpu"

    result = matrices[0].to(float32) if convert else matrices[0]
    for mat in matrices[1:]:
        result = result @ mat.to(result.dtype)

    return result.to(dtype) if convert else result


def supported_eye(n: int, **kwargs: Any) -> Tensor:
    """Same as PyTorch's `eye`, but uses higher precision if unsupported.

    Args:
        n: The number of rows.
        kwargs: Keyword arguments to `torch.eye`.

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


def supported_einsum(equation, *operands: Tensor) -> Tensor:
    """Compute an `einsum` with the same or higher numerical precision.

    If the `einsum` is not supported on the hardware,
    carry out the multiplication in single precision.

    Args:
        equation: The `einsum` equation.
        operands: The operands to the `einsum`.

    Returns:
        The result of the `einsum` in the original precision.

    Raises:
        RuntimeError: If the operands are not on the same device.
    """
    devices = {m.device for m in operands}
    if len(devices) > 1:
        raise RuntimeError("Operands must be on the same device.")
    dev = devices.pop()

    # Use the first tensor's data type as the result's data type.
    # The tensors may have different data types if `autocast` was used.
    dtype = operands[0].dtype

    # @ not supported on CPU for float16 (bfloat16 is supported)
    convert = dtype == float16 and str(dev) == "cpu"

    operands = tuple(m.to(float32) if convert else m for m in operands)
    result = einsum(equation, *operands)

    return result.to(dtype) if convert else result


def all_traces(mat: Tensor) -> Tensor:
    """Compute the traces of a matrix across all diagonals.

    A matrix of shape `[N, M]` has `N + M - 1` diagonals.

    Args:
        mat: A matrix of shape `[N, M]`.

    Returns:
        A tensor of shape `[N + M - 1]` containing the traces of the matrix. Element
        `[N - 1]` contains the main diagonal's trace. Elements to the left contain
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


def supported_conv1d(
    input: Tensor, weight: Tensor, padding: int = 0, groups: int = 1
) -> Tensor:
    """Same as PyTorch's `conv1d`, but uses higher precision if unsupported.

    For now, we don't support bias and non-default hyper-parameters.

    Args:
        input: The input of the convolution. Has shape `[N, C_in, I_1]`.
        weight: The kernel of the convolution. Has shape `[C_out, C_in // G, K_1]`.
        padding: The amount of padding on both sides of the input. Default: `0`.
        groups: The number of groups `G`. Default: `1`.

    Returns:
        The output of the convolution in the same precision as `input`.
        Has shape `[N, C_out, O_1]`, where `O_1 = I_1 - K_1 + 1`.

    Raises:
        RuntimeError: If input and kernel are not on the same device.
    """
    devices = {input.device, weight.device}
    if len(devices) > 1:
        raise RuntimeError("Input and kernel must be on the same device.")
    dev = devices.pop()

    # Use the input's data type as the result's data type.
    # Input and kernel may have different data types if `autocast` was used.
    dtype = input.dtype

    # 'slow_conv2d_cpu' not implemented for 'Half' (bfloat16 is supported)
    if dtype == float16 and str(dev) == "cpu":
        return conv1d(
            input.to(float32), weight.to(float32), padding=padding, groups=groups
        ).to(dtype)
    else:
        return conv1d(input, weight, padding=padding, groups=groups)


def toeplitz_matmul(coeffs: Tensor, mat: Tensor) -> Tensor:
    """Compute the product of a Toeplitz matrix and a matrix.

    Let `N` denote the expanded Toeplitz matrix dimension.

    Args:
        coeffs: A tensor of shape `[2 * N - 1]` containing the elements on the
            Toeplitz matrix's diagonals, starting from most negative (bottom left)
            to top right, that is the central element contains the value of the
            Toeplitz matrix's main diagonal.
        mat: A matrix of shape `[N, M]`.

    Returns:
        The product of the Toeplitz matrix and the matrix. Has shape `[N, M]`.

    Raises:
        RuntimeError: If the specified tensors have incorrect shape.
    """
    toeplitz_dim = (coeffs.shape[0] + 1) // 2
    if toeplitz_dim != mat.shape[0]:
        raise RuntimeError(f"Toeplitz dim={toeplitz_dim}, but `mat` is {mat.shape}.")

    num_rows = mat.shape[0]
    num_cols = mat.shape[1]
    padding = num_rows - 1

    # columns act as channels
    conv_input = mat.T
    conv_weight = coeffs.unsqueeze(0).unsqueeze(0).expand(num_cols, -1, -1)
    conv_result = supported_conv1d(
        conv_input, conv_weight, padding=padding, groups=num_cols
    )

    return conv_result.T


def diag_add_(mat: Tensor, value: float) -> Tensor:
    """In-place add a value to the main diagonal of a matrix.

    Args:
        mat: A square matrix of shape `[N, N]`.
        value: The value to add to the main diagonal.

    Raises:
        ValueError: If the specified tensor is not square.

    Returns:
        The input matrix with the value added to the main diagonal.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square matrix, but got {mat.shape}.")

    dim = mat.shape[0]
    idxs = arange(dim, device=mat.device)
    mat[idxs, idxs] += value

    return mat


def lowest_precision(*dtypes: torch.dtype) -> torch.dtype:
    """Return the data type of lowest precision.

    Args:
        *dtypes: The data types to compare.

    Returns:
        The data type of lowest precision (`float16 < bfloat16 < float32`).

    Raises:
        NotImplementedError: If any of the specified data types is not supported.
    """
    supported = [float16, bfloat16, float32]
    if any(dtype not in supported for dtype in dtypes):
        unsupported = [dtype for dtype in dtypes if dtype not in supported]
        raise NotImplementedError(f"Unsupported data type(s): {unsupported}.")
    min_score = min(supported.index(dtype) for dtype in dtypes)
    return supported[min_score]
