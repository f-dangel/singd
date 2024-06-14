"""Utility functions for the structured matrices."""

from typing import Any

import torch
from torch import (
    Tensor,
    arange,
    bfloat16,
    device,
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
    conv_result = conv1d(conv_input, conv_weight, padding=padding, groups=num_cols)

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
