"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor, arange, cat, zeros
from torch.nn.functional import pad

from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.utils import (
    all_traces,
    lowest_precision,
    supported_conv1d,
    toeplitz_matmul,
)


class TrilToeplitzMatrix(StructuredMatrix):
    """Lower-triangular Toeplitz-structured matrix in ``StructuredMatrix`` interface.

    We follow the representation of such matrices using the SciPy terminology, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """

    WARN_NAIVE_EXCEPTIONS = {  # hard to leverage structure for efficient implementation
        "from_inner",
        "from_inner2",
    }

    def __init__(self, diag_consts: Tensor) -> None:
        """Store the lower-triangular Toeplitz matrix internally.

        Args:
            diag_consts: A vector containing the constants of all diagonals, i.e.
                the first entry corresponds to the constant on the diagonal, the
                second entry to the constant on the lower first off-diagonal, etc.
        """
        self._mat_column = diag_consts

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tensor that need to be synchronized across devices.
        """
        return (self._mat_column,)

    @classmethod
    def from_dense(cls, mat: Tensor) -> TrilToeplitzMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TrilToeplitzMatrix``.

        Returns:
            ``TrilToeplitzMatrix`` approximating the passed matrix.
        """
        assert mat.shape[0] == mat.shape[1]
        traces = all_traces(mat)

        # sum the lower- and upper-diagonal traces
        dim = mat.shape[0]
        col = zeros(dim, dtype=mat.dtype, device=mat.device)
        idx_main = dim - 1
        col[0] += traces[idx_main]
        col[1:] += traces[idx_main + 1 :]
        col[1:] += traces[:idx_main].flip(0)

        normalization = arange(dim, 0, step=-1, dtype=mat.dtype, device=mat.device)
        col.div_(normalization)

        return cls(col)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_column.size(0)
        i, j = torch.tril_indices(row=dim, col=dim, offset=0)
        mat = torch.zeros(
            (dim, dim), device=self._mat_column.device, dtype=self._mat_column.dtype
        )
        mat[i, j] = self._mat_column[i - j]
        return mat

    def __add__(self, other: TrilToeplitzMatrix) -> TrilToeplitzMatrix:
        """Add with another tril Toeplitz matrix.

        Args:
            other: Another tril Toeplitz matrix which will be added.

        Returns:
            A tril Toeplitz matrix resulting from the addition.
        """
        return TrilToeplitzMatrix(self._mat_column + other._mat_column)

    def __mul__(self, other: float) -> TrilToeplitzMatrix:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the triu Toeplitz matrix.

        Returns:
            A triu Toeplitz matrix resulting from the multiplication.
        """
        return TrilToeplitzMatrix(self._mat_column * other)

    def __matmul__(
        self, other: Union[TrilToeplitzMatrix, Tensor]
    ) -> Union[TrilToeplitzMatrix, Tensor]:
        """Multiply the triu Toeplitz matrix onto another or a Tensor (@ operator).

        Args:
            other: A matrix which will be multiplied onto. Can be represented by a
                PyTorch tensor or another ``TrilToeplitzMatrix``.

        Returns:
            Result of the multiplication. If a PyTorch tensor was passed as argument,
            the result will be a PyTorch tensor. If a triu Toeplitz matrix was passed,
            the result will be returned as a ``TrilToeplitzMatrix``.
        """
        col = self._mat_column
        dim = col.shape[0]

        if isinstance(other, Tensor):
            coeffs = cat(
                [col.flip(0), zeros(dim - 1, device=col.device, dtype=col.dtype)]
            )
            return toeplitz_matmul(coeffs, other)

        else:
            # need to create fake channel dimensions
            conv_input = pad(other._mat_column, (dim - 1, 0)).unsqueeze(0)
            conv_weight = col.flip(0).unsqueeze(0).unsqueeze(0)
            mat_column = supported_conv1d(conv_input, conv_weight).squeeze(0)
            return TrilToeplitzMatrix(mat_column)

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply ``mat`` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                diagonal matrix.

        Returns:
            The result of ``self.T @ mat``.
        """
        col = self._mat_column
        dim = col.shape[0]
        coeffs = cat([zeros(dim - 1, device=col.device, dtype=col.dtype), col])

        out_dtype = col.dtype
        compute_dtype = lowest_precision(col.dtype, mat.dtype)
        return toeplitz_matmul(coeffs.to(compute_dtype), mat.to(compute_dtype)).to(
            out_dtype
        )

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def trace(self) -> Tensor:
        """Compute the trace of the represented matrix.

        Returns:
            The trace of the represented matrix.
        """
        dim = self._mat_column.shape[0]
        return self._mat_column[0] * dim

    def diag_add_(self, value: float) -> TrilToeplitzMatrix:
        """In-place add a value to the diagonal of the represented matrix.

        Args:
            value: Value to add to the diagonal.

        Returns:
            A reference to the updated matrix.
        """
        self._mat_column[0].add_(value)
        return self

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################
    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> TrilToeplitzMatrix:
        """Create a structured matrix representing the zero matrix.

        Args:
            dim: Dimension of the (square) matrix.
            dtype: Optional data type of the matrix. If not specified, uses the default
                tensor type.
            device: Optional device of the matrix. If not specified, uses the default
                tensor type.

        Returns:
            A structured matrix representing the zero matrix.
        """
        return TrilToeplitzMatrix(zeros(dim, dtype=dtype, device=device))

    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> TrilToeplitzMatrix:
        """Create a diagonal matrix representing the identity matrix.

        Args:
            dim: Dimension of the (square) matrix.
            dtype: Optional data type of the matrix. If not specified, uses the default
                tensor type.
            device: Optional device of the matrix. If not specified, uses the default
                tensor type.

        Returns:
            A diagonal matrix representing the identity matrix.
        """
        coeffs = zeros(dim, dtype=dtype, device=device)
        coeffs[0] = 1.0
        return TrilToeplitzMatrix(coeffs)
