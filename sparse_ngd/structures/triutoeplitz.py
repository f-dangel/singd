"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor, arange, cat, triu_indices, zeros
from torch.nn.functional import pad

from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.utils import all_traces, supported_conv1d, toeplitz_matmul


class TriuToeplitzMatrix(StructuredMatrix):
    """Upper-triangular Toeplitz-structured matrix.

    We follow the representation of such matrices using the SciPy terminology, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """

    WARN_NAIVE_EXCEPTIONS = {  # hard to leverage structure for efficient implementation
        "from_inner",
        "from_inner2",
    }

    def __init__(self, diag_consts: Tensor) -> None:
        """Store the upper-triangular Toeplitz matrix internally.

        Args:
            diag_consts: A vector containing the constants of all diagonals, i.e.
                the first entry corresponds to the constant on the diagonal, the
                second entry to the constant on the upper first off-diagonal, etc.
        """
        self._mat_row = diag_consts

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tensor that need to be synchronized across devices.
        """
        return (self._mat_row,)

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuToeplitzMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TriuToeplitzMatrix``.

        Returns:
            ``TriuToeplitzMatrix`` approximating the passed matrix.
        """
        assert mat.shape[0] == mat.shape[1]
        traces = all_traces(mat)

        # sum the lower- and upper-diagonal traces
        dim = mat.shape[0]
        row = zeros(dim, dtype=mat.dtype, device=mat.device)
        idx_main = dim - 1
        row[0] += traces[idx_main]
        row[1:] += traces[idx_main + 1 :]
        row[1:] += traces[:idx_main].flip(0)

        normalization = arange(dim, 0, step=-1, dtype=mat.dtype, device=mat.device)
        row.div_(normalization)

        return cls(row)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_row.shape[0]
        i, j = triu_indices(row=dim, col=dim, offset=0)
        mat = zeros((dim, dim), dtype=self._mat_row.dtype, device=self._mat_row.device)
        mat[i, j] = self._mat_row[j - i]
        return mat

    def __add__(self, other: TriuToeplitzMatrix) -> TriuToeplitzMatrix:
        """Add with another triu Toeplitz matrix.

        Args:
            other: Another triu Toeplitz matrix which will be added.

        Returns:
            A triu Toeplitz matrix resulting from the addition.
        """
        return TriuToeplitzMatrix(self._mat_row + other._mat_row)

    def __mul__(self, other: float) -> TriuToeplitzMatrix:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the triu Toeplitz matrix.

        Returns:
            A triu Toeplitz matrix resulting from the multiplication.
        """
        return TriuToeplitzMatrix(self._mat_row * other)

    def __matmul__(
        self, other: Union[TriuToeplitzMatrix, Tensor]
    ) -> Union[TriuToeplitzMatrix, Tensor]:
        """Multiply the triu Toeplitz matrix onto another or a Tensor (@ operator).

        Args:
            other: A matrix which will be multiplied onto. Can be represented by a
                PyTorch tensor or another ``TriuToeplitzMatrix``.

        Returns:
            Result of the multiplication. If a PyTorch tensor was passed as argument,
            the result will be a PyTorch tensor. If a triu Toeplitz matrix was passed,
            the result will be returned as a ``TriuToeplitzMatrix``.
        """
        row = self._mat_row
        dim = row.shape[0]

        if isinstance(other, Tensor):
            coeffs = cat([zeros(dim - 1, device=row.device, dtype=row.dtype), row])
            return toeplitz_matmul(coeffs, other)

        else:
            # need to create fake channel dimensions
            conv_input = pad(other._mat_row, (dim - 1, 0)).unsqueeze(0)
            conv_weight = row.flip(0).unsqueeze(0).unsqueeze(0)
            mat_row = supported_conv1d(conv_input, conv_weight).squeeze(0)
            return TriuToeplitzMatrix(mat_row)

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply ``mat`` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                diagonal matrix.

        Returns:
            The result of ``self.T @ mat``.
        """
        row = self._mat_row
        dim = row.shape[0]
        coeffs = cat([row.flip(0), zeros(dim - 1, device=row.device, dtype=row.dtype)])
        return toeplitz_matmul(coeffs, mat)

    def trace(self) -> Tensor:
        """Compute the trace of the represented matrix.

        Returns:
            The trace of the represented matrix.
        """
        dim = self._mat_row.shape[0]
        return self._mat_row[0] * dim

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################
    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> TriuToeplitzMatrix:
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
        return TriuToeplitzMatrix(zeros(dim, dtype=dtype, device=device))

    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> TriuToeplitzMatrix:
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
        return TriuToeplitzMatrix(coeffs)
