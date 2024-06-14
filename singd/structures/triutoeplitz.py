"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor, arange, cat, triu_indices, zeros
from torch.linalg import vector_norm
from torch.nn.functional import conv1d, pad

from singd.structures.base import StructuredMatrix
from singd.structures.utils import all_traces, lowest_precision, toeplitz_matmul


class TriuToeplitzMatrix(StructuredMatrix):
    r"""Class for upper-triangular Toeplitz-structured matrices.

    An upper-triangular Toeplitz matrix is defined by:

    \(
    \begin{pmatrix}
        d_1 & d_2 & \cdots & d_K \\
        0 & d_1 & \ddots & \vdots \\
        \vdots & \ddots & \ddots & d_2 \\
        0 & \cdots & 0 & d_1 \\
    \end{pmatrix} \in \mathbb{R}^{K \times K}
    \quad
    \text{with}
    \quad
    \mathbf{d}
    :=
    \begin{pmatrix}
        d_1 \\
        d_2 \\
        \vdots \\
        d_K \\
    \end{pmatrix} \in \mathbb{R}^K\,.
    \)
    """

    WARN_NAIVE_EXCEPTIONS = {  # hard to leverage structure for efficient implementation
        "from_inner",
        "from_inner2",
    }

    def __init__(self, upper_diags: Tensor) -> None:
        r"""Store the upper-triangular Toeplitz matrix internally.

        Args:
            upper_diags: A vector \(\mathbf{d}\) containing the constants of all
                upper diagonals, starting with the main diagonal.
        """
        super().__init__()
        self._upper_diags: Tensor
        self.register_tensor(upper_diags, "_upper_diags")

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
        dim = self._upper_diags.shape[0]
        i, j = triu_indices(row=dim, col=dim, offset=0)
        mat = zeros(
            (dim, dim), dtype=self._upper_diags.dtype, device=self._upper_diags.device
        )
        mat[i, j] = self._upper_diags[j - i]
        return mat

    def __add__(self, other: TriuToeplitzMatrix) -> TriuToeplitzMatrix:
        """Add with another triu Toeplitz matrix.

        Args:
            other: Another triu Toeplitz matrix which will be added.

        Returns:
            A triu Toeplitz matrix resulting from the addition.
        """
        return TriuToeplitzMatrix(self._upper_diags + other._upper_diags)

    def __mul__(self, other: float) -> TriuToeplitzMatrix:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the triu Toeplitz matrix.

        Returns:
            A triu Toeplitz matrix resulting from the multiplication.
        """
        return TriuToeplitzMatrix(self._upper_diags * other)

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
        row = self._upper_diags
        dim = row.shape[0]

        if isinstance(other, Tensor):
            coeffs = cat([zeros(dim - 1, device=row.device, dtype=row.dtype), row])
            return toeplitz_matmul(coeffs, other)

        else:
            # need to create fake channel dimensions
            conv_input = pad(other._upper_diags, (dim - 1, 0)).unsqueeze(0)
            conv_weight = row.flip(0).unsqueeze(0).unsqueeze(0)
            mat_row = conv1d(conv_input, conv_weight).squeeze(0)
            return TriuToeplitzMatrix(mat_row)

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply ``mat`` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                diagonal matrix.

        Returns:
            The result of ``self.T @ mat``.
        """
        row = self._upper_diags
        dim = row.shape[0]
        coeffs = cat([row.flip(0), zeros(dim - 1, device=row.device, dtype=row.dtype)])

        out_dtype = row.dtype
        compute_dtype = lowest_precision(row.dtype, mat.dtype)
        return toeplitz_matmul(coeffs.to(compute_dtype), mat.to(compute_dtype)).to(
            out_dtype
        )

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def average_trace(self) -> Tensor:
        """Compute the average trace of the represented matrix.

        Returns:
            The average trace of the represented matrix.
        """
        return self._upper_diags[0]

    def diag_add_(self, value: float) -> TriuToeplitzMatrix:
        """In-place add a value to the diagonal of the represented matrix.

        Args:
            value: Value to add to the diagonal.

        Returns:
            A reference to the updated matrix.
        """
        self._upper_diags[0].add_(value)
        return self

    def frobenius_norm(self) -> Tensor:
        """Compute the Frobenius norm of the represented matrix.

        Returns:
            The Frobenius norm of the represented matrix.
        """
        (dim,) = self._upper_diags.shape
        multiplicity = arange(
            dim,
            0,
            step=-1,
            dtype=self._upper_diags.dtype,
            device=self._upper_diags.device,
        )
        return vector_norm(self._upper_diags * multiplicity.sqrt())

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
