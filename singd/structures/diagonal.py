"""Diagonal matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor, einsum, ones, zeros
from torch.linalg import vector_norm

from singd.structures.base import StructuredMatrix


class DiagonalMatrix(StructuredMatrix):
    r"""Diagonal matrix implemented in the ``StructuredMatrix`` interface.

    A diagonal matrix is defined as

    \(
    \begin{pmatrix}
        d_1 & 0 & \cdots & 0 \\
        0 & d_2 & \ddots & \vdots \\
        \vdots & \ddots & \ddots & 0 \\
        0 & \cdots & \ddots & d_K \\
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
    \end{pmatrix} \in \mathbb{R}^K
    \)
    """

    def __init__(self, mat_diag: Tensor) -> None:
        r"""Store the dense matrix internally.

        Args:
            mat_diag: A 1d tensor representing the matrix diagonal \(\mathbf{d}\).
        """
        super().__init__()
        self._mat_diag: Tensor
        self.register_tensor(mat_diag, "_mat_diag")

    def __matmul__(
        self, other: Union[DiagonalMatrix, Tensor]
    ) -> Union[DiagonalMatrix, Tensor]:
        """Multiply the diagonal matrix onto a (@ operator).

        Args:
            other: A matrix which will be multiplied onto. Can be represented by a
                PyTorch tensor or a structured matrix.

        Returns:
            Result of the multiplication. If a PyTorch tensor was passed as argument,
            the result will be a PyTorch tensor. If a diagonal matrix was passed, the
            result will be returned as a ``DiagonalMatrix``.
        """
        if isinstance(other, Tensor):
            return einsum("i,i...->i...", self._mat_diag, other)
        else:
            return DiagonalMatrix(self._mat_diag * other._mat_diag)

    def __add__(self, other: DiagonalMatrix) -> DiagonalMatrix:
        """Add with another diagonal matrix.

        Args:
            other: Another diagonal matrix which will be added.

        Returns:
            A diagonal matrix resulting from the addition.
        """
        return DiagonalMatrix(self._mat_diag + other._mat_diag)

    def __mul__(self, other: float) -> DiagonalMatrix:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the diagonal matrix.

        Returns:
            A diagonal matrix resulting from the multiplication.
        """
        return DiagonalMatrix(self._mat_diag * other)

    @classmethod
    def from_dense(cls, sym_mat: Tensor) -> DiagonalMatrix:
        """Construct a diagonal matrix from a PyTorch tensor.

        This will discard elements that are not part of the diagonal, even if they
        are non-zero.

        Args:
            sym_mat: A symmetric dense matrix which will be represented as
                ``DiagonalMatrix``.

        Returns:
            ``DiagonalMatrix`` representing the passed matrix's diagonal.
        """
        return cls(sym_mat.diag())

    def to_dense(self) -> Tensor:
        """Convert diagonal matrix into a dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        return self._mat_diag.diag()

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply ``mat`` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                diagonal matrix.

        Returns:
            The result of ``self.T @ mat``.
        """
        return self @ mat

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def from_inner(self, X: Union[Tensor, None] = None) -> DiagonalMatrix:
        """Represent the matrix diagonal of ``self.T @ X @ X^T @ self``.

        Let ``K := self``. Then we have to compute ``diag(K^T @ X @ X^T @ K)``.
        Since ``K`` is diagonal, this is ``diag(K^T) * diag(X @ X^T) * diag(K)``,
        which is ``diag(K^T) ** 2 * diag(X @ X^T)``. Further, ``diag(X @ X^T)``
        is simply the element-wise square ``X ** 2``, summed over columns.

        Args:
            X: Optional arbitrary 2d tensor. If ``None``, ``X = I`` will be used.

        Returns:
            A ``DiagonalMatrix`` representing matrix diagonal of
            ``self.T @ X @ X^T @ self``.
        """
        mat_diag = self._mat_diag**2
        if X is not None:
            mat_diag *= (X**2).sum(1)
        return DiagonalMatrix(mat_diag)

    @classmethod
    def from_mat_inner(cls, X: Tensor) -> DiagonalMatrix:
        """Extract a structured matrix from `X @ X.T`.

        Args:
            X: Arbitrary 2d tensor.

        Returns:
            The structured matrix extracted from `X @ X^T`.
        """
        return DiagonalMatrix((X**2).sum(1))

    def from_inner2(self, XXT: Tensor) -> StructuredMatrix:
        """Represent the matrix diagonal of ``self.T @ XXT @ self``.

        Args:
            XXT: 2d square symmetric matrix.

        Returns:
            A ``DiagonalMatrix`` representing matrix diagonal of
            ``self.T @ XXT @ self``.
        """
        return DiagonalMatrix(self._mat_diag**2 * XXT.diag())

    def average_trace(self) -> Tensor:
        """Compute the average trace of the represented matrix.

        Returns:
            The average trace of the represented matrix.
        """
        return self._mat_diag.mean()

    def diag_add_(self, value: float) -> DiagonalMatrix:
        """In-place add a value to the diagonal of the represented matrix.

        Args:
            value: Value to add to the diagonal.

        Returns:
            A reference to the updated matrix.
        """
        self._mat_diag.add_(value)
        return self

    def frobenius_norm(self) -> Tensor:
        """Compute the Frobenius norm of the represented matrix.

        Returns:
            The Frobenius norm of the represented matrix.
        """
        return vector_norm(self._mat_diag)

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################
    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> DiagonalMatrix:
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
        return DiagonalMatrix(zeros(dim, dtype=dtype, device=device))

    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> DiagonalMatrix:
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
        return DiagonalMatrix(ones(dim, dtype=dtype, device=device))
