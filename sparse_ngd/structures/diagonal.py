"""Diagonal matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Union

from torch import Tensor

from sparse_ngd.structures.base import StructuredMatrix


class DiagonalMatrix(StructuredMatrix):
    """Diagonal matrix implemented in the ``StructuredMatrix`` interface."""

    def __init__(self, mat_diag: Tensor) -> None:
        """Store the dense matrix internally.

        Args:
            mat_diag: A 1d tensor representing the matrix diagonal.
        """
        self._mat_diag = mat_diag

    def __matmul__(self, other: DiagonalMatrix) -> DiagonalMatrix:
        """Multiply with another diagonal matrix (@ operator).

        Args:
            other: Another diagonal matrix which will be multiplied onto.

        Returns:
            A diagonal matrix resulting from the multiplication.
        """
        return DiagonalMatrix(self._mat_diag * other._mat_diag)

    @classmethod
    def from_dense(cls, mat: Tensor) -> DiagonalMatrix:
        """Construct a diagonal matrix from a PyTorch tensor.

        This will discard elements that are not part of the diagonal, even if they
        are non-zero.

        Args:
            mat: A dense square matrix which will be represented as ``DiagonalMatrix``.

        Returns:
            ``DiagonalMatrix`` representing the passed matrix's diagonal.
        """
        return cls(mat.diag())

    def to_dense(self) -> Tensor:
        """Convert diagonal matrix into a dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        return self._mat_diag.diag()

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
            The matrix diagonal of ``self.T @ X @ X^T @ self``.
        """
        mat_diag = self._mat_diag**2
        if X is not None:
            mat_diag *= (X**2).sum(1)
        return DiagonalMatrix(mat_diag)
