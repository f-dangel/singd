"""Diagonal matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor, einsum, ones, zeros

from sparse_ngd.structures.base import StructuredMatrix


class DiagonalMatrix(StructuredMatrix):
    """Diagonal matrix implemented in the ``StructuredMatrix`` interface."""

    def __init__(self, mat_diag: Tensor) -> None:
        """Store the dense matrix internally.

        Args:
            mat_diag: A 1d tensor representing the matrix diagonal.
        """
        self._mat_diag = mat_diag

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
            The transpose of the represented matrix.
        """
        return self @ mat

    def all_reduce(
        self,
        op: dist.ReduceOp = dist.ReduceOp.AVG,
        group: Union[dist.ProcessGroup, None] = None,
        async_op: bool = False,
    ) -> Union[None, torch._C.Future]:
        """Reduce the structured matrix across all workers.

        Args:
            op: The reduction operation to perform (default: ``dist.ReduceOp.AVG``).
            group: The process group to work on. If ``None``, the default process group
                will be used.
            async_op: If ``True``, this function will return a
                ``torch.distributed.Future`` object.
                Otherwise, it will block until the reduction completes
                (default: ``False``).

        Returns:
            If ``async_op`` is ``True``, a ``torch.distributed.Future``
            object, else ``None``.
        """
        if async_op:
            return dist.all_reduce(
                self._mat_diag, op=op, group=group, async_op=async_op
            )
        dist.all_reduce(self._mat_diag, op=op, group=group, async_op=async_op)

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

    def from_inner2(self, XXT: Tensor) -> StructuredMatrix:
        """Represent the matrix diagonal of ``self.T @ XXT @ self``.

        Args:
            XXT: 2d square symmetric matrix.

        Returns:
            A ``DiagonalMatrix`` representing matrix diagonal of
            ``self.T @ XXT @ self``.
        """
        return DiagonalMatrix(self._mat_diag**2 * XXT.diag())

    def trace(self) -> Tensor:
        """Compute the trace of the represented matrix.

        Returns:
            The trace of the represented matrix.
        """
        return self._mat_diag.sum()

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################
    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> StructuredMatrix:
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
