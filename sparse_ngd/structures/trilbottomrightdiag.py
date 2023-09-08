"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple, Union

import torch.distributed as dist
from torch import Tensor, arange, zeros
from torch._C import Future

from sparse_ngd.structures.base import StructuredMatrix


class TrilBottomRightDiagonalMatrix(StructuredMatrix):
    """Sparse lower-triangular matrix with bottom right diagonal.

    ``
    [[c1, 0],
    [[c2, D]]
    ``

    where
    - ``c1`` is a scalar,
    - ``c2`` is a row vector, and
    - ``D`` is a diagonal matrix.
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, diag: Tensor, col: Tensor) -> None:
        """Store the matrix internally.

        Args:
            diag: The diagonal elements of the matrix (``diag(D)``).
            col: The first column of the matrix (concatenation of ``c1`` and ``c2``).
        """
        assert diag.size(0) + 1 == col.size(0)

        self._mat_col = col
        self._mat_diag = diag

    @classmethod
    def from_dense(cls, mat: Tensor) -> TrilBottomRightDiagonalMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TrilBottomRightDiagonalMatrix``.

        Returns:
            ``TrilBottomRightDiagonalMatrix`` approximating the passed matrix.
        """
        diag = mat.diag()
        col = mat[:, 0] + mat[0, :]
        col[0] = diag[0]
        return cls(diag[1:], col)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_col.size(0)
        mat = zeros((dim, dim), dtype=self._mat_col.dtype, device=self._mat_col.device)
        k = arange(1, dim)
        mat[k, k] = self._mat_diag
        mat[:, 0] = self._mat_col

        return mat

    def all_reduce(
        self,
        op: dist.ReduceOp = dist.ReduceOp.AVG,
        group: Union[dist.ProcessGroup, None] = None,
        async_op: bool = False,
    ) -> Union[None, Tuple[Future, Future]]:
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
            If ``async_op`` is ``True``, a tuple of ``torch.distributed.Future``
            objects, else ``None``.
        """
        if async_op:
            handle_col = dist.all_reduce(
                self._mat_col, op=op, group=group, async_op=async_op
            )
            handle_diag = dist.all_reduce(
                self._mat_diag, op=op, group=group, async_op=async_op
            )
            return handle_col, handle_diag
        dist.all_reduce(self._mat_col, op=op, group=group, async_op=async_op)
        dist.all_reduce(self._mat_diag, op=op, group=group, async_op=async_op)
