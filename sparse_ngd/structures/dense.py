"""Dense matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Union

import torch.distributed as dist
from torch import Tensor
from torch._C import Future

from sparse_ngd.structures.base import StructuredMatrix


class DenseMatrix(StructuredMatrix):
    """Unstructured dense matrix implemented in the ``StructuredMatrix`` interface."""

    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, mat: Tensor) -> None:
        """Store the dense matrix internally.

        Args:
            mat: A dense square matrix.
        """
        self._mat = mat

    @classmethod
    def from_dense(cls, sym_mat: Tensor) -> DenseMatrix:
        """Construct from a PyTorch tensor.

        Args:
            sym_mat: A dense symmetric matrix which will be represented as
                ``DenseMatrix``.

        Returns:
            ``DenseMatrix`` representing the passed matrix.
        """
        return cls(sym_mat)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        return self._mat

    def all_reduce(
        self,
        op: dist.ReduceOp = dist.ReduceOp.AVG,
        group: Union[dist.ProcessGroup, None] = None,
        async_op: bool = False,
    ) -> Union[None, Future]:
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
            return dist.all_reduce(self._mat, op=op, group=group, async_op=async_op)
        dist.all_reduce(self._mat, op=op, group=group, async_op=async_op)
