"""Dense matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor

from singd.structures.base import StructuredMatrix


class DenseMatrix(StructuredMatrix):
    """Unstructured dense matrix implemented in the ``StructuredMatrix`` interface."""

    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, mat: Tensor) -> None:
        """Store the dense matrix internally.

        Args:
            mat: A dense square matrix.
        """
        self._mat = mat

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training.

        Returns:
            A tensor that needs to be synchronized across devices.
        """
        return (self._mat,)

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
