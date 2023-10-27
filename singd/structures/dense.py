"""Dense matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

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
        super().__init__()
        self._mat: Tensor
        self.register_tensor(mat, "_mat")

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
