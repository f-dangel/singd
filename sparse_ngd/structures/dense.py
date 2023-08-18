"""Dense matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from torch import Tensor

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
    def from_dense(cls, mat: Tensor) -> DenseMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense square matrix which will be represented as ``DenseMatrix``.

        Returns:
            ``DenseMatrix`` representing the passed matrix.
        """
        return cls(mat)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        return self._mat

    def transpose(self) -> DenseMatrix:
        """Create a structured matrix representing the transpose.

        Returns:
            The transpose of the represented matrix.
        """
        return DenseMatrix(self._mat.T)
