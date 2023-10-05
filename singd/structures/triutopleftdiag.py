"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor, arange, zeros

from singd.structures.base import StructuredMatrix


class TriuTopLeftDiagonalMatrix(StructuredMatrix):
    """Sparse upper-triangular matrix with top left diagonal entries.

    ``
    [[D, c1],
    [[0, c2]]
    ``

    where
    - ``D`` is a diagonal matrix,
    - ``c1`` is a row vector, and
    - ``c2`` is a scalar.
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, diag: Tensor, col: Tensor) -> None:
        """Store the matrix internally.

        Args:
            diag: The diagonal elements of the matrix (``diag(D)``).
            col: The last column of the matrix (concatenation of ``c1`` and ``c2``).
        """
        assert diag.size(0) + 1 == col.size(0)

        self._mat_col = col
        self._mat_diag = diag

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor, Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tuple of tensors that need to be synchronized across devices.
        """
        return (self._mat_col, self._mat_diag)

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuTopLeftDiagonalMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TriuTopLeftDiagonalMatrix``.

        Returns:
            ``TriuTopLeftDiagonalMatrix`` approximating the passed matrix.
        """
        diag = mat.diag()
        col = mat[:, -1] + mat[-1, :]
        col[-1] = diag[-1]
        return cls(diag[:-1], col)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_col.size(0)
        mat = zeros((dim, dim), dtype=self._mat_col.dtype, device=self._mat_col.device)
        k = arange(dim - 1)
        mat[k, k] = self._mat_diag
        mat[:, -1] = self._mat_col

        return mat
