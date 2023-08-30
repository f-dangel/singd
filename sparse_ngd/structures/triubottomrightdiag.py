"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from torch import Tensor, arange, zeros

from sparse_ngd.structures.base import StructuredMatrix


class TriuBottomRightDiagonalMatrix(StructuredMatrix):
    """Sparse upper-triangular matrix with bottom right diagonal entries.

    ``
    [[r1, r2],
    [[0,  D]]
    ``

    where
    - ``r1`` is a scalar,
    - ``r2`` is a row vector, and
    - ``D`` is a diagonal matrix,
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, diag: Tensor, row: Tensor) -> None:
        """Store the matrix internally.

        Args:
            diag: The diagonal elements of the matrix (``diag(D)``).
            row: the first row of the matrix (concatenation of ``r1`` and ``r2``).
        """
        assert diag.size(0) + 1 == row.size(0)

        self._mat_row = row
        self._mat_diag = diag

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuBottomRightDiagonalMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TriuBottomRightDiagonalMatrix``.

        Returns:
            ``TriuBottomRightDiagonalMatrix`` approximating the passed matrix.
        """
        diag = mat.diag()
        row = mat[:, 0] + mat[0, :]
        row[0] = diag[0]
        return cls(diag[1:], row)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_row.size(0)
        mat = zeros((dim, dim), dtype=self._mat_row.dtype, device=self._mat_row.device)

        k = arange(1, dim)
        mat[k, k] = self._mat_diag
        mat[0, :] = self._mat_row

        return mat
