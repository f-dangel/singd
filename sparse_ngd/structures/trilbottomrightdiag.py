"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

import torch
from torch import Tensor

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

        self._mat_column = col
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
        dim = self._mat_column.size(0)
        mat = torch.zeros(
            (dim, dim), dtype=self._mat_column.dtype, device=self._mat_column.device
        )

        k = torch.tensor(range(dim - 1)) + 1
        mat[k, k] = self._mat_diag
        mat[:, 0] = self._mat_column

        return mat
