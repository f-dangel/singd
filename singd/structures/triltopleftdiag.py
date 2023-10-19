"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor, arange, zeros

from singd.structures.base import StructuredMatrix


class TrilTopLeftDiagonalMatrix(StructuredMatrix):
    """Sparse lower-triangular matrix with top left diagonal entries.

    ``
    [[D,  0],
    [[r1, r2]]
    ``
    where
    - ``D`` is a diagonal matrix,
    - ``r1`` is a scalar, and
    - ``r2`` is a row vector.
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, diag: Tensor, row: Tensor) -> None:
        """Store the matrix internally.

        Args:
            diag: The diagonal elements of the matrix (``diag(D)``).
            row: The last row of the matrix (concatenation of ``r1`` and ``r2``).
        """
        assert diag.size(0) + 1 == row.size(0)

        self._mat_row = row
        self._mat_diag = diag

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor, Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training.

        Returns:
            A tuple of tensors that need to be synchronized across devices.
        """
        return (self._mat_row, self._mat_diag)

    @classmethod
    def from_dense(cls, mat: Tensor) -> TrilTopLeftDiagonalMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TrilTopLeftDiagonalMatrix``.

        Returns:
            ``TrilTopLeftDiagonalMatrix`` approximating the passed matrix.
        """
        diag = mat.diag()
        row = mat[:, -1] + mat[-1, :]
        row[-1] = diag[-1]
        return cls(diag[:-1], row)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_row.size(0)
        mat = zeros((dim, dim), dtype=self._mat_row.dtype, device=self._mat_row.device)
        k = arange(dim - 1)
        mat[k, k] = self._mat_diag
        mat[-1, :] = self._mat_row

        return mat
