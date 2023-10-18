"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.recursive import RecursiveTopRightMatrixTemplate


class TriuBottomRightDiagonalMatrix(RecursiveTopRightMatrixTemplate):
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

    MAX_DIMS = (1, float("inf"))
    CLS_A = DenseMatrix
    CLS_C = DiagonalMatrix
