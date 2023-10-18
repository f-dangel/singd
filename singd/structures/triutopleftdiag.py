"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.recursive import RecursiveTopRightMatrixTemplate


class TriuTopLeftDiagonalMatrix(RecursiveTopRightMatrixTemplate):
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

    MAX_DIMS = (float("inf"), 1)
    CLS_A = DiagonalMatrix
    CLS_C = DenseMatrix
