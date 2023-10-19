"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.recursive import RecursiveBottomLeftMatrixTemplate


class TrilBottomRightDiagonalMatrix(RecursiveBottomLeftMatrixTemplate):
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

    MAX_DIMS = (1, float("inf"))
    CLS_A = DenseMatrix
    CLS_C = DiagonalMatrix
