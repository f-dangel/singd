"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.recursive import RecursiveBottomLeftMatrixTemplate


class TrilTopLeftDiagonalMatrix(RecursiveBottomLeftMatrixTemplate):
    r"""Sparse lower-triangular matrix with top left diagonal entries.

    \(
    \begin{pmatrix}
    \mathbf{D} & \mathbf{0} \\
    r_1 & \mathbf{r}_2
    \end{pmatrix}
    \)

    where
    - \(\mathbf{D}\) is a diagonal matrix,
    - \(r_1\) is a scalar, and
    - \(\mathbf{r}_2\) is a row vector.
    """

    MAX_DIMS = (float("inf"), 1)
    CLS_A = DiagonalMatrix
    CLS_C = DenseMatrix
