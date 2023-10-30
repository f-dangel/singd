"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.recursive import RecursiveTopRightMatrixTemplate


class TriuTopLeftDiagonalMatrix(RecursiveTopRightMatrixTemplate):
    r"""Sparse upper-triangular matrix with top left diagonal entries.

    This matrix is defined as follows:

    \(
    \begin{pmatrix}
        \mathbf{A} & \mathbf{b} \\
        \mathbf{0} & c \\
    \end{pmatrix} \in \mathbb{R}^{K \times K}
    \)

    where

    - \(\mathbf{A} \in \mathbb{R}^{(K-1)\times (K-1)}\) is a diagonal matrix represented
        as a `DiagonalMatrix`.
    - \(\mathbf{b} \in \mathbb{R}^{K-1}\) is a column vector, represented as PyTorch
        `Tensor`, and
    - \(c \in \mathbb{R}\) is a scalar, represented by a `DenseMatrix`.
    """

    MAX_DIMS = (float("inf"), 1)
    CLS_A = DiagonalMatrix
    CLS_C = DenseMatrix
