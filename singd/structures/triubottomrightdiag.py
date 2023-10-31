"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.recursive import RecursiveTopRightMatrixTemplate


class TriuBottomRightDiagonalMatrix(RecursiveTopRightMatrixTemplate):
    r"""Sparse upper-triangular matrix with bottom right diagonal entries.

    This matrix is defined as follows:

    \(
    \begin{pmatrix}
        a & \mathbf{b} \\
        \mathbf{0} & \mathbf{C} \\
    \end{pmatrix} \in \mathbb{R}^{K \times K}
    \)

    where

    - \(a \in \mathbb{R}\) is a scalar, represented by a `DenseMatrix`
    - \(\mathbf{b} \in \mathbb{R}^{K-1}\) is a column vector, represented as PyTorch
        `Tensor`, and
    - \(\mathbf{C} \in \mathbb{R}^{(K-1)\times (K-1)}\) is a diagonal matrix represented
        as a `DiagonalMatrix`.
    """

    MAX_DIMS = (1, float("inf"))
    CLS_A = DenseMatrix
    CLS_C = DiagonalMatrix
