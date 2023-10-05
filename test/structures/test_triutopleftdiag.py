"""Test ``singd.structures.triutopleftdiag``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor, zeros_like

from singd.structures.triutopleftdiag import TriuTopLeftDiagonalMatrix


class TestTriuTopLeftDiagonalMatrix(_TestStructuredMatrix):
    """Test suite for ``TriuTopLeftDiagonalMatrix`` class."""

    STRUCTURED_MATRIX_CLS = TriuTopLeftDiagonalMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a triu matrix w/ top left diagonal.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            An upper-triangular matrix with top left diagonal.
        """
        tril_top_left_diag = zeros_like(sym_mat)

        # last column (except for last entry)
        tril_top_left_diag[:-1, -1] = sym_mat[:-1, -1] + sym_mat[-1, :-1]
        # diagonal
        for i in range(sym_mat.shape[0]):
            tril_top_left_diag[i, i] = sym_mat[i, i]

        return tril_top_left_diag
