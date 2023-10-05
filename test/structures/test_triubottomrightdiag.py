"""Test ``singd.structures.triubottomrightdiag``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor, zeros_like

from singd.structures.triubottomrightdiag import TriuBottomRightDiagonalMatrix


class TestTriuBottomRightDiagonalMatrix(_TestStructuredMatrix):
    """Test suite for ``TriuBottomRightDiagonalMatrix`` class."""

    STRUCTURED_MATRIX_CLS = TriuBottomRightDiagonalMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a triu matrix w/ bottom right diagonal.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            An upper-triangular matrix with bottom right diagonal.
        """
        tril_bottom_right_diag = zeros_like(sym_mat)

        # first row (except for first entry)
        tril_bottom_right_diag[0, 1:] = sym_mat[0, 1:] + sym_mat[1:, 0]
        # diagonal
        for i in range(sym_mat.shape[0]):
            tril_bottom_right_diag[i, i] = sym_mat[i, i]

        return tril_bottom_right_diag
