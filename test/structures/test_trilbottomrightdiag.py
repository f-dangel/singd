"""Test ``singd.structures.trilbottomrightdiag``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor, zeros_like

from singd.structures.trilbottomrightdiag import TrilBottomRightDiagonalMatrix


class TestTrilBottomRightDiagonalMatrix(_TestStructuredMatrix):
    """Test suite for ``TrilBottomRightDiagonalMatrix`` class."""

    STRUCTURED_MATRIX_CLS = TrilBottomRightDiagonalMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a tril matrix w/ bottom right diagonal.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            A lower-triangular matrix with bottom right diagonal.
        """
        tril_bottom_right_diag = zeros_like(sym_mat)

        # first column (except for first entry)
        tril_bottom_right_diag[1:, 0] = sym_mat[1:, 0] + sym_mat[0, 1:]
        # diagonal
        for i in range(sym_mat.shape[0]):
            tril_bottom_right_diag[i, i] = sym_mat[i, i]

        return tril_bottom_right_diag
