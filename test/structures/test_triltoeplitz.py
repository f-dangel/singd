"""Test ``singd.structures.triltoeplitz``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor, ones_like, zeros_like

from singd.structures.triltoeplitz import TrilToeplitzMatrix


class TestTrilToeplitzMatrix(_TestStructuredMatrix):
    """Test suite for ``TrilToeplitzMatrix`` class."""

    STRUCTURED_MATRIX_CLS = TrilToeplitzMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a lower-triangular Toeplitz matrix.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            A lower-triangular Toeplitz matrix.
        """
        tril_toeplitz = zeros_like(sym_mat)

        # average diagonal and upper off-diagonals and fill into a matrix
        for d in range(sym_mat.shape[0]):
            if d == 0:
                d_diag = sym_mat.diag(diagonal=d)
            else:
                d_diag = sym_mat.diag(diagonal=-d) + sym_mat.diag(diagonal=d)
            d_const = d_diag.mean()
            tril_toeplitz += (d_const * ones_like(d_diag)).diag(diagonal=-d)

        return tril_toeplitz
