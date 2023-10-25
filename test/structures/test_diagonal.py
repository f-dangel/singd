"""Test `singd.structures.diagonal`."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor

from singd.structures.diagonal import DiagonalMatrix


class TestDiagonalMatrix(_TestStructuredMatrix):
    """Test suite for `DiagonalMatrix` class."""

    STRUCTURED_MATRIX_CLS = DiagonalMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a diagonal matrix.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            A matrix containing the diagonal of `mat` on its diagonal.
        """
        return sym_mat.diag().diag()
