"""Test ``singd.structures.triutoeplitz``."""

from test.structures.test_triltoeplitz import TestTrilToeplitzMatrix
from test.structures.utils import _TestStructuredMatrix

from torch import Tensor

from singd.structures.triutoeplitz import TriuToeplitzMatrix


class TestTriuToeplitzMatrix(_TestStructuredMatrix):
    """Test suite for ``TriuToeplitzMatrix`` class."""

    STRUCTURED_MATRIX_CLS = TriuToeplitzMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a lower-triangular Toeplitz matrix.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            A lower-triangular Toeplitz matrix.
        """
        return TestTrilToeplitzMatrix().project(sym_mat).T
