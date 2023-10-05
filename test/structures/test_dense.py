"""Test ``singd.structures.dense``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor

from singd.structures.dense import DenseMatrix


class TestDenseMatrix(_TestStructuredMatrix):
    """Test suite for ``DenseMatrix`` class."""

    STRUCTURED_MATRIX_CLS = DenseMatrix

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a dense symmetric matrix onto a dense symmetric matrix.

        This is just a no-op.

        Args:
            sym_mat: A dense symmetric matrix.

        Returns:
            The same matrix.
        """
        return sym_mat
