"""Test ``sparse_ngd.structures.dense``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor

from sparse_ngd.structures.dense import DenseMatrix


class TestDenseMatrix(_TestStructuredMatrix):
    """Test suite for ``DenseMatrix`` class."""

    STRUCTURED_MATRIX_CLS = DenseMatrix

    def project(self, mat: Tensor) -> Tensor:
        """Project a dense matrix onto a dense matrix.

        This is just a no-op.

        Args:
            mat: A dense matrix.

        Returns:
            The same matrix.
        """
        return mat
