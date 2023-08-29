"""Test ``sparse_ngd.structures.diagonal``."""

from test.structures.utils import _TestStructuredMatrix

from torch import Tensor

from sparse_ngd.structures.diagonal import DiagonalMatrix


class TestDiagonalMatrix(_TestStructuredMatrix):
    """Test suite for ``DiagonalMatrix`` class."""

    STRUCTURED_MATRIX_CLS = DiagonalMatrix

    def project(self, mat: Tensor) -> Tensor:
        """Project a matrix onto its diagonal.

        Args:
            mat: A square matrix.

        Returns:
            A matrix containing the diagonal of ``mat`` on its diagonal.
        """
        return mat.diag().diag()
