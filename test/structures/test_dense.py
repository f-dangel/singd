"""Test ``sparse_ngd.structures.dense``."""

from test.structures.utils import _test_from_inner, _test_matmul

from torch import Tensor, manual_seed, rand

from sparse_ngd.structures.dense import DenseMatrix


def project_dense(mat: Tensor) -> Tensor:
    """Project a dense matrix onto a dense matrix.

    This is just a no-op.

    Args:
        mat: A dense matrix.

    Returns:
        The same matrix.
    """
    return mat


def test_matmul():
    """Test matrix multiplication of two dense matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_matmul(mat1, mat2, DenseMatrix, project_dense)


def test_from_inner():
    """Test structure extraction after self-inner product w/o intermediate term."""
    manual_seed(0)

    mat = rand((10, 10))
    X = None
    _test_from_inner(mat, DenseMatrix, project_dense, X)

    mat = rand((10, 10))
    X = rand((10, 20))
    _test_from_inner(mat, DenseMatrix, project_dense, X)
