"""Test ``sparse_ngd.structures.dense``."""

from test.structures.utils import (
    _test_add,
    _test_eye,
    _test_from_inner,
    _test_from_inner2,
    _test_matmul,
    _test_mul,
    _test_rmatmat,
    _test_sub,
    _test_trace,
    _test_zeros,
)

from torch import Tensor, device, float16, float32, manual_seed, rand

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


def test_add():
    """Test matrix addition of two dense matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_add(mat1, mat2, DenseMatrix, project_dense)


def test_sub():
    """Test matrix subtraction of two dense matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_sub(mat1, mat2, DenseMatrix, project_dense)


def test_matmul():
    """Test matrix multiplication of two dense matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_matmul(mat1, mat2, DenseMatrix, project_dense)


def test_mul():
    """Test multiplication of a dense matrices with a scalar."""
    manual_seed(0)
    mat = rand((10, 10))
    factor = 0.3
    _test_mul(mat, factor, DenseMatrix, project_dense)


def test_rmatmat():
    """Test multiplication with the transpose of a dense matrix."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 20))
    _test_rmatmat(mat1, mat2, DenseMatrix, project_dense)


def test_from_inner():
    """Test structure extraction after self-inner product w/o intermediate term."""
    manual_seed(0)

    mat = rand((10, 10))
    X = None
    _test_from_inner(mat, DenseMatrix, project_dense, X)

    mat = rand((10, 10))
    X = rand((10, 20))
    _test_from_inner(mat, DenseMatrix, project_dense, X)


def test_from_inner2():
    """Test structure extraction after self-inner product w/ intermediate matrix."""
    manual_seed(0)

    mat = rand((10, 10))
    X = rand((10, 20))
    XXT = X @ X.T
    _test_from_inner2(mat, DenseMatrix, project_dense, XXT)


def test_eye():
    """Test initializing a structured matrix representing the identity matrix."""
    _test_eye(DenseMatrix, 10, float32, device("cpu"))
    _test_eye(DenseMatrix, 10, float16, device("cpu"))


def test_zeros():
    """Test initializing a structured matrix representing the zero matrix."""
    _test_zeros(DenseMatrix, 10, float32, device("cpu"))
    _test_zeros(DenseMatrix, 10, float16, device("cpu"))


def test_trace():
    """Test trace of a structured dense matrix."""
    manual_seed(0)

    mat = rand((10, 10))
    _test_trace(mat, DenseMatrix)
