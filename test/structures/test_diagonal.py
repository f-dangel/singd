"""Test ``sparse_ngd.structures.diagonal``."""

from test.structures.utils import (
    _test_add,
    _test_eye,
    _test_from_inner,
    _test_from_inner2,
    _test_matmul,
    _test_mul,
    _test_sub,
    _test_trace,
    _test_transpose,
    _test_zeros,
)

from torch import Tensor, device, float16, float32, manual_seed, rand

from sparse_ngd.structures.diagonal import DiagonalMatrix


def project_diagonal(mat: Tensor) -> Tensor:
    """Project a matrix onto its diagonal.

    Args:
        mat: A square matrix.

    Returns:
        A matrix containing the diagonal of ``mat`` on its diagonal.
    """
    return mat.diag().diag()


def test_matmul():
    """Test matrix multiplication of two diagonal matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_matmul(mat1, mat2, DiagonalMatrix, project_diagonal)


def test_mul():
    """Test multiplication of a diagonal matrix with a scalar."""
    manual_seed(0)
    mat = rand((10, 10))
    scale = 0.3
    _test_mul(mat, scale, DiagonalMatrix, project_diagonal)


def test_transpose():
    """Test transpose of a diagonal matrix."""
    manual_seed(0)
    mat = rand((10, 10))
    _test_transpose(mat, DiagonalMatrix, project_diagonal)


def test_add():
    """Test matrix addition of two diagonal matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_add(mat1, mat2, DiagonalMatrix, project_diagonal)


def test_sub():
    """Test matrix subtractionof two diagonal matrices."""
    manual_seed(0)
    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    _test_sub(mat1, mat2, DiagonalMatrix, project_diagonal)


def test_from_inner():
    """Test diagonal extraction after self-inner product w/o intermediate term."""
    manual_seed(0)

    mat = rand((10, 10))
    X = None
    _test_from_inner(mat, DiagonalMatrix, project_diagonal, X)

    mat = rand((10, 10))
    X = rand((10, 20))
    _test_from_inner(mat, DiagonalMatrix, project_diagonal, X)


def test_from_inner2():
    """Test diagonal extraction after self-inner product w/ intermediate matrix."""
    manual_seed(0)

    mat = rand((10, 10))
    X = rand((10, 20))
    XXT = X @ X.T
    _test_from_inner2(mat, DiagonalMatrix, project_diagonal, XXT)


def test_eye():
    """Test initializing a diagonal matrix representing the identity matrix."""
    _test_eye(DiagonalMatrix, 10, float32, device("cpu"))
    _test_eye(DiagonalMatrix, 10, float16, device("cpu"))


def test_zeros():
    """Test initializing a diagonal matrix representing the zero matrix."""
    _test_zeros(DiagonalMatrix, 10, float32, device("cpu"))
    _test_zeros(DiagonalMatrix, 10, float16, device("cpu"))


def test_trace():
    """Test trace of a structured dense matrix."""
    manual_seed(0)

    mat = rand((10, 10))
    _test_trace(mat, DiagonalMatrix)
