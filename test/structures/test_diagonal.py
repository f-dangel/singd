"""Test ``sparse_ngd.structures.diagonal``."""

from test.structures.utils import _test_from_inner, _test_matmul

from torch import Tensor, manual_seed, rand

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


def test_from_inner():
    """Test diagonal extraction after self-inner product w/o intermediate term."""
    manual_seed(0)

    mat = rand((10, 10))
    X = None
    _test_from_inner(mat, DiagonalMatrix, project_diagonal, X)

    mat = rand((10, 10))
    X = rand((10, 20))
    _test_from_inner(mat, DiagonalMatrix, project_diagonal, X)
