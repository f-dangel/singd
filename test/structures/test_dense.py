"""Test ``sparse_ngd.structures.dense``."""

from torch import Tensor, allclose, manual_seed, rand

from sparse_ngd.structures.dense import DenseMatrix


def project(mat: Tensor) -> Tensor:
    return mat


def test_matmul():
    manual_seed(0)

    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    truth = project(mat1) @ project(mat2)
    mat1_mat2 = DenseMatrix.from_dense(mat1) @ DenseMatrix.from_dense(mat2)
    assert allclose(truth, mat1_mat2.to_dense())


def test_from_inner():
    manual_seed(0)

    # X = None
    mat = rand((10, 10))
    truth = project(mat.T) @ project(mat)
    mat_T_mat = DenseMatrix.from_dense(mat).from_inner()
    assert allclose(truth, mat_T_mat.to_dense())

    # X != None
    mat = rand((10, 10))
    X = rand((10, 20))
    truth = project(mat).T @ X @ X.T @ project(mat)
    mat_T_X_X_T_mat = DenseMatrix.from_dense(mat).from_inner(X=X)
    assert allclose(truth, mat_T_X_X_T_mat.to_dense())
