"""Test ``sparse_ngd.structures.dense``."""

from torch import allclose, manual_seed, rand

from sparse_ngd.structures.dense import DenseMatrix


def test_matmul():
    manual_seed(0)

    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    truth = mat1 @ mat2
    mat1mat2 = DenseMatrix(mat1) @ DenseMatrix(mat2)
    assert allclose(truth, mat1mat2.to_dense())


def test_from_inner():
    manual_seed(0)

    # X = None
    mat = rand((10, 10))
    truth = mat.T @ mat
    mat_T_mat = DenseMatrix(mat).from_inner()
    assert allclose(truth, mat_T_mat.to_dense())

    # X != None
    mat = rand((10, 10))
    X = rand((10, 20))
    truth = mat.T @ X @ X.T @ mat
    mat_T_X_X_T_mat = DenseMatrix(mat).from_inner(X=X)
    assert allclose(truth, mat_T_X_X_T_mat.to_dense())


def test_from_dense():
    manual_seed(0)

    mat = rand((10, 10))
    assert allclose(mat, DenseMatrix.from_dense(mat).to_dense())
