"""Test ``sparse_ngd.structures.diagonal``."""

from torch import Tensor, allclose, manual_seed, rand

from sparse_ngd.structures.diagonal import DiagonalMatrix


def project(mat: Tensor) -> Tensor:
    return mat.diag().diag()


def test_matmul():
    manual_seed(0)

    mat1 = rand((10, 10))
    mat2 = rand((10, 10))
    truth = project(mat1) @ project(mat2)
    diag1_diag2 = DiagonalMatrix.from_dense(mat1) @ DiagonalMatrix.from_dense(mat2)
    assert allclose(truth, diag1_diag2.to_dense())


def test_from_inner():
    manual_seed(0)

    # X = None
    mat = rand((10, 10))
    truth = project(mat.T) @ project(mat)
    diagmat_T_diagmat = DiagonalMatrix.from_dense(mat).from_inner()
    assert allclose(truth, diagmat_T_diagmat.to_dense())

    # X != None
    mat = rand((10, 10))
    X = rand((10, 20))
    truth = project(project(mat).T @ X @ X.T @ project(mat))
    diagmat_T_X_X_T_diagmat = DiagonalMatrix.from_dense(mat).from_inner(X=X)
    assert allclose(truth, diagmat_T_X_X_T_diagmat.to_dense())
