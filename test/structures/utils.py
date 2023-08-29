"""Utility functions for testing the interface of structured matrices."""

from abc import ABC, abstractmethod
from typing import Callable, Type, Union

import torch
from torch import (
    Tensor,
    allclose,
    device,
    eye,
    float16,
    float32,
    manual_seed,
    rand,
    zeros,
)

from sparse_ngd.structures.base import StructuredMatrix


def _test_matmul(
    sym_mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``@`` operation of any child of ``StructuredMatrix``.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        mat2: Another dense matrix which will be (symmetrized then) converted into a
            structured matrix.
        structured_matrix_cls: The class of the structured matrix into which
            ``sym_mat1`` and a symmetrization of ``mat2`` will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into
            a dense matrix of the tested structure. Used to establish the ground truth.
    """
    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    sym_mat2 = symmetrize(mat2)
    sym_mat2_structured = structured_matrix_cls.from_dense(sym_mat2)

    # multiplication with a structured matrix
    truth = project(sym_mat1) @ project(sym_mat2)
    assert allclose(truth, (sym_mat1_structured @ sym_mat2_structured).to_dense())

    # multiplication with a PyTorch tensor
    truth = project(sym_mat1) @ mat2
    assert allclose(truth, sym_mat1_structured @ mat2)


def _test_add(
    sym_mat1: Tensor,
    sym_mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``+`` operation of any child of ``StructuredMatrix``.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        sym_mat2: Another symmetric dense matrix which be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which
            ``sym_mat1`` and ``sym_mat2`` will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(sym_mat1) + project(sym_mat2)
    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    sym_mat2_structured = structured_matrix_cls.from_dense(sym_mat2)
    assert allclose(truth, (sym_mat1_structured + sym_mat2_structured).to_dense())


def _test_sub(
    sym_mat1: Tensor,
    sym_mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``-`` operation of any child of ``StructuredMatrix``.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        sym_mat2: Another symmetric dense matrix which be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which
            ``sym_mat1`` and ``sym_mat2`` will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(sym_mat1) - project(sym_mat2)
    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    sym_mat2_structured = structured_matrix_cls.from_dense(sym_mat2)
    assert allclose(truth, (sym_mat1_structured - sym_mat2_structured).to_dense())


def _test_mul(
    sym_mat: Tensor,
    factor: float,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``+`` operation of any child of ``StructuredMatrix``.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        factor: Scalar which will be multiplied onto the structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``sym_mat``
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(factor * sym_mat)
    mat_structured = structured_matrix_cls.from_dense(sym_mat)
    assert allclose(truth, (mat_structured * factor).to_dense())


def _test_rmatmat(
    sym_mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``rmatmat`` operation of any child of ``StructuredMatrix``.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        mat2: A dense matrix onto which ``sym_mat1``'s structured matrix transpose
            will be multiplied onto.
        structured_matrix_cls: The class of the structured matrix into which ``mat``
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(sym_mat1).T @ mat2
    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    assert allclose(truth, sym_mat1_structured.rmatmat(mat2))


def _test_from_inner(
    sym_mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
    X: Union[Tensor, None],
):
    """Test ``from_inner`` method of any child of ``StructuredMatrix``.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which ``sym_mat``
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
        X: An optional matrix which will be passed to the ``from_inner`` method.
    """
    if X is None:
        truth = project(project(sym_mat).T @ project(sym_mat))
    else:
        truth = project(project(sym_mat).T @ X @ X.T @ project(sym_mat))

    sym_mat_structured = structured_matrix_cls.from_dense(sym_mat)
    assert allclose(truth, sym_mat_structured.from_inner(X=X).to_dense())


def _test_from_inner2(
    sym_mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
    XXT: Tensor,
):
    """Test ``from_inner2`` method of any child of ``StructuredMatrix``.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which ``sym_mat``
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
        XXT: An symmetric PSD matrix that will be passed to ``from_inner2``.
    """
    truth = project(project(sym_mat).T @ XXT @ project(sym_mat))
    sym_mat_structured = structured_matrix_cls.from_dense(sym_mat)
    assert allclose(truth, sym_mat_structured.from_inner2(XXT).to_dense())


def _test_zeros(
    structured_matrix_cls: Type[StructuredMatrix],
    dim: int,
    dtype: Union[torch.dtype, None] = None,
    device: Union[torch.device, None] = None,
):
    """Test initializing a structured matrix representing the zero matrix.

    Args:
        structured_matrix_cls: The class of the structured matrix to be tested.
        dim: Dimension of the (square) zero matrix.
        dtype: Optional data type of the matrix. If not specified, uses the default
            tensor type.
        device: Optional device of the matrix. If not specified, uses the default
            tensor type.
    """
    truth = zeros((dim, dim), dtype=dtype, device=device)
    structured_zero_matrix = structured_matrix_cls.zeros(
        dim, dtype=dtype, device=device
    )
    zero_matrix = structured_zero_matrix.to_dense()
    assert truth.dtype == zero_matrix.dtype
    assert truth.device == zero_matrix.device
    assert allclose(truth, zero_matrix)


def _test_eye(
    structured_matrix_cls: Type[StructuredMatrix],
    dim: int,
    dtype: Union[torch.dtype, None] = None,
    device: Union[torch.device, None] = None,
):
    """Test initializing a structured matrix representing the identity matrix.

    Args:
        structured_matrix_cls: The class of the structured matrix to be tested.
        dim: Dimension of the (square) zero matrix.
        dtype: Optional data type of the matrix. If not specified, uses the default
            tensor type.
        device: Optional device of the matrix. If not specified, uses the default
            tensor type.
    """
    truth = eye(dim, dtype=dtype, device=device)
    structured_identity_matrix = structured_matrix_cls.eye(
        dim, dtype=dtype, device=device
    )
    identity_matrix = structured_identity_matrix.to_dense()
    assert truth.dtype == identity_matrix.dtype
    assert truth.device == identity_matrix.device
    assert allclose(truth, identity_matrix)


def _test_trace(
    sym_mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
):
    """Test ``trace`` operation of any child of ``StructuredMatrix``.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which ``sym_mat``
            will be converted.
    """
    truth = sym_mat.trace()
    mat_structured = structured_matrix_cls.from_dense(sym_mat)
    assert allclose(truth, mat_structured.trace())


def symmetrize(mat: Tensor) -> Tensor:
    """Symmetrize a matrix.

    Args:
        mat: A square matrix.

    Returns:
        The symmetrized matrix.
    """
    return (mat + mat.T) / 2.0


class _TestStructuredMatrix(ABC):
    """Abstract class for testing ``StructuredMatrix`` implementations.

    To test a new structured matrix type, create a new class and specify the class
    attributes, then implement the ``project`` method.

    ``
    class TestDenseMatrix(_TestStructuredMatrix):
        STRUCTURED_MATRIX_CLS = DenseMatrix

        def project(self, mat: Tensor) -> Tensor:):
            ...
    ``

    ``pytest`` will automatically pick up the tests defined for ``TestDenseMatrix``
    via the base class.

    Attributes:
        STRUCTURED_MATRIX_CLS: The class of the structured matrix that is tested.
        PROJECT: A function which converts a symmetric square matrix into the structured
            tested matrix.
    """

    STRUCTURED_MATRIX_CLS: Type[StructuredMatrix]

    @abstractmethod
    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric dense matrix onto a structured matrix.

        Args:
            mat: A symmetric dense matrix.

        Returns:
            The same matrix.
        """
        raise NotImplementedError("Must be implemented by a child class")

    def test_add(self):
        """Test matrix addition of two structured matrices."""
        manual_seed(0)
        sym_mat1 = symmetrize(rand((10, 10)))
        sym_mat2 = symmetrize(rand((10, 10)))
        _test_add(sym_mat1, sym_mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    def test_sub(self):
        """Test matrix subtraction of two structured matrices."""
        manual_seed(0)
        sym_mat1 = symmetrize(rand((10, 10)))
        sym_mat2 = symmetrize(rand((10, 10)))
        _test_sub(sym_mat1, sym_mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    def test_matmul(self):
        """Test matrix multiplication of two structured matrices."""
        manual_seed(0)
        sym_mat1 = symmetrize(rand((10, 10)))
        mat2 = rand((10, 10))
        _test_matmul(sym_mat1, mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    def test_mul(self):
        """Test multiplication of a structured matrix with a scalar."""
        manual_seed(0)
        sym_mat = symmetrize(rand((10, 10)))
        factor = 0.3
        _test_mul(sym_mat, factor, self.STRUCTURED_MATRIX_CLS, self.project)

    def test_rmatmat(self):
        """Test multiplication with the transpose of a structured matrix."""
        manual_seed(0)
        sym_mat1 = symmetrize(rand((10, 10)))
        mat2 = rand((10, 20))
        _test_rmatmat(sym_mat1, mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    def test_from_inner(self):
        """Test structure extraction after self-inner product w/o intermediate term."""
        manual_seed(0)

        sym_mat = symmetrize(rand((10, 10)))
        X = None
        _test_from_inner(sym_mat, self.STRUCTURED_MATRIX_CLS, self.project, X)

        sym_mat = symmetrize(rand((10, 10)))
        X = rand((10, 20))
        _test_from_inner(sym_mat, self.STRUCTURED_MATRIX_CLS, self.project, X)

    def test_from_inner2(self):
        """Test structure extraction after self-inner product w/ intermediate matrix."""
        manual_seed(0)

        sym_mat = symmetrize(rand((10, 10)))
        X = rand((10, 20))
        XXT = X @ X.T
        _test_from_inner2(sym_mat, self.STRUCTURED_MATRIX_CLS, self.project, XXT)

    def test_eye(self):
        """Test initializing a structured matrix representing the identity matrix."""
        _test_eye(self.STRUCTURED_MATRIX_CLS, 10, float32, device("cpu"))
        _test_eye(self.STRUCTURED_MATRIX_CLS, 10, float16, device("cpu"))

    def test_zeros(self):
        """Test initializing a structured matrix representing the zero matrix."""
        _test_zeros(self.STRUCTURED_MATRIX_CLS, 10, float32, device("cpu"))
        _test_zeros(self.STRUCTURED_MATRIX_CLS, 10, float16, device("cpu"))

    def test_trace(self):
        """Test trace of a structured dense matrix."""
        manual_seed(0)

        mat = rand((10, 10))
        _test_trace(mat, self.STRUCTURED_MATRIX_CLS)
