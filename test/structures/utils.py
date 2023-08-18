"""Utility functions for testing the interface of structured matrices."""

from typing import Callable, Type, Union

import torch
from torch import Tensor, allclose, eye, zeros

from sparse_ngd.structures.base import StructuredMatrix


def _test_matmul(
    mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``@`` operation of any child of ``StructuredMatrix``.

    Args:
        mat1: A dense matrix which will be converted into a structured matrix.
        mat2: Another dense matrix which be converted into a structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat1``
            and ``mat2`` will be converted.
        project: A function which converts an arbitrary dense matrix into a dense
            matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(mat1) @ project(mat2)
    mat1_structured = structured_matrix_cls.from_dense(mat1)
    mat2_structured = structured_matrix_cls.from_dense(mat2)
    assert allclose(truth, (mat1_structured @ mat2_structured).to_dense())


def _test_add(
    mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``+`` operation of any child of ``StructuredMatrix``.

    Args:
        mat1: A dense matrix which will be converted into a structured matrix.
        mat2: Another dense matrix which be converted into a structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat1``
            and ``mat2`` will be converted.
        project: A function which converts an arbitrary dense matrix into a dense
            matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(mat1) + project(mat2)
    mat1_structured = structured_matrix_cls.from_dense(mat1)
    mat2_structured = structured_matrix_cls.from_dense(mat2)
    assert allclose(truth, (mat1_structured + mat2_structured).to_dense())


def _test_sub(
    mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``-`` operation of any child of ``StructuredMatrix``.

    Args:
        mat1: A dense matrix which will be converted into a structured matrix.
        mat2: Another dense matrix which be converted into a structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat1``
            and ``mat2`` will be converted.
        project: A function which converts an arbitrary dense matrix into a dense
            matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(mat1) - project(mat2)
    mat1_structured = structured_matrix_cls.from_dense(mat1)
    mat2_structured = structured_matrix_cls.from_dense(mat2)
    assert allclose(truth, (mat1_structured - mat2_structured).to_dense())


def _test_mul(
    mat: Tensor,
    factor: float,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test ``+`` operation of any child of ``StructuredMatrix``.

    Args:
        mat: A dense matrix which will be converted into a structured matrix.
        factor: Scalar which will be multiplied onto the structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat1``
            and ``mat2`` will be converted.
        project: A function which converts an arbitrary dense matrix into a dense
            matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(factor * mat)
    mat_structured = structured_matrix_cls.from_dense(mat)
    assert allclose(truth, (mat_structured * factor).to_dense())


def _test_from_inner(
    mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
    X: Union[Tensor, None],
):
    """Test ``from_inner`` method of any child of ``StructuredMatrix``.

    Args:
        mat: A dense matrix which will be converted into a structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat``
            will be converted.
        project: A function which converts an arbitrary dense matrix into a dense
            matrix of the tested structure. Used to establish the ground truth.
        X: An optional matrix which will be passed to the ``from_inner`` method.
    """
    if X is None:
        truth = project(project(mat).T @ project(mat))
    else:
        truth = project(project(mat).T @ X @ X.T @ project(mat))

    mat_structured = structured_matrix_cls.from_dense(mat)
    assert allclose(truth, mat_structured.from_inner(X=X).to_dense())


def _test_from_inner2(
    mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
    XXT: Tensor,
):
    """Test ``from_inner2`` method of any child of ``StructuredMatrix``.

    Args:
        mat: A dense matrix which will be converted into a structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat``
            will be converted.
        project: A function which converts an arbitrary dense matrix into a dense
            matrix of the tested structure. Used to establish the ground truth.
        XXT: An symmetric square matrix that will be passed to ``from_inner2``.
    """
    truth = project(project(mat).T @ XXT @ project(mat))
    mat_structured = structured_matrix_cls.from_dense(mat)
    assert allclose(truth, mat_structured.from_inner2(XXT).to_dense())


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
    mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
):
    """Test ``trace`` operation of any child of ``StructuredMatrix``.

    Args:
        mat: A dense matrix which will be converted into a structured matrix.
        structured_matrix_cls: The class of the structured matrix into which ``mat``
            will be converted.
    """
    truth = mat.trace()
    mat_structured = structured_matrix_cls.from_dense(mat)
    assert allclose(truth, mat_structured.trace())
