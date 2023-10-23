"""Test `singd.structures.hierarchical`."""

from abc import ABC
from test.structures.utils import _TestStructuredMatrix
from typing import Type

from torch import Tensor, arange, zeros_like

from singd.structures.hierarchical import (
    Hierarchical3_2Matrix,
    Hierarchical15_15Matrix,
    HierarchicalMatrixTemplate,
)


class _TestHierarchicalMatrix(_TestStructuredMatrix, ABC):
    """Test suite for classes created with `HierarchicalMatrixTemplate`."""

    STRUCTURED_MATRIX_CLS: Type[HierarchicalMatrixTemplate]

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a hierarchical matrix.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            A matrix containing the hierarchical matrix.
        """
        dim = sym_mat.shape[0]
        hierarchical = zeros_like(sym_mat)

        if dim <= self.STRUCTURED_MATRIX_CLS.MAX_K1:
            K1, diag_dim = dim, 0
        elif (
            dim <= self.STRUCTURED_MATRIX_CLS.MAX_K1 + self.STRUCTURED_MATRIX_CLS.MAX_K2
        ):
            K1, diag_dim = self.STRUCTURED_MATRIX_CLS.MAX_K1, 0
        else:
            K1, diag_dim = (
                self.STRUCTURED_MATRIX_CLS.MAX_K1,
                dim
                - self.STRUCTURED_MATRIX_CLS.MAX_K1
                - self.STRUCTURED_MATRIX_CLS.MAX_K2,
            )

        # A
        hierarchical[:K1, :K1] = sym_mat[:K1, :K1]
        # B, NOTE that we have to sum the elements from the upper and lower block
        hierarchical[:K1, K1:] = sym_mat[:K1, K1:] + sym_mat[K1:, :K1].T
        # C
        diag_idx = arange(K1, K1 + diag_dim)
        hierarchical[diag_idx, diag_idx] = sym_mat[diag_idx, diag_idx]
        # D, NOTE that we have to sum the elements from the upper and lower block
        hierarchical[K1 + diag_dim :, K1 : K1 + diag_dim] = (
            sym_mat[K1 + diag_dim :, K1 : K1 + diag_dim]
            + sym_mat[K1 : K1 + diag_dim, K1 + diag_dim :].T
        )
        # E
        hierarchical[K1 + diag_dim :, K1 + diag_dim :] = sym_mat[
            K1 + diag_dim :, K1 + diag_dim :
        ]

        return hierarchical


class TestHierarchical15_15Matrix(_TestHierarchicalMatrix):
    """Test suite for `Hierarchical15_15Matrix` class."""

    STRUCTURED_MATRIX_CLS = Hierarchical15_15Matrix


class TestHierarchical3_2Matrix(_TestHierarchicalMatrix):
    """Test suite for `Hierarchical3_2Matrix` class."""

    DIMS = _TestHierarchicalMatrix.DIMS + [
        3,  # only A
        4,  # A, smaller E, smaller B, no D, no C
        5,  # A, normal B, no D, no C
        6,  # A, B, C, D, E
    ]
    STRUCTURED_MATRIX_CLS = Hierarchical3_2Matrix
