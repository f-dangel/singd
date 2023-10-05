"""Test ``singd.structures.blockdiagonal``."""

from abc import ABC
from test.structures.utils import _TestStructuredMatrix
from typing import Type

from torch import Tensor, zeros_like

from singd.structures.blockdiagonal import (
    Block3DiagonalMatrix,
    Block30DiagonalMatrix,
    BlockDiagonalMatrixTemplate,
)


class _TestBlockDiagonalMatrix(_TestStructuredMatrix, ABC):
    """Test suite for classes created with ``BlockDiagonalMatrixTemplate``."""

    STRUCTURED_MATRIX_CLS: Type[BlockDiagonalMatrixTemplate]

    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric matrix onto a block diagonal matrix.

        Args:
            sym_mat: A symmetric matrix.

        Returns:
            A matrix containing the block diagonal of ``sym_mat`` on its diagonal.
        """
        dim = self.STRUCTURED_MATRIX_CLS.BLOCK_DIM
        num_blocks = sym_mat.shape[0] // dim

        mat = zeros_like(sym_mat)

        for i in range(num_blocks):
            start, end = i * dim, (i + 1) * dim
            mat[start:end, :][:, start:end] = sym_mat[start:end, :][:, start:end]

        start = 0 if num_blocks == 0 else num_blocks * dim
        mat[start:, :][:, start:] = sym_mat[start:, :][:, start:]

        return mat


class TestBlock30DiagonalMatrix(_TestBlockDiagonalMatrix):
    """Test suite for ``BlockDiagonal30Matrix`` class."""

    STRUCTURED_MATRIX_CLS = Block30DiagonalMatrix


class TestBlock3DiagonalMatrix(_TestBlockDiagonalMatrix):
    """Test suite for ``BlockDiagonal3Matrix`` class."""

    STRUCTURED_MATRIX_CLS = Block3DiagonalMatrix
