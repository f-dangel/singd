"""Block-diagonal dense matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, arange, cat, zeros

from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.utils import (
    supported_einsum,
    supported_eye,
    supported_matmul,
    supported_trace,
)


class BlockDiagonalMatrixTemplate(StructuredMatrix):
    """Template for symmetric block-diagonal dense matrix.

    ``
    [[A₁, 0,   ..., 0,   0],
     [0,  A₂,  0,   ..., 0],
     [0,  0,   ..., 0,   0],
     [0,  ..., 0,   A_N, 0],
     [0,  0,   0,   0,   B]]
    ``

    where
    - ``A₁, ..., A_N`` are symmetric matrices of size ``block_dim``
    - ``B`` is a symmetric matrix of size ``last_dim`` if ``block_dim`` does not divide
      the total matrix dimension.

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the ``BLOCK_DIM`` class attribute.

    Attributes:
        BLOCK_DIM: The dimension of a diagonal block.
    """

    BLOCK_DIM: int

    def __init__(self, blocks: Tensor, last: Tensor) -> None:
        """Store the matrix internally.

        Args:
            blocks: The diagonal blocks ``A₁, A₂, ..., A_N`` of the matrix, supplied
                as a tensor of shape ``[N, BLOCK_DIM, BLOCK_DIM]``. If there are no
                blocks, has shape ``[0, BLOCK_DIM, BLOCK_DIM]``.
            last: The last block if ``BLOCK_DIM`` which contains the remaining matrix
                if ``BLOCK_DIM`` does not divide the matrix dimension.
                Has shape ``[last_dim, last_dim]`` where ``last_dim`` may be zero.

        Raises:
            ValueError: If the passed tensors have incorrect shape.
        """
        if blocks.dim() != 3:
            raise ValueError(
                f"Diagonal blocks must be 3-dimensional, got {blocks.dim()}."
            )
        if blocks.shape[1] != blocks.shape[2] != self.BLOCK_DIM:
            raise ValueError(
                f"Diagonal blocks must be square with dimension {self.BLOCK_DIM},"
                f" got {blocks.shape[1:]} instead."
            )
        if last.dim() != 2 or last.shape[0] != last.shape[1]:
            raise ValueError(f"Last block must be square, got {last.shape}.")
        if last.shape[0] >= self.BLOCK_DIM or last.shape[1] >= self.BLOCK_DIM:
            raise ValueError(
                f"Last block must have dimension at most {self.BLOCK_DIM},"
                f" got {last.shape} instead."
            )
        self._blocks = blocks
        self._last = last

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor, Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tuple of tensors that need to be synchronized across devices.
        """
        return (self._blocks, self._last)

    @classmethod
    def from_dense(cls, mat: Tensor) -> BlockDiagonalMatrixTemplate:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``BlockDiagonalMatrixTemplate``.

        Returns:
            ``BlockDiagonalMatrixTemplate`` approximating the passed matrix.
        """
        num_blocks = mat.shape[0] // cls.BLOCK_DIM

        last_start = num_blocks * cls.BLOCK_DIM
        last = mat[last_start:, last_start:]

        blocks_end = num_blocks * cls.BLOCK_DIM
        mat = mat.narrow(0, 0, blocks_end).narrow(1, 0, blocks_end)
        mat = mat.reshape(num_blocks, cls.BLOCK_DIM, num_blocks, cls.BLOCK_DIM)
        idxs = arange(num_blocks, device=mat.device)
        blocks = mat[idxs, :, idxs, :]

        return cls(blocks, last)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        num_blocks = self._blocks.shape[0]
        last_dim = self._last.shape[0]
        total_dim = num_blocks * self.BLOCK_DIM + last_dim

        mat = zeros(
            total_dim, total_dim, dtype=self._blocks.dtype, device=self._blocks.device
        )

        for i in range(num_blocks):
            start, end = i * self.BLOCK_DIM, (i + 1) * self.BLOCK_DIM
            mat[start:end, start:end] = self._blocks[i, :, :]

        last_start = num_blocks * self.BLOCK_DIM
        mat[last_start:, last_start:] = self._last

        return mat

    def __matmul__(
        self, other: Union[BlockDiagonalMatrixTemplate, Tensor]
    ) -> Union[BlockDiagonalMatrixTemplate, Tensor]:
        """Multiply with another block-diagonal matrix or PyTorch tensor (@ operator).

        Args:
            other: A matrix which will be multiplied onto. Can be represented by a
                PyTorch tensor or a ``BlockDiagonalMatrix``.

        Returns:
            Result of the multiplication. If a PyTorch tensor was passed as argument,
            the result will be a PyTorch tensor. If a block-diagonal matrix was passed,
            the result will be returned as a ``BlockDiagonalMatrixTemplate``.

        Raises:
            ValueError: If ``other``'s shape is incompatible.
        """
        if isinstance(other, Tensor):
            num_blocks, last_dim = self._blocks.shape[0], self._last.shape[0]
            total_dim = num_blocks * self.BLOCK_DIM + last_dim
            if other.shape[0] != total_dim or other.dim() != 2:
                raise ValueError(
                    f"Expect matrix with {total_dim} rows. Got {other.shape}."
                )
            other_blocks, other_last = other.split(
                [num_blocks * self.BLOCK_DIM, last_dim]
            )

            dims = {"block": num_blocks, "row": self.BLOCK_DIM}
            other_blocks = rearrange(
                other_blocks, "(block row) col -> block row col", **dims
            )
            result_blocks = supported_einsum("nij,njk->nik", self._blocks, other_blocks)
            result_blocks = rearrange(
                result_blocks, "block row col -> (block row) col", **dims
            )

            result_last = supported_matmul(self._last, other_last)

            return cat([result_blocks, result_last])

        else:
            out_blocks = supported_einsum("nij,njk->nik", self._blocks, other._blocks)
            out_last = supported_matmul(self._last, other._last)
            return self.__class__(out_blocks, out_last)

    def __add__(
        self, other: BlockDiagonalMatrixTemplate
    ) -> BlockDiagonalMatrixTemplate:
        """Add with another block-diagonal matrix.

        Args:
            other: Another block-diagonal matrix which will be added.

        Returns:
            A block-diagonal matrix resulting from the addition.
        """
        return self.__class__(self._blocks + other._blocks, self._last + other._last)

    def __mul__(self, other: float) -> BlockDiagonalMatrixTemplate:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the diagonal matrix.

        Returns:
            A diagonal matrix resulting from the multiplication.
        """
        return self.__class__(other * self._blocks, other * self._last)

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply ``mat`` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                block-diagonal matrix.

        Returns:
            The result of the multiplication with the represented matrix's transpose.
        """
        return self @ mat

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def from_inner(self, X: Union[Tensor, None] = None) -> BlockDiagonalMatrixTemplate:
        """Represent the matrix block-diagonal of ``self.T @ X @ X^T @ self``.

        Let ``K := self``. We can first re-write ``K.T @ X @ X^T @ K`` into
        ``S @ S.T`` where ``S = K.T @ X``. Next, note that ``S`` has block structure:
        Write ``K := blockdiag(K₁, K₂, ...)`` and write ``X`` as a stack of matrices
        ``X = vstack(X₁, X₂, ...)`` where ``Xᵢ`` is associated with the ``i``th diagonal
        block. Then ``S = vstack( K₁.T @ X₁, K₂ @ X₂, ...) = vstack(S₁ S₂, ...)`` where
        we have introduced ``Sᵢ = Kᵢ.T @ Xᵢ``. Consequently, ``S @ S.T`` consists of
        blocks ``(i, j)`` with structure ``Sᵢ @ Sⱼ.T``. We are only interested in the
        diagonal blocks. So we need to compute
        ``Sᵢ @ Sᵢ.T = (Kᵢ.T @ Xᵢ) @ (Kᵢ.T @ Xᵢ).T`` for all ``i``.

        Args:
            X: Optional arbitrary 2d tensor. If ``None``, ``X = I`` will be used.

        Returns:
            A ``DiagonalMatrix`` representing matrix block diagonal of
            ``self.T @ X @ X^T @ self``.
        """
        if X is None:
            S_blocks, S_last = self._blocks, self._last
        else:
            num_blocks, last_dim = self._blocks.shape[0], self._last.shape[0]
            S_blocks, S_last = self.rmatmat(X).split(
                [num_blocks * self.BLOCK_DIM, last_dim]
            )
            dims = {"block": num_blocks, "row": self.BLOCK_DIM}
            S_blocks = rearrange(S_blocks, "(block row) col -> block row col", **dims)

        out_blocks = supported_einsum("nij,nkj->nik", S_blocks, S_blocks)
        out_last = supported_matmul(S_last, S_last.T)

        return self.__class__(out_blocks, out_last)

    def trace(self) -> Tensor:
        """Compute the trace of the represented matrix.

        Returns:
            The trace of the represented matrix.
        """
        return supported_einsum("nii->", self._blocks) + supported_trace(self._last)

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################

    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> BlockDiagonalMatrixTemplate:
        """Create a structured matrix representing the zero matrix.

        Args:
            dim: Dimension of the (square) matrix.
            dtype: Optional data type of the matrix. If not specified, uses the default
                tensor type.
            device: Optional device of the matrix. If not specified, uses the default
                tensor type.

        Returns:
            A structured matrix representing the zero matrix.
        """
        num_blocks, last_dim = divmod(dim, cls.BLOCK_DIM)
        blocks = zeros(
            num_blocks, cls.BLOCK_DIM, cls.BLOCK_DIM, dtype=dtype, device=device
        )
        last = zeros(last_dim, last_dim, dtype=dtype, device=device)
        return cls(blocks, last)

    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> BlockDiagonalMatrixTemplate:
        """Create a structured matrix representing the identity matrix.

        Args:
            dim: Dimension of the (square) matrix.
            dtype: Optional data type of the matrix. If not specified, uses the default
                tensor type.
            device: Optional device of the matrix. If not specified, uses the default
                tensor type.

        Returns:
            A block-diagonal matrix representing the identity matrix.
        """
        num_blocks, last_dim = divmod(dim, cls.BLOCK_DIM)

        one_block = supported_eye(cls.BLOCK_DIM, dtype=dtype, device=device)
        blocks = one_block.unsqueeze(0).repeat(num_blocks, 1, 1)

        last = supported_eye(last_dim, dtype=dtype, device=device)

        return cls(blocks, last)


class Block30DiagonalMatrix(BlockDiagonalMatrixTemplate):
    """Block-diagonal matrix with blocks of size 30."""

    BLOCK_DIM = 30


class Block3DiagonalMatrix(BlockDiagonalMatrixTemplate):
    """Block-diagonal matrix with blocks of size 3."""

    BLOCK_DIM = 3
