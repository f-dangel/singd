"""Block-diagonal dense matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Union

from torch import Tensor, arange, zeros

from sparse_ngd.structures.base import StructuredMatrix


class BlockDiagonalMatrixTemplate(StructuredMatrix):
    """Template for block-diagonal dense matrix.

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

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, blocks: Union[Tensor, None], last: Union[Tensor, None]) -> None:
        """Store the matrix internally.

        Args:
            blocks: The diagonal blocks ``A₁, A₂, ..., A_N`` of the matrix, supplied
                as a tensor of shape ``[N, BLOCK_DIM, BLOCK_DIM]``. If ``None``,
                there is only the ``B`` block.
            last: The last block if ``BLOCK_DIM`` does not divide the matrix dimension.
                Has shape ``[last_dim, last_dim]``. ``None`` means that ``BLOCK_DIM``
                divides the matrix dimension.

        Raises:
            ValueError: If the passed tensors have incorrect shape.
        """
        if blocks is None and last is None:
            raise ValueError(
                "Either the diagonal blocks or the last block must be tensors."
            )
        if blocks is not None:
            if blocks.dim() != 3:
                raise ValueError(
                    f"Diagonal blocks must be 3-dimensional, got {blocks.dim()}."
                )
            if blocks.shape[1] != blocks.shape[2] != self.BLOCK_DIM:
                raise ValueError(
                    f"Diagonal blocks must be square with dimension {self.BLOCK_DIM},"
                    f" got {blocks.shape[1:]} instead."
                )
        if last is not None:
            if last.dim() != 2:
                raise ValueError(f"Last block must be 2-dimensional, got {last.dim()}.")
            if last.shape[0] != last.shape[1]:
                raise ValueError(
                    f"Last block must be square, got {last.shape} instead."
                )
            if last.shape[0] >= self.BLOCK_DIM or last.shape[1] >= self.BLOCK_DIM:
                raise ValueError(
                    f"Last block must have dimension at most {self.BLOCK_DIM},"
                    f" got {last.shape} instead."
                )

        self._blocks = blocks
        self._last = last

    @classmethod
    def from_dense(cls, mat: Tensor) -> BlockDiagonalMatrixTemplate:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``BlockDiagonalMatrixTemplate``.

        Returns:
            ``BlockDiagonalMatrixTemplate`` approximating the passed matrix.
        """
        dim = cls.BLOCK_DIM
        num_blocks, last_dim = divmod(mat.shape[0], dim)

        if last_dim == 0:
            last = None
        else:
            start = 0 if num_blocks == 0 else num_blocks * dim
            last = mat[start:, :][:, start:]

        if num_blocks == 0:
            blocks = None
        else:
            mat = mat.narrow(0, 0, num_blocks * dim).narrow(1, 0, num_blocks * dim)
            mat = mat.reshape(num_blocks, dim, num_blocks, dim)
            idxs = arange(num_blocks)
            blocks = mat[idxs, :, idxs, :]

        return cls(blocks, last)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self.BLOCK_DIM

        num_blocks = 0 if self._blocks is None else self._blocks.shape[0]
        last_dim = 0 if self._last is None else self._last.shape[0]
        total_dim = num_blocks * dim + last_dim

        device = self._blocks.device if self._blocks is not None else self._last.device
        dtype = self._blocks.dtype if self._blocks is not None else self._last.dtype
        mat = zeros(total_dim, total_dim, dtype=dtype, device=device)

        if self._blocks is not None:
            for i in range(num_blocks):
                start, end = i * dim, (i + 1) * dim
                mat[start:end, :][:, start:end] = self._blocks[i, :, :]

        if self._last is not None:
            start = 0 if self._blocks is None else num_blocks * dim
            mat[start:, :][:, start:] = self._last

        return mat


class Block30DiagonalMatrix(BlockDiagonalMatrixTemplate):
    """Block-diagonal matrix with blocks of size 30."""

    BLOCK_DIM = 30


class Block3DiagonalMatrix(BlockDiagonalMatrixTemplate):
    """Block-diagonal matrix with blocks of size 3."""

    BLOCK_DIM = 3
