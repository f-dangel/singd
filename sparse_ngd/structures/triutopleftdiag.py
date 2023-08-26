"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

import torch
from torch import Tensor
import cupy as np

from sparse_ngd.structures.base import StructuredMatrix
import torch.nn.functional as F


class TriuTopLeftDiagonalMatrix(StructuredMatrix):
    """Sparse Upper-triangular matrix with Top Left Diagonal entries implemented in the ``StructuredMatrix`` interface.
        [ D  c1 ]
        [ 0  c2 ]
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK


    def __init__(self, diag:Tensor,  col: Tensor) -> None:
        """Store the matrix internally.

        Args:
            diag: the diagonal elements of the matrix
            row: the last row of the matrix
        """
        assert diag.size(0) + 1 == col.size(0)

        self._mat_col = col 
        self._mat_diag = diag

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuTopLeftDiagonalMatrix:
        """Construct from a PyTorch tensor.
        Args:
            mat: A dense and symmetric square matrix which will be approximated by a ``TriuTopLeftDiagonalMatrix``.

        Returns:
            ``TriuTopLeftDiagonalMatrix`` approximating the passed matrix.
        """


        diag = mat.diag()
        col = mat[:,-1] + mat[-1,:]
        col[-1] = diag[-1]
        return cls(diag[:-1], col)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """

        dim = self._mat_col.size(0)
        mat = torch.zeros((dim, dim))

        k = torch.tensor( range(dim-1) )
        mat[k,k] = self._mat_diag
        mat[:,-1] = self._mat_col

        return mat
