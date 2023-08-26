"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

import torch
from torch import Tensor
import cupy as np

from sparse_ngd.structures.base import StructuredMatrix
import torch.nn.functional as F


class TriuBottomRightDiagonalMatrix(StructuredMatrix):
    """Sparse Upper-triangular matrix with Bottom Right Diagonal entries implemented in the ``StructuredMatrix`` interface.
        [ r1 r2 ]
        [ 0  D  ]
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK


    def __init__(self, diag:Tensor,  row: Tensor) -> None:
        """Store the matrix internally.

        Args:
            diag: the diagonal elements of the matrix
            row: the first row of the matrix
        """
        assert diag.size(0) + 1 == row.size(0)

        self._mat_row = row
        self._mat_diag = diag

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuBottomRightDiagonalMatrix:
        """Construct from a PyTorch tensor.
        Args:
            mat: A dense and symmetric square matrix which will be approximated by a ``TriuBottomRightDiagonalMatrix``.

        Returns:
            ``TriuBottomRightDiagonalMatrix`` approximating the passed matrix.
        """


        diag = mat.diag()
        row = mat[:,0] + mat[0,:]
        row[0] = diag[0]
        return cls(diag[1:], row)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """

        dim = self._mat_row.size(0)
        mat = torch.zeros((dim, dim))

        k = torch.tensor( range(dim-1) ) + 1
        mat[k,k] = self._mat_diag
        mat[0,:] = self._mat_row

        return mat
