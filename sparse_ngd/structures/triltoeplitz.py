"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

import torch
from torch import Tensor
import cupy as np

from sparse_ngd.structures.base import StructuredMatrix
import torch.nn.functional as F


class TrilToeplitzMatrix(StructuredMatrix):
    """Lower-triangular Toeplitz-structured matrix implemented in the ``StructuredMatrix`` interface.

    We follow the representation of such matrices using the SciPy terminology, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK


    def __init__(self, col: Tensor) -> None:
        """Store the lower-triangular Toeplitz matrix internally.

        Args:
            c: the first column of the matrix
        """
        self._mat_column = col

    @classmethod
    def from_dense(cls, mat: Tensor) -> TrilToeplitzMatrix:
        """Construct from a PyTorch tensor.
        Args:
            mat: A dense and symmetric square matrix which will be approximated by a ``TrilToeplitzMatrix``.

        Returns:
            ``TrilToeplitzMatrix`` representing the passed matrix.
        """
        #Reference:
        # https://stackoverflow.com/questions/57347896/sum-all-diagonals-in-feature-maps-in-parallel-in-pytorch
        # Note the conv2d is too slow when dim is large

        dim = mat.size(0)
        x = torch.fliplr(mat)
        digitized = np.sum(np.indices(x.shape), axis=0).ravel()
        # digitized_tensor = torch.from_numpy(digitized) #using numpy
        digitized_tensor = torch.as_tensor(digitized) #using cupy
        result = torch.bincount(digitized_tensor, x.view(-1))

        col = result [ range(dim) ].flip(0) + result [ (dim-1): ]
        col.div_( (1.0 + torch.Tensor(range(dim)) ).flip(0) )
        col[0] = col[0]/2.0

        return cls(col)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_column.size(0)
        i,j  = torch.tril_indices(row=dim, col=dim, offset=0)
        mat = torch.zeros((dim, dim))
        mat[i,j] = self._mat_column[i-j]
        return mat
