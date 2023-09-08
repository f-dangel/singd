"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

# import cupy as np
import numpy as np
import torch
from torch import Tensor

from sparse_ngd.structures.base import StructuredMatrix


class TriuToeplitzMatrix(StructuredMatrix):
    """Upper-triangular Toeplitz-structured matrix.

    We follow the representation of such matrices using the SciPy terminology, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, diag_consts: Tensor) -> None:
        """Store the upper-triangular Toeplitz matrix internally.

        Args:
            diag_consts: A vector containing the constants of all diagonals, i.e.
                the first entry corresponds to the constant on the diagonal, the
                second entry to the constant on the upper first off-diagonal, etc.
        """
        self._mat_row = diag_consts

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuToeplitzMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TriuToeplitzMatrix``.

        Returns:
            ``TriuToeplitzMatrix`` approximating the passed matrix.
        """
        # Reference:
        # https://stackoverflow.com/questions/57347896/sum-all-diagonals-in-feature-maps-in-parallel-in-pytorch
        # Note the conv2d is too slow when dim is large
        dim = mat.size(0)
        x = torch.fliplr(mat)
        digitized = np.sum(np.indices(x.shape), axis=0).ravel()
        # digitized_tensor = torch.from_numpy(digitized) #using numpy
        digitized_tensor = torch.as_tensor(digitized).to(
            x.device
        )  # using cupy instead of numpy to avoid a cpu-to-gpu call
        result = torch.bincount(digitized_tensor, x.view(-1))

        row = result[range(dim)].flip(0) + result[(dim - 1) :]
        row.div_((1.0 + torch.Tensor(range(dim)).to(row.device)).flip(0))
        row[0] = row[0] / 2.0

        row = row.to(mat.device).to(mat.dtype)

        return cls(row)  # the same as the tril case

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_row.size(0)
        i, j = torch.triu_indices(row=dim, col=dim, offset=0)
        mat = torch.zeros(
            (dim, dim), dtype=self._mat_row.dtype, device=self._mat_row.device
        )
        mat[i, j] = self._mat_row[j - i]
        return mat
