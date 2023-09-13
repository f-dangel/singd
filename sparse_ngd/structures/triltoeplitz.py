"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, arange, zeros

from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.utils import all_traces


class TrilToeplitzMatrix(StructuredMatrix):
    """Lower-triangular Toeplitz-structured matrix in ``StructuredMatrix`` interface.

    We follow the representation of such matrices using the SciPy terminology, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, diag_consts: Tensor) -> None:
        """Store the lower-triangular Toeplitz matrix internally.

        Args:
            diag_consts: A vector containing the constants of all diagonals, i.e.
                the first entry corresponds to the constant on the diagonal, the
                second entry to the constant on the lower first off-diagonal, etc.
        """
        self._mat_column = diag_consts

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tensor that need to be synchronized across devices.
        """
        return (self._mat_column,)

    @classmethod
    def from_dense(cls, mat: Tensor) -> TrilToeplitzMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TrilToeplitzMatrix``.

        Returns:
            ``TrilToeplitzMatrix`` approximating the passed matrix.
        """
        assert mat.shape[0] == mat.shape[1]
        traces = all_traces(mat)

        # sum the lower- and upper-diagonal traces
        dim = mat.shape[0]
        col = zeros(dim, dtype=mat.dtype, device=mat.device)
        idx_main = dim - 1
        col[0] += traces[idx_main]
        col[1:] += traces[idx_main + 1 :]
        col[1:] += traces[:idx_main].flip(0)

        normalization = arange(dim, 0, step=-1, dtype=mat.dtype, device=mat.device)
        col.div_(normalization)

        return cls(col)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_column.size(0)
        i, j = torch.tril_indices(row=dim, col=dim, offset=0)
        mat = torch.zeros(
            (dim, dim), device=self._mat_column.device, dtype=self._mat_column.dtype
        )
        mat[i, j] = self._mat_column[i - j]
        return mat
