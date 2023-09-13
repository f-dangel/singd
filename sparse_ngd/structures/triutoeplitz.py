"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor, arange, triu_indices, zeros

from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.utils import all_traces


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

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tensor that need to be synchronized across devices.
        """
        return (self._mat_row,)

    @classmethod
    def from_dense(cls, mat: Tensor) -> TriuToeplitzMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense and symmetric square matrix which will be approximated by a
                ``TriuToeplitzMatrix``.

        Returns:
            ``TriuToeplitzMatrix`` approximating the passed matrix.
        """
        assert mat.shape[0] == mat.shape[1]
        traces = all_traces(mat)

        # sum the lower- and upper-diagonal traces
        dim = mat.shape[0]
        row = zeros(dim, dtype=mat.dtype, device=mat.device)
        idx_main = dim - 1
        row[0] += traces[idx_main]
        row[1:] += traces[idx_main + 1 :]
        row[1:] += traces[:idx_main].flip(0)

        normalization = arange(dim, 0, step=-1, dtype=mat.dtype, device=mat.device)
        row.div_(normalization)

        return cls(row)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        dim = self._mat_row.shape[0]
        i, j = triu_indices(row=dim, col=dim, offset=0)
        mat = zeros((dim, dim), dtype=self._mat_row.dtype, device=self._mat_row.device)
        mat[i, j] = self._mat_row[j - i]
        return mat
