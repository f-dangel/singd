"""Hierarchically structured matrix."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor, arange, zeros

from sparse_ngd.structures.base import StructuredMatrix


class HierarchicalMatrixTemplate(StructuredMatrix):
    """Template for hierarchical matrices.

    ``[[A,   B ],
      [ 0, C, 0],
      [ 0, D, E],]``

    where (denoting ``K`` the matrix dimension)

    - ``A`` is dense square and has shape ``[K1, K1]``
    - ``B`` is dense rectangular of shape ``[K1, K - K1]``
    - ``C`` is a diagonal matrix of shape ``[K - K1 - K2, K - K1 - K2]``
    - ``D`` is dense rectangular of shape ``[K2, K - K1 - K2]``
    - ``E`` is dense square and has shape ``[K2, K2]``

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the ``MAX_K1`` and ``MAX_K2`` class attributes.

    Given specific values for ``K1, K2``, if the matrix to be represented is not
    big enough to fit all structures, we use the following prioritization:

    1. If ``K <= K1``, start by filling ``A``.
    2. If ``K1 < K <= K1+K2``, fill ``A`` and start filling ``B`` and ``E``.
    3. If ``K1+K2 < K``, use all structures.

    Attributes:
        MAX_K1: Maximum dimension of the top left.
        MAX_K2: Maximum dimension of the bottom right block.
    """

    MAX_K1: int
    MAX_K2: int
    WARN_NAIVE: bool = False  # TODO Implement optimized interface functions

    def __init__(self, A: Tensor, B: Tensor, C: Tensor, D: Tensor, E: Tensor):
        """Store the structural components internally.

        Please read the class docstring for more information.

        Args:
            A: Dense square matrix of shape ``[K1, K1]`` or smaller.
            B: Dense rectangular matrix of shape ``[K1, K - K1]``.
            C: Diagonal of shape ``[K - K1 - K2]``.
            D: Dense rectangular matrix of shape ``[K2, K - K1 - K2]``.
            E: Dense square matrix of shape ``[K2, K2]`` or smaller.

        Raises:
            ValueError: If the shapes of the arguments are invalid.
        """
        if A.dim() != 2 or B.dim() != 2 or C.dim() != 1 or D.dim() != 2 or E.dim() != 2:
            raise ValueError(
                "Invalid tensor dimensions. Expected 2, 2, 1, 2, 2."
                + f" Got {A.dim()}, {B.dim()}, {C.dim()}, {D.dim()}, {E.dim()}."
            )
        if A.shape[0] != A.shape[1] or E.shape[0] != E.shape[1]:
            raise ValueError(f"Expected square A, E. Got {A.shape}, {E.shape}.")

        self.K1 = A.shape[0]
        self.K2 = E.shape[0]
        self.diag_dim = C.shape[0]
        self.dim = self.K1 + self.K2 + self.diag_dim

        if A.shape[0] > self.MAX_K1 or E.shape[0] > self.MAX_K2:
            raise ValueError(
                f"Expected A, E to be smaller than {self.MAX_K1}, {self.MAX_K2}."
                + f" Got {A.shape}, {E.shape}."
            )
        if D.shape != (self.K2, self.diag_dim):
            raise ValueError(
                f"Expected D to have shape {self.K2, self.diag_dim}. Got {D.shape}."
            )
        if B.shape != (self.K1, self.diag_dim + self.K2):
            raise ValueError(
                f"Expected B to have shape {self.K1, self.diag_dim + self.K2}."
                + " Got {B.shape}."
            )

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training. If ``None``,
        this structured matrix does not support distributed data parallel training.

        Returns:
            A tuple of tensors that needs to be synchronized across devices.
        """
        return (self.A, self.B, self.C, self.D, self.E)

    @classmethod
    def from_dense(cls, sym_mat: Tensor) -> HierarchicalMatrixTemplate:
        """Construct from a PyTorch tensor.

        Args:
            sym_mat: A dense symmetric matrix which will be represented as
                ``Hierarchical``.

        Returns:
            ``HierarchicalMatrix`` representing the passed matrix.

        Raises:
            ValueError: If the passed tensor is not square.
        """
        if sym_mat.shape[0] != sym_mat.shape[1] or sym_mat.dim() != 2:
            raise ValueError(
                f"Expected square matrix. Got tensor shape {sym_mat.shape}."
            )
        dim = sym_mat.shape[0]

        if dim <= cls.MAX_K1:
            K1, diag_dim = dim, 0
        elif dim <= cls.MAX_K1 + cls.MAX_K2:
            K1, diag_dim = cls.MAX_K1, 0
        else:
            K1, diag_dim = cls.MAX_K1, dim - cls.MAX_K1 - cls.MAX_K2

        A = sym_mat[:K1, :K1]
        B = sym_mat[:K1, K1:]
        C = sym_mat.diag()[K1 : K1 + diag_dim]
        D = sym_mat[K1 + diag_dim :, K1 : K1 + diag_dim]
        E = sym_mat[K1 + diag_dim :, K1 + diag_dim :]

        return cls(A, B, C, D, E)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        mat = zeros(self.dim, self.dim, dtype=self.A.dtype, device=self.A.device)

        mat[: self.K1, : self.K1] = self.A
        mat[: self.K1, self.K1 :] = self.B
        diag_idx = arange(self.K1, self.K1 + self.diag_dim)
        mat[diag_idx, diag_idx] = self.C
        mat[self.K1 + self.diag_dim :, self.K1 : self.K1 + self.diag_dim] = self.D
        mat[self.K1 + self.diag_dim :, self.K1 + self.diag_dim :] = self.E

        return mat


class Hierarchical15_15Matrix(HierarchicalMatrixTemplate):
    """Hierarchical matrix with ``K1=15`` and ``K2=15``."""

    MAX_K1 = 15
    MAX_K2 = 15


class Hierarchical3_2Matrix(HierarchicalMatrixTemplate):
    """Hierarchical matrix with ``K1=3`` and ``K2=2``."""

    MAX_K1 = 3
    MAX_K2 = 2
