"""Hierarchically structured matrix."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor, arange, cat, ones, zeros

from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.utils import (
    diag_add_,
    supported_einsum,
    supported_eye,
    supported_matmul,
    supported_trace,
)


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
        K1, diag_dim, _ = cls._compute_block_dims(dim)

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
        diag_idx = arange(self.K1, self.K1 + self.diag_dim, device=self.A.device)
        mat[diag_idx, diag_idx] = self.C
        mat[self.K1 + self.diag_dim :, self.K1 : self.K1 + self.diag_dim] = self.D
        mat[self.K1 + self.diag_dim :, self.K1 + self.diag_dim :] = self.E

        return mat

    def __matmul__(
        self, other: Union[HierarchicalMatrixTemplate, Tensor]
    ) -> Union[HierarchicalMatrixTemplate, Tensor]:
        """Multiply a hierarchical matrix onto another one or a Tensor (@ operator).

        Args:
            other: A matrix which will be multiplied onto. Can be represented by a
                PyTorch tensor or a structured matrix.

        Returns:
            Result of the multiplication. If a PyTorch tensor was passed as argument,
            the result will be a PyTorch tensor. If a hierarchial matrix was passed,
            the result will be returned as a ``HierarchicalMatrixTemplate``.
        """
        # parts of B that share columns with C, E
        B_C, B_E = self.B.split([self.diag_dim, self.K2], dim=1)

        if isinstance(other, Tensor):
            other_top, other_middle, other_bottom = other.split(
                [self.K1, self.diag_dim, self.K2]
            )

            top = (
                supported_matmul(self.A, other_top)
                + supported_matmul(B_C, other_middle)
                + supported_matmul(B_E, other_bottom)
            )
            middle = supported_einsum("i,ij->ij", self.C, other_middle)
            bottom = supported_matmul(self.D, other_middle) + supported_matmul(
                self.E, other_bottom
            )

            return cat([top, middle, bottom], dim=0)

        else:
            A_new = supported_matmul(self.A, other.A)
            C_new = self.C * other.C
            E_new = supported_matmul(self.E, other.E)
            D_new = supported_einsum("ij,j->ij", self.D, other.C) + supported_matmul(
                self.E, other.D
            )

            B_C_other, B_E_other = other.B.split([other.diag_dim, other.K2], dim=1)
            B_new = cat(
                [
                    supported_matmul(self.A, B_C_other)
                    + supported_einsum("ij,j->ij", B_C, other.C)
                    + supported_matmul(B_E, other.D),
                    supported_matmul(self.A, B_E_other)
                    + supported_matmul(B_E, other.E),
                ],
                dim=1,
            )

            return self.__class__(A_new, B_new, C_new, D_new, E_new)

    def __add__(self, other: HierarchicalMatrixTemplate) -> HierarchicalMatrixTemplate:
        """Add with another hierarchical matrix.

        Args:
            other: Another hierarchical matrix which will be added.

        Returns:
            A hierarchical matrix resulting from the addition.
        """
        return self.__class__(
            self.A + other.A,
            self.B + other.B,
            self.C + other.C,
            self.D + other.D,
            self.E + other.E,
        )

    def __mul__(self, other: float) -> HierarchicalMatrixTemplate:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the hierarchical matrix.

        Returns:
            A hierarchical matrix resulting from the multiplication.
        """
        return self.__class__(
            self.A * other,
            self.B * other,
            self.C * other,
            self.D * other,
            self.E * other,
        )

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply ``mat`` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                hierarchical matrix.

        Returns:
            The result of the multiplication with the represented matrix's transpose.
        """
        mat_top, mat_middle, mat_bottom = mat.split([self.K1, self.diag_dim, self.K2])
        # parts of B that share columns with C, E
        B_C, B_E = self.B.split([self.diag_dim, self.K2], dim=1)

        top = supported_matmul(self.A.T, mat_top)
        middle = (
            supported_matmul(B_C.T, mat_top)
            + supported_einsum("i,ij->ij", self.C, mat_middle)
            + supported_matmul(self.D.T, mat_bottom)
        )
        bottom = supported_matmul(B_E.T, mat_top) + supported_matmul(
            self.E.T, mat_bottom
        )

        return cat([top, middle, bottom])

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def from_inner(self, X: Union[Tensor, None] = None) -> HierarchicalMatrixTemplate:
        """Represent the hierarchical matrix of ``self.T @ X @ X^T @ self``.

        Args:
            X: Optional arbitrary 2d tensor. If ``None``, ``X = I`` will be used.

        Returns:
            A ``HierarchicalMatrix`` representing hierarchical matrix of
            ``self.T @ X @ X^T @ self``.
        """
        if X is None:
            A_new = supported_matmul(self.A.T, self.A)
            B_new = supported_matmul(self.A.T, self.B)

            # parts of B that share columns with C, E
            B_C, B_E = self.B.split([self.diag_dim, self.K2], dim=1)

            C_new = self.C**2 + (B_C**2).sum(0) + (self.D**2).sum(0)
            D_new = supported_matmul(B_E.T, B_C) + supported_matmul(self.E.T, self.D)
            E_new = supported_matmul(self.E.T, self.E) + supported_matmul(B_E.T, B_E)
        else:
            S_A, S_C, S_E = self.rmatmat(X).split([self.K1, self.diag_dim, self.K2])
            A_new = supported_matmul(S_A, S_A.T)
            B_new = cat(
                [supported_matmul(S_A, S_C.T), supported_matmul(S_A, S_E.T)], dim=1
            )
            C_new = (S_C**2).sum(1)
            D_new = supported_matmul(S_E, S_C.T)
            E_new = supported_matmul(S_E, S_E.T)

        return self.__class__(A_new, B_new, C_new, D_new, E_new)

    def trace(self) -> Tensor:
        """Compute the trace of the represented matrix.

        Returns:
            The trace of the represented matrix.
        """
        return supported_trace(self.A) + self.C.sum() + supported_trace(self.E)

    def diag_add_(self, value: float) -> HierarchicalMatrixTemplate:
        """In-place add a value to the diagonal of the represented matrix.

        Args:
            value: Value to add to the diagonal.

        Returns:
            A reference to the updated matrix.
        """
        diag_add_(self.A, value)
        self.C.add_(value)
        diag_add_(self.E, value)
        return self

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################

    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> HierarchicalMatrixTemplate:
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
        K1, diag_dim, K2 = cls._compute_block_dims(dim)

        A = zeros((K1, K1), dtype=dtype, device=device)
        B = zeros((K1, diag_dim + K2), dtype=dtype, device=device)
        C = zeros(diag_dim, dtype=dtype, device=device)
        D = zeros((K2, diag_dim), dtype=dtype, device=device)
        E = zeros((K2, K2), dtype=dtype, device=device)

        return cls(A, B, C, D, E)

    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> HierarchicalMatrixTemplate:
        """Create a hierarchical matrix representing the identity matrix.

        Args:
            dim: Dimension of the (square) matrix.
            dtype: Optional data type of the matrix. If not specified, uses the default
                tensor type.
            device: Optional device of the matrix. If not specified, uses the default
                tensor type.

        Returns:
            A hierarchical matrix representing the identity matrix.
        """
        K1, diag_dim, K2 = cls._compute_block_dims(dim)

        A = supported_eye(K1, dtype=dtype, device=device)
        B = zeros((K1, diag_dim + K2), dtype=dtype, device=device)
        C = ones(diag_dim, dtype=dtype, device=device)
        D = zeros((K2, diag_dim), dtype=dtype, device=device)
        E = supported_eye(K2, dtype=dtype, device=device)

        return cls(A, B, C, D, E)

    @classmethod
    def _compute_block_dims(cls, dim: int) -> Tuple[int, int, int]:
        """Compute the dimensions of ``A, C, E``.

        Args:
            dim: Total dimension of the (square) matrix.

        Returns:
            A tuple of the form ``(K1, diag_dim, K2)``.
        """
        if dim <= cls.MAX_K1:
            K1, diag_dim, K2 = dim, 0, 0
        elif dim <= cls.MAX_K1 + cls.MAX_K2:
            K1, diag_dim, K2 = cls.MAX_K1, 0, dim - cls.MAX_K1
        else:
            K1, diag_dim, K2 = cls.MAX_K1, dim - cls.MAX_K1 - cls.MAX_K2, cls.MAX_K2
        return K1, diag_dim, K2


class Hierarchical15_15Matrix(HierarchicalMatrixTemplate):
    """Hierarchical matrix with ``K1=15`` and ``K2=15``."""

    MAX_K1 = 15
    MAX_K2 = 15


class Hierarchical3_2Matrix(HierarchicalMatrixTemplate):
    """Hierarchical matrix with ``K1=3`` and ``K2=2``."""

    MAX_K1 = 3
    MAX_K2 = 2
