"""Hierarchically structured matrix."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor, arange, cat, einsum, ones, zeros
from torch.linalg import vector_norm

from singd.structures.base import StructuredMatrix
from singd.structures.utils import diag_add_, lowest_precision, supported_eye


class HierarchicalMatrixTemplate(StructuredMatrix):
    r"""Template class for creating hierarchical matrices.

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the `MAX_K1` and `MAX_K2` class attributes. See the example below.

    Hierarchical matrices have the following structure:

    \(
    \begin{pmatrix}
    \mathbf{A} & \mathbf{B}_1 & \mathbf{B}_2 \\
    \mathbf{0} & \mathbf{C} & \mathbf{0} \\
    \mathbf{0} & \mathbf{D} & \mathbf{E} \\
    \end{pmatrix}
    \in \mathbb{R}^{K \times K}
    \)

    where (denoting
    \(\mathbf{B} := \begin{pmatrix}\mathbf{B}_1 & \mathbf{B}_2\end{pmatrix}\))

    - \(\mathbf{A} \in \mathbb{R}^{K_1 \times K_1}\) is dense symmetric
    - \(\mathbf{B} \in \mathbb{R}^{K_1 \times (K - K_1)}\) is dense rectangular
    - \(\mathbf{C} \in \mathbb{R}^{(K - K_2 - K_1) \times (K - K_2 - K_1)}\) is diagonal
    - \(\mathbf{D} \in \mathbb{R}^{K_2 \times (K - K_2 - K_1)}\) is dense rectangular
    - \(\mathbf{E} \in \mathbb{R}^{K_2 \times K_2}\) is dense symmetric

    For fixed values of \(K_1, K_2\), if the matrix to be represented is not
    big enough to fit all structures, we use the following prioritization:

    1. If \(K \le K_1\), start by filling \(\mathbf{A}\).
    2. If \(K_1 < K \le K_1+K_2\), fill \(\mathbf{A}\) and start filling \(\mathbf{B}\)
       and \(\mathbf{E}\).
    3. If \(K_1+K_2 < K\), use all structures.

    Attributes:
        MAX_K1: Maximum dimension \(K_1\) of the top left block \(\mathbf{A}\).
        MAX_K2: Maximum dimension \(K_2\) of the bottom right block \(\mathbf{E}\).

    Examples:
        >>> from torch import ones
        >>>
        >>> class Hierarchical2_3Matrix(HierarchicalMatrixTemplate):
        ...     '''Hierarchical matrix with 2x2 top left and 3x3 bottom right block.'''
        ...     MAX_K1 = 2
        ...     MAX_K2 = 3
        >>>
        >>> # A hierarchical matrix with total dimension K=7
        >>> A, C, E = ones(2, 2), 3 * ones(2), 5 * ones(3, 3)
        >>> B, D = 2 * ones(2, 5), 4 * ones(3, 2)
        >>> mat = Hierarchical2_3Matrix(A, B, C, D, E)
        >>> mat.to_dense()
        tensor([[1., 1., 2., 2., 2., 2., 2.],
                [1., 1., 2., 2., 2., 2., 2.],
                [0., 0., 3., 0., 0., 0., 0.],
                [0., 0., 0., 3., 0., 0., 0.],
                [0., 0., 4., 4., 5., 5., 5.],
                [0., 0., 4., 4., 5., 5., 5.],
                [0., 0., 4., 4., 5., 5., 5.]])
    """

    MAX_K1: int
    MAX_K2: int

    def __init__(self, A: Tensor, B: Tensor, C: Tensor, D: Tensor, E: Tensor):
        r"""Store the structural components internally.

        Args:
            A: Dense symmetric matrix of shape `[K1, K1]` or smaller representing
                \(\mathbf{A}\).
            B: Dense rectangular matrix of shape `[K1, K - K1]` representing
                \(\mathbf{B}\).
            C: Vector of shape `[K - K1 - K2]` representing the diagonal of
                \(\mathbf{C}\).
            D: Dense rectangular matrix of shape `[K2, K - K1 - K2]` representing
                \(\mathbf{D}\).
            E: Dense symmetric matrix of shape `[K2, K2]` or smaller representing
                \(\mathbf{E}\).

        Note:
            For performance reasons, symmetry is not checked internally and must
            be ensured by the caller.

        Raises:
            ValueError: If the shapes of the arguments are invalid.
        """
        super().__init__()
        if A.ndim != 2 or B.ndim != 2 or C.ndim != 1 or D.ndim != 2 or E.ndim != 2:
            raise ValueError(
                "Invalid tensor dimensions. Expected 2, 2, 1, 2, 2."
                + f" Got {A.ndim}, {B.ndim}, {C.ndim}, {D.ndim}, {E.ndim}."
            )
        self._check_square(A, name="A")
        self._check_square(E, name="E")

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

        self.A: Tensor
        self.register_tensor(A, "A")

        self.B: Tensor
        self.register_tensor(B, "B")

        self.C: Tensor
        self.register_tensor(C, "C")

        self.D: Tensor
        self.register_tensor(D, "D")

        self.E: Tensor
        self.register_tensor(E, "E")

    @classmethod
    def from_dense(cls, sym_mat: Tensor) -> HierarchicalMatrixTemplate:
        """Construct from a PyTorch tensor.

        Args:
            sym_mat: A dense symmetric matrix which will be represented as
                `Hierarchical`.

        Returns:
            `HierarchicalMatrix` representing the passed matrix.
        """
        cls._check_square(sym_mat)
        dim = sym_mat.shape[0]
        K1, diag_dim, _ = cls._compute_block_dims(dim)

        A = sym_mat[:K1, :K1]
        B = 2 * sym_mat[:K1, K1:]
        C = sym_mat.diag()[K1 : K1 + diag_dim]
        D = 2 * sym_mat[K1 + diag_dim :, K1 : K1 + diag_dim]
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
            the result will be returned as a `HierarchicalMatrixTemplate`.
        """
        # parts of B that share columns with C, E
        B_C, B_E = self.B.split([self.diag_dim, self.K2], dim=1)

        if isinstance(other, Tensor):
            other_top, other_middle, other_bottom = other.split(
                [self.K1, self.diag_dim, self.K2]
            )

            top = self.A @ other_top + B_C @ other_middle + B_E @ other_bottom
            middle = einsum("i,ij->ij", self.C, other_middle)
            bottom = self.D @ other_middle + self.E @ other_bottom

            return cat([top, middle, bottom], dim=0)

        else:
            A_new = self.A @ other.A
            C_new = self.C * other.C
            E_new = self.E @ other.E
            D_new = einsum("ij,j->ij", self.D, other.C) + self.E @ other.D

            B_C_other, B_E_other = other.B.split([other.diag_dim, other.K2], dim=1)
            B_new = cat(
                [
                    self.A @ B_C_other
                    + einsum("ij,j->ij", B_C, other.C)
                    + B_E @ other.D,
                    self.A @ B_E_other + B_E @ other.E,
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
        """Multiply `mat` with the transpose of the structured matrix.

        Args:
            mat: A matrix which will be multiplied by the transpose of the represented
                hierarchical matrix.

        Returns:
            The result of the multiplication with the represented matrix's transpose.
        """
        mat_top, mat_middle, mat_bottom = mat.split([self.K1, self.diag_dim, self.K2])
        # parts of B that share columns with C, E
        B_C, B_E = self.B.split([self.diag_dim, self.K2], dim=1)

        top = self.A.T @ mat_top

        compute_dtype = lowest_precision(self.C.dtype, mat_middle.dtype)
        out_dtype = self.C.dtype
        middle = (
            B_C.T @ mat_top
            + einsum(
                "i,ij->ij", self.C.to(compute_dtype), mat_middle.to(compute_dtype)
            ).to(out_dtype)
            + self.D.T @ mat_bottom
        )
        bottom = B_E.T @ mat_top + self.E.T @ mat_bottom

        return cat([top, middle, bottom])

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def from_inner(self, X: Union[Tensor, None] = None) -> HierarchicalMatrixTemplate:
        """Represent the hierarchical matrix of `self.T @ X @ X^T @ self`.

        Args:
            X: Optional arbitrary 2d tensor. If `None`, `X = I` will be used.

        Returns:
            A `HierarchicalMatrix` representing hierarchical matrix of
            `self.T @ X @ X^T @ self`.
        """
        if X is None:
            A_new = self.A.T @ self.A
            B_new = 2 * self.A.T @ self.B

            # parts of B that share columns with C, E
            B_C, B_E = self.B.split([self.diag_dim, self.K2], dim=1)

            C_new = self.C**2 + (B_C**2).sum(0) + (self.D**2).sum(0)
            D_new = 2 * (B_E.T @ B_C + self.E.T @ self.D)
            E_new = self.E.T @ self.E + B_E.T @ B_E
        else:
            S_A, S_C, S_E = self.rmatmat(X).split([self.K1, self.diag_dim, self.K2])
            A_new = S_A @ S_A.T
            B_new = 2 * cat([S_A @ S_C.T, S_A @ S_E.T], dim=1)
            C_new = (S_C**2).sum(1)
            D_new = 2 * S_E @ S_C.T
            E_new = S_E @ S_E.T

        return self.__class__(A_new, B_new, C_new, D_new, E_new)

    def average_trace(self) -> Tensor:
        """Compute the average trace of the represented matrix.

        Returns:
            The average trace of the represented matrix.
        """
        dim_A, dim_C, dim_E = self.A.shape[0], self.C.shape[0], self.E.shape[0]
        dim = dim_A + dim_C + dim_E
        return (
            (self.A.diag() / dim).sum()
            + (self.C / dim).sum()
            + (self.E.diag() / dim).sum()
        )

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

    def frobenius_norm(self) -> Tensor:
        """Compute the Frobenius norm of the represented matrix.

        Returns:
            The Frobenius norm of the represented matrix.
        """
        return vector_norm(
            cat([t.flatten() for _, t in self.named_tensors() if t.numel() > 0])
        )

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
        """Compute the dimensions of `A, C, E`.

        Args:
            dim: Total dimension of the (square) matrix.

        Returns:
            A tuple of the form `(K1, diag_dim, K2)`.
        """
        if dim <= cls.MAX_K1:
            K1, diag_dim, K2 = dim, 0, 0
        elif dim <= cls.MAX_K1 + cls.MAX_K2:
            K1, diag_dim, K2 = cls.MAX_K1, 0, dim - cls.MAX_K1
        else:
            K1, diag_dim, K2 = cls.MAX_K1, dim - cls.MAX_K1 - cls.MAX_K2, cls.MAX_K2
        return K1, diag_dim, K2


class Hierarchical15_15Matrix(HierarchicalMatrixTemplate):
    """Hierarchical matrix with `K1=15` and `K2=15`.

    Note:
        See the template class `HierarchicalMatrixTemplate` for a mathematical
        description.
    """

    MAX_K1 = 15
    MAX_K2 = 15


class Hierarchical3_2Matrix(HierarchicalMatrixTemplate):
    """Hierarchical matrix with `K1=3` and `K2=2`.

    Note:
        See the template class `HierarchicalMatrixTemplate` for a mathematical
        description.
    """

    MAX_K1 = 3
    MAX_K2 = 2
