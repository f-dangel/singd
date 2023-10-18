"""Recursively defined structured matrices."""

from __future__ import annotations

from typing import Tuple, Type, Union

from torch import Tensor, zeros

from singd.structures.base import StructuredMatrix


class RecursiveTopRightMatrixTemplate(StructuredMatrix):
    r"""Template to define recursive structured matrices with top right dense block.

    This matrix is defined by

    \(\begin{pmatrix}
    \mathbf{A} & \mathbf{B} \\
    \mathbf{0} & \mathbf{C}
    \end{pmatrix}\)

    where

    - \(\mathbf{A}, \mathbf{C}\) are structured matrices (which can be recursive).
    - \(\mathbf{B}\) is a dense rectangular matrix.

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the attributes `MAX_DIM_A`, `CLS_A`, and `CLS_B`.

    Attributes:
        MAX_DIMS: A tuple that contains integers and `float('inf')` which indicate
            the maximum dimension of `A` and `C`. For example, `(10, float('inf'))`
            means that `A` will be used for dimensions up to 10, and `C` will be used
            in addition for larger dimensions.
        CLS_A: Structured matrix class used for the top left block.
        CLS_B: Structured matrix class used for the the bottom right block.
    """
    MAX_DIMS: Tuple[Union[int, float], Union[int, float]]
    CLS_A: Type[StructuredMatrix]
    CLS_C: Type[StructuredMatrix]

    def __init__(self, A: StructuredMatrix, B: Tensor, C: StructuredMatrix):
        """Store the matrix internally.

        Args:
            A: Top left block.
            B: Top right block.
            C: Bottom right block.

        Raises:
            ValueError: If the dimensions of the blocks do not match.
        """
        # TODO Add a `dim` property to make this cheaper
        dim_A, dim_C = A.to_dense().shape[0], C.to_dense().shape[0]
        if B.shape != (dim_A, dim_C):
            raise ValueError(f"Shape of `B` ({B.shape}) should be ({(dim_A, dim_C)}).")

        max_dim_A, max_dim_C = self.MAX_DIMS
        if dim_A > max_dim_A:
            raise ValueError(f"Dim. of A ({dim_A}) exceeds max dim. ({max_dim_A}).")
        if dim_C > max_dim_C:
            raise ValueError(f"Dim. of C ({dim_A}) exceeds max dim. ({max_dim_C}).")

        self.A = A
        self.B = B
        self.C = C

    @property
    def _tensors_to_sync(self) -> Tuple[Tensor, ...]:
        """Tensors that need to be synchronized across devices.

        This is used to support distributed data parallel training.

        Returns:
            A tuple of tensors that need to be synchronized across devices.
        """
        return self.A._tensors_to_sync + (self.B,) + self.C._tensors_to_sync

    @classmethod
    def from_dense(cls, sym_mat: Tensor) -> RecursiveTopRightMatrixTemplate:
        """Construct from a PyTorch tensor.

        Args:
            sym_mat: A dense and symmetric square matrix which will be approximated
            by a `RecursiveTopRightMatrixTemplate`.

        Returns:
            `RecursiveTopRightMatrixTemplate` approximating the passed matrix.

        Raises:
            ValueError: If the passed tensor is not square.
        """
        if sym_mat.dim() != 2 or sym_mat.shape[0] != sym_mat.shape[1]:
            raise ValueError(
                f"Expected square matrix. Got tensor shape {sym_mat.shape}."
            )

        assert len(cls.MAX_DIMS) == 2
        assert any(dim == float("inf") for dim in cls.MAX_DIMS)
        assert any(dim != float("inf") for dim in cls.MAX_DIMS)

        if cls.MAX_DIMS[0] == float("inf"):
            boundary = max(0, sym_mat.shape[0] - cls.MAX_DIMS[1])
        else:
            boundary = min(cls.MAX_DIMS[0], sym_mat.shape[0])

        A = cls.CLS_A.from_dense(sym_mat[:boundary, :boundary])
        B = sym_mat[:boundary, boundary:] + sym_mat[boundary:, :boundary].T
        C = cls.CLS_C.from_dense(sym_mat[boundary:, boundary:])

        return cls(A, B, C)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        A = self.A.to_dense()
        C = self.C.to_dense()

        dim_A, dim_C = A.shape[0], C.shape[0]
        dim = dim_A + dim_C
        mat = zeros(dim, dim, dtype=self.B.dtype, device=self.B.device)

        mat[:dim_A, :dim_A] = A
        mat[:dim_A, dim_A:] = self.B
        mat[dim_A:, dim_A:] = C

        return mat