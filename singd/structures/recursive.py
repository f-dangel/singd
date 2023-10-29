"""Recursively defined structured matrices."""

from __future__ import annotations

from typing import Tuple, Type, Union

from torch import Tensor, block_diag

from singd.structures.base import StructuredMatrix


class RecursiveTopRightMatrixTemplate(StructuredMatrix):
    r"""Template to define recursively structured matrices with top right dense block.

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the attributes `MAX_DIMS`, `CLS_A`, and `CLS_C`. See the example
        below.

    This matrix is defined by

    \(\begin{pmatrix}
    \mathbf{A} & \mathbf{B} \\
    \mathbf{0} & \mathbf{C}
    \end{pmatrix}\)

    where

    - \(\mathbf{A}, \mathbf{C}\) are structured matrices (which can be recursive).
    - \(\mathbf{B}\) is a dense rectangular matrix.


    Attributes:
        MAX_DIMS: A tuple that contains an integer and a `float('inf')` which indicate
            the maximum dimensions of \(\mathbf{A}\) and \(\mathbf{C}\). For example,
            `(10, float('inf'))` means that only \(\mathbf{A}\) will be used for
            dimensions up to 10, and \(\mathbf{C}\) will be used in addition for larger
            dimensions.
        CLS_A: Structured matrix class used for the top left block \(\mathbf{A}\).
        CLS_C: Structured matrix class used for the the bottom right block
            \(\mathbf{B}\).

    Examples:
        >>> from torch import ones
        >>> from singd.structures.dense import DenseMatrix
        >>> from singd.structures.diagonal import DiagonalMatrix
        >>>
        >>> class Dense3DiagonalTopRightMatrix(RecursiveTopRightMatrixTemplate):
        ...     '''Structured matrix with 3 dense rows upper and lower diagonal part.'''
        ...     MAX_DIMS = (3, float('inf'))
        ...     CLS_A = DenseMatrix
        ...     CLS_C = DiagonalMatrix
        >>>
        >>> # A 5x5 matrix with 3 dense rows in the upper and lower diagonal part
        >>> A = DenseMatrix(ones(3, 3))
        >>> B = 2 * ones(3, 2)
        >>> C = DiagonalMatrix(3 * ones(2))
        >>> mat = Dense3DiagonalTopRightMatrix(A, B, C)
        >>> mat.to_dense()
        tensor([[1., 1., 1., 2., 2.],
                [1., 1., 1., 2., 2.],
                [1., 1., 1., 2., 2.],
                [0., 0., 0., 3., 0.],
                [0., 0., 0., 0., 3.]])
    """
    MAX_DIMS: Tuple[Union[int, float], Union[int, float]]
    CLS_A: Type[StructuredMatrix]
    CLS_C: Type[StructuredMatrix]

    def __init__(self, A: StructuredMatrix, B: Tensor, C: StructuredMatrix):
        r"""Store the matrix internally.

        Args:
            A: Structured matrix representing the top left block \(\mathbf{A}\).
            B: Rectangular tensor representing the top right block \(\mathbf{B}\).
            C: Structured matrix representing the bottom right block \(\mathbf{C}\).

        Note:
            For performance reasons, symmetry is not checked internally and must
            be ensured by the caller.

        Raises:
            ValueError: If the dimensions of the blocks do not match or the
                structured matrices are of wrong type.
        """
        if not isinstance(A, self.CLS_A) or not isinstance(C, self.CLS_C):
            raise ValueError(
                f"Matrices A and C must be of type {self.CLS_A} and "
                f"{self.CLS_C}, respectively. Got {type(A)} and {type(C)}."
            )

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
            sym_mat: A dense and symmetric matrix which will be approximated
                by a `RecursiveTopRightMatrixTemplate`.

        Returns:
            `RecursiveTopRightMatrixTemplate` approximating the passed matrix.
        """
        cls._check_square(sym_mat)

        boundary = _get_boundary(sym_mat.shape[0], cls.MAX_DIMS)
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
        mat = block_diag(A, C)

        dim_A = A.shape[0]
        mat[:dim_A, dim_A:] = self.B

        return mat


class RecursiveBottomLeftMatrixTemplate(StructuredMatrix):
    r"""Template to define recursively structured matrices with bottom left dense block.

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the attributes `MAX_DIMS`, `CLS_A`, and `CLS_C`. See the example
        below.

    This matrix is defined by

    \(\begin{pmatrix}
    \mathbf{A} & \mathbf{0} \\
    \mathbf{B} & \mathbf{C}
    \end{pmatrix}\)

    where

    - \(\mathbf{A}, \mathbf{C}\) are structured matrices (which can be recursive).
    - \(\mathbf{B}\) is a dense rectangular matrix.


    Attributes:
        MAX_DIMS: A tuple that contains an integer and a `float('inf')` which indicate
            the maximum dimensions of \(\mathbf{A}\) and \(\mathbf{C}\). For example,
            `(10, float('inf'))` means that only \(\mathbf{A}\) will be used for
            dimensions up to 10, and \(\mathbf{C}\) will be used in addition for larger
            dimensions.
        CLS_A: Structured matrix class used for the top left block \(\mathbf{A}\).
        CLS_C: Structured matrix class used for the the bottom right block
            \(\mathbf{C}\).

    Examples:
        >>> from torch import ones
        >>> from singd.structures.dense import DenseMatrix
        >>> from singd.structures.diagonal import DiagonalMatrix
        >>>
        >>> class Dense3DiagonalBottomLeftMatrix(RecursiveBottomLeftMatrixTemplate):
        ...     '''Structured matrix with 3 left columns and right diagonal part.'''
        ...     MAX_DIMS = (3, float('inf'))
        ...     CLS_A = DenseMatrix
        ...     CLS_C = DiagonalMatrix
        >>>
        >>> # A 5x5 matrix with 3 left columns and right diagonal part
        >>> A = DenseMatrix(ones(3, 3))
        >>> B = 2 * ones(2, 3)
        >>> C = DiagonalMatrix(3 * ones(2))
        >>> mat = Dense3DiagonalBottomLeftMatrix(A, B, C)
        >>> mat.to_dense()
        tensor([[1., 1., 1., 0., 0.],
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 0., 0.],
                [2., 2., 2., 3., 0.],
                [2., 2., 2., 0., 3.]])
    """
    MAX_DIMS: Tuple[Union[int, float], Union[int, float]]
    CLS_A: Type[StructuredMatrix]
    CLS_C: Type[StructuredMatrix]

    def __init__(self, A: StructuredMatrix, B: Tensor, C: StructuredMatrix):
        r"""Store the matrix internally.

        Args:
            A: Structured matrix representing the top left block \(\mathbf{A}\).
            B: Rectangular tensor representing the bottom left block \(\mathbf{B}\).
            C: Structured matrix representing the bottom right block \(\mathbf{C}\).

        Note:
            For performance reasons, symmetry is not checked internally and must
            be ensured by the caller.

        Raises:
            ValueError: If the dimensions of the blocks do not match or the structured
                matrices are of wrong type.
        """
        if not isinstance(A, self.CLS_A) or not isinstance(C, self.CLS_C):
            raise ValueError(
                f"Matrices A and C must be of type {self.CLS_A} and "
                f"{self.CLS_C}, respectively. Got {type(A)} and {type(C)}."
            )

        # TODO Add a `dim` property to make this cheaper
        dim_A, dim_C = A.to_dense().shape[0], C.to_dense().shape[0]
        if B.shape != (dim_C, dim_A):
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
    def from_dense(cls, sym_mat: Tensor) -> RecursiveBottomLeftMatrixTemplate:
        """Construct from a PyTorch tensor.

        Args:
            sym_mat: A dense and symmetric matrix which will be approximated
                by a `RecursiveBottomLeftMatrixTemplate`.

        Returns:
            `RecursiveTopRightMatrixTemplate` approximating the passed matrix.
        """
        cls._check_square(sym_mat)

        boundary = _get_boundary(sym_mat.shape[0], cls.MAX_DIMS)
        A = cls.CLS_A.from_dense(sym_mat[:boundary, :boundary])
        B = sym_mat[boundary:, :boundary] + sym_mat[:boundary, boundary:].T
        C = cls.CLS_C.from_dense(sym_mat[boundary:, boundary:])

        return cls(A, B, C)

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        A = self.A.to_dense()
        C = self.C.to_dense()
        mat = block_diag(A, C)

        dim_A = A.shape[0]
        mat[dim_A:, :dim_A] = self.B

        return mat


def _get_boundary(
    dim: int, max_dims: Tuple[Union[int, float], Union[int, float]]
) -> int:
    """Determine the boundary index between `A` and `C`.

    Args:
        dim: Total dimension of the recursive matrix.
        max_dims: A 2-tuple containing an integer and a `float('inf')` which indicate
            the maximum dimension of `A` and `C`.

    Returns:
        The boundary index between `A` and `C`.

    Raises:
        ValueError: If `max_dims`'s value is invalid.
    """
    if len(max_dims) != 2:
        raise ValueError(f"Invalid `MAX_DIMS` {max_dims}. Expected a 2-tuple.")

    dim_A, dim_C = max_dims

    if dim_A == float("inf") and isinstance(dim_C, int):
        boundary = max(0, dim - dim_C)
    elif dim_C == float("inf") and isinstance(dim_A, int):
        boundary = min(dim_A, dim)
    else:
        raise ValueError(
            f"Invalid `max_dims` {max_dims}. "
            "One dimension should be `float('inf')`, the other should be `int`."
        )

    return boundary
