"""Recursively defined structured matrices."""

from __future__ import annotations

from typing import Iterator, List, Tuple, Type, Union

from torch import Tensor, block_diag

from singd.structures.base import StructuredMatrix


class RecursiveStructuredMatrix(StructuredMatrix):
    """Base class for recursively defined structured matrices.

    Note:
        To register another structured matrix, use the `register_substructure` method.
        This is similar to PyTorch modules which have a `register_module` method.
    """

    def __init__(self) -> None:
        """Initialize the recursively structured matrix."""
        super().__init__()
        self._substructure_names: List[str] = []

    def register_substructure(self, substructure: StructuredMatrix, name: str) -> None:
        """Register a substructure that represents a part of the matrix.

        Args:
            substructure: A structured matrix
            name: A name for the structured matrix. The matrix will be available under
                `self.name`.
        """
        if hasattr(self, name):
            raise ValueError(f"Variable name {name!r} is already in use.")

        setattr(self, name, substructure)
        self._substructure_names.append(name)

    def named_tensors(self) -> Iterator[Tuple[str, Tensor]]:
        """Yield all tensors that represent the matrix and their names.

        Yields:
            A tuple of the tensor's name and the tensor itself.
        """
        for name in self._tensor_names:
            yield name, getattr(self, name)
        for name in self._substructure_names:
            substructure = getattr(self, name)
            for sub_name, tensor in substructure.named_tensors():
                yield f"{name}.{sub_name}", tensor


class RecursiveTopRightMatrixTemplate(RecursiveStructuredMatrix):
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
        then specify the attributes `MAX_DIMS`, `CLS_A`, and `CLS_C`.

    Attributes:
        MAX_DIMS: A tuple that contains integers and `float('inf')` which indicate
            the maximum dimension of `A` and `C`. For example, `(10, float('inf'))`
            means that `A` will be used for dimensions up to 10, and `C` will be used
            in addition for larger dimensions.
        CLS_A: Structured matrix class used for the top left block.
        CLS_C: Structured matrix class used for the the bottom right block.
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
        super().__init__()
        # TODO Add a `dim` property to make this cheaper
        dim_A, dim_C = A.to_dense().shape[0], C.to_dense().shape[0]
        if B.shape != (dim_A, dim_C):
            raise ValueError(f"Shape of `B` ({B.shape}) should be ({(dim_A, dim_C)}).")

        max_dim_A, max_dim_C = self.MAX_DIMS
        if dim_A > max_dim_A:
            raise ValueError(f"Dim. of A ({dim_A}) exceeds max dim. ({max_dim_A}).")
        if dim_C > max_dim_C:
            raise ValueError(f"Dim. of C ({dim_A}) exceeds max dim. ({max_dim_C}).")

        self.A: StructuredMatrix
        self.register_substructure(A, "A")

        self.B: Tensor
        self.register_tensor(B, "B")

        self.C: StructuredMatrix
        self.register_substructure(C, "C")

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


class RecursiveBottomLeftMatrixTemplate(RecursiveStructuredMatrix):
    r"""Template to define recursive structured matrices with bottom left dense block.

    This matrix is defined by

    \(\begin{pmatrix}
    \mathbf{A} & \mathbf{0} \\
    \mathbf{B} & \mathbf{C}
    \end{pmatrix}\)

    where

    - \(\mathbf{A}, \mathbf{C}\) are structured matrices (which can be recursive).
    - \(\mathbf{B}\) is a dense rectangular matrix.

    Note:
        This is a template class. To define an actual class, inherit from this class,
        then specify the attributes `MAX_DIMS`, `CLS_A`, and `CLS_C`.

    Attributes:
        MAX_DIMS: A tuple that contains integers and `float('inf')` which indicate
            the maximum dimension of `A` and `C`. For example, `(10, float('inf'))`
            means that `A` will be used for dimensions up to 10, and `C` will be used
            in addition for larger dimensions.
        CLS_A: Structured matrix class used for the top left block.
        CLS_C: Structured matrix class used for the the bottom right block.
    """
    MAX_DIMS: Tuple[Union[int, float], Union[int, float]]
    CLS_A: Type[StructuredMatrix]
    CLS_C: Type[StructuredMatrix]

    def __init__(self, A: StructuredMatrix, B: Tensor, C: StructuredMatrix):
        """Store the matrix internally.

        Args:
            A: Top left block.
            B: Bottom left block.
            C: Bottom right block.

        Raises:
            ValueError: If the dimensions of the blocks do not match.
        """
        super().__init__()
        # TODO Add a `dim` property to make this cheaper
        dim_A, dim_C = A.to_dense().shape[0], C.to_dense().shape[0]
        if B.shape != (dim_C, dim_A):
            raise ValueError(f"Shape of `B` ({B.shape}) should be ({(dim_A, dim_C)}).")

        max_dim_A, max_dim_C = self.MAX_DIMS
        if dim_A > max_dim_A:
            raise ValueError(f"Dim. of A ({dim_A}) exceeds max dim. ({max_dim_A}).")
        if dim_C > max_dim_C:
            raise ValueError(f"Dim. of C ({dim_A}) exceeds max dim. ({max_dim_C}).")

        self.A: StructuredMatrix
        self.register_substructure(A, "A")

        self.B: Tensor
        self.register_tensor(B, "B")

        self.C: StructuredMatrix
        self.register_substructure(C, "C")

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
