"""Implementations of matrices with structure that form a group under multiplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Set, Tuple, Union
from warnings import warn

import torch
import torch.distributed as dist
from torch import Tensor, zeros
from torch.linalg import matrix_norm

from singd.structures.utils import diag_add_, supported_eye


class StructuredMatrix(ABC):
    """Base class for structured matrices closed under addition and multiplication.

    This base class defines the functions that need to be implemented to support
    a new structured matrix class with SINGD.

    The minimum amount of work to add a new structured matrix class requires
    implementing the following methods:

    - `to_dense`
    - `from_dense`

    All other operations will then use a naive implementation which internally
    re-constructs unstructured dense matrices. By default, these operations
    will trigger a warning which can be used to identify functions that can be
    implemented more efficiently using structure.

    Note:
        You need to register tensors that represent parts of the represented
        matrix using the `register_tensor` method. This is similar to the
        mechanism in PyTorch modules, which have a `register_parameter` method.
        It allows to support many operations out of the box.

    Attributes:
        WARN_NAIVE: Warn the user if a method falls back to a naive implementation
            of this base class. This indicates a method that should be implemented to
            save memory and run time by considering the represented structure.
            Default: `True`.
        WARN_NAIVE_EXCEPTIONS: Set of methods that should not trigger a warning even
            if `WARN_NAIVE` is `True`. This can be used to silence warnings for
            methods for which it is too complicated to leverage a specific structure
            and which should therefore call out to this class's implementation without
            performance warnings.
    """

    WARN_NAIVE: bool = True
    WARN_NAIVE_EXCEPTIONS: Set[str] = set()

    def __init__(self) -> None:
        """Initialize the structured matrix."""
        self._tensor_names: List[str] = []

    def register_tensor(self, tensor: Tensor, name: str) -> None:
        """Register a tensor that represents a part of the matrix structure.

        Args:
            tensor: A tensor that represents a part of the matrix structure.
            name: A name for the tensor. The tensor will be available under
                `self.name`.

        Raises:
            ValueError: If the name is already in use.
        """
        if hasattr(self, name):
            raise ValueError(f"Variable name {name!r} is already in use.")

        setattr(self, name, tensor)
        self._tensor_names.append(name)

    def named_tensors(self) -> Iterator[Tuple[str, Tensor]]:
        """Yield all tensors that represent the matrix and their names.

        Yields:
            A tuple of the tensor's name and the tensor itself.
        """
        for name in self._tensor_names:
            yield name, getattr(self, name)

    def __matmul__(
        self, other: Union[StructuredMatrix, Tensor]
    ) -> Union[StructuredMatrix, Tensor]:
        """Multiply onto a matrix ([@ operator](https://peps.python.org/pep-0465/)).

        Args:
            other: Another matrix which will be multiplied onto. Can be represented
                by a PyTorch tensor or a structured matrix.

        Returns:
            Result of the multiplication. If a PyTorch tensor was passed as argument,
            the result will be a PyTorch tensor. Otherwise, it will be a a structured
            matrix.
        """
        self._warn_naive_implementation("__matmul__")

        dense = self.to_dense()
        if isinstance(other, Tensor):
            return dense @ other

        other_dense = other.to_dense()
        return self.from_dense(dense @ other_dense)

    @classmethod
    @abstractmethod
    def from_dense(cls, sym_mat: Tensor) -> StructuredMatrix:
        """Extract the represented structure from a dense symmetric matrix.

        This will discard elements that are not part of the structure, even if they
        are non-zero.

        Warning:
            We do not verify whether `mat` is symmetric internally.

        Args:
            sym_mat: A symmetric dense matrix which will be converted into a structured
                one.

        Returns:
            Structured matrix.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError

    @abstractmethod
    def to_dense(self) -> Tensor:
        """Return a dense tensor representing the structured matrix.

        Returns:
            A dense PyTorch tensor representing the matrix.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError

    def __mul__(self, other: float) -> StructuredMatrix:
        """Multiply with a scalar.

        Args:
            other: A scalar that will be multiplied onto the structured matrix.

        Returns:
            The structured matrix, multiplied by the scalar.
        """
        self._warn_naive_implementation("__mul__")
        return self.from_dense(self.to_dense() * other)

    def mul_(self, value: float) -> StructuredMatrix:
        """In-place multiplication with a scalar.

        Args:
            value: A scalar that will be multiplied onto the structured matrix.

        Returns:
            Reference to the in-place updated matrix.
        """
        for _, tensor in self.named_tensors():
            tensor.mul_(value)

        return self

    def __add__(self, other: StructuredMatrix) -> StructuredMatrix:
        """Add another matrix of same structure.

        Args:
            other: Another structured matrix which will be added.

        Returns:
            A structured matrix resulting from the addition.
        """
        self._warn_naive_implementation("__add__")
        return self.from_dense(self.to_dense() + other.to_dense())

    def add_(self, other: StructuredMatrix, alpha: float = 1.0) -> StructuredMatrix:
        """In-place addition with another structured matrix.

        Args:
            other: Another structured matrix which will be added in-place.
            alpha: A scalar that will be multiplied onto `other` before adding it.
                Default: `1.0`.

        Returns:
            Reference to the in-place updated matrix.
        """
        for (_, tensor), (_, tensor_other) in zip(
            self.named_tensors(), other.named_tensors()
        ):
            tensor.add_(tensor_other, alpha=alpha)

        return self

    def rmatmat(self, mat: Tensor) -> Tensor:
        """Multiply the structured matrix's transpose onto a matrix (`self.T @ mat`).

        Args:
            mat: A dense matrix that will be multiplied onto.

        Returns:
            A dense PyTorch tensor resulting from the multiplication.
        """
        self._warn_naive_implementation("rmatmat")
        return self.to_dense().T @ mat

    @classmethod
    def _warn_naive_implementation(cls, fn_name: str):
        """Warn the user that a naive implementation is called.

        This suggests that a child class does not implement a specialized version
        that is usually more efficient.

        You can turn off the warning by setting the `WARN_NAIVE` class attribute.

        Args:
            fn_name: Name of the function whose naive version is being called.
        """
        if cls.WARN_NAIVE and fn_name not in cls.WARN_NAIVE_EXCEPTIONS:
            cls_name = cls.__name__
            warn(
                f"Calling naive implementation of {cls_name}.{fn_name}."
                + f"Consider implementing {cls_name}.{fn_name} using structure."
            )

    def all_reduce(
        self,
        op: dist.ReduceOp = dist.ReduceOp.AVG,
        group: Union[dist.ProcessGroup, None] = None,
        async_op: bool = False,
    ) -> Union[None, Tuple[torch._C.Future, ...]]:
        """Reduce the structured matrix across all devices.

        This method only has to be implemented to support distributed data
        parallel training.

        Args:
            op: The reduction operation to perform (default: `dist.ReduceOp.AVG`).
            group: The process group to work on. If `None`, the default process group
                will be used.
            async_op: If `True`, this function will return a
                `torch.distributed.Future` object.
                Otherwise, it will block until the reduction completes
                (default: `False`).

        Returns:
            If `async_op` is `True`, a (tuple of) `torch.distributed.Future`
            object(s), else `None`.
        """
        handles = []
        for _, tensor in self.named_tensors():
            tensor = tensor.contiguous()
            if async_op:
                handles.append(
                    dist.all_reduce(tensor, op=op, group=group, async_op=True)
                )
            else:
                dist.all_reduce(tensor, op=op, group=group, async_op=False)
        if async_op:
            return tuple(handles)

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def from_inner(self, X: Union[Tensor, None] = None) -> StructuredMatrix:
        """Extract the represented structure from `self.T @ X @ X^T @ self`.

        We can recycle terms by writing `self.T @ X @ X^T @ self` as `S @ S^T`
        with `S := self.T @ X`.

        Args:
            X: Optional arbitrary 2d tensor. If `None`, `X = I` will be used.

        Returns:
            The structured matrix extracted from `self.T @ X @ X^T @ self`.
        """
        self._warn_naive_implementation("from_inner")
        S_dense = self.to_dense().T if X is None else self.rmatmat(X)
        return self.from_mat_inner(S_dense)

    @classmethod
    def from_mat_inner(cls, X: Tensor) -> StructuredMatrix:
        """Extract the represented structure from `X @ X^T`.

        Args:
            X: Arbitrary 2d tensor.

        Returns:
            The structured matrix extracted from `X @ X^T`.
        """
        cls._warn_naive_implementation("from_mat_inner")
        return cls.from_dense(X @ X.T)

    # NOTE This operation should be removed long-term as implementing IF-KFAC
    # with `from_inner` is more efficient. For now, it will exist as it makes
    # integrating this interface into existing implementations of sparse IF-KFAC
    # easier, as they have access to the input/gradient covariance matrices.
    def from_inner2(self, XXT: Tensor) -> StructuredMatrix:
        """Extract the represented structure from `self.T @ XXT @ self`.

        Args:
            XXT: 2d square symmetric matrix.

        Returns:
            The structured matrix extracted from `self.T @ XXT @ self`.
        """
        self._warn_naive_implementation("from_inner2")
        dense = self.to_dense()
        return self.from_dense(dense.T @ XXT @ dense)

    def average_trace(self) -> Tensor:
        """Compute the average trace of the represented matrix.

        Returns:
            The average trace of the represented matrix.
        """
        self._warn_naive_implementation("trace")
        return self.to_dense().diag().mean()

    def diag_add_(self, value: float) -> StructuredMatrix:
        """In-place add a value to the diagonal of the represented matrix.

        Args:
            value: Value to add to the diagonal.

        Returns:
            A reference to the updated matrix.
        """
        self._warn_naive_implementation("diag_add_")
        dense = self.to_dense()
        diag_add_(dense, value)

        # NOTE `self` is immutable, so we have to update its state with the following
        # hack (otherwise, the call `a.diag_add_(b)` will not modify `a`). See
        # https://stackoverflow.com/a/37658673 and https://stackoverflow.com/q/1015592.
        new = self.from_dense(dense)
        self.__dict__.update(new.__dict__)
        return self

    def infinity_vector_norm(self) -> Tensor:
        """Compute the infinity vector norm.

        The infinity vector norm is the absolute value of the largest entry.
        Note that this is different from the infinity matrix norm, compare
        [here](https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)
        and
        [here](https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html).

        Returns:
            The matrix's infinity vector norm.
        """
        # NOTE `.max` can only be called on tensors with non-zero shape
        return max(t.abs().max() for _, t in self.named_tensors() if t.numel() > 0)

    def frobenius_norm(self) -> Tensor:
        """Compute the Frobenius norm of the represented matrix.

        Returns:
            The Frobenius norm of the represented matrix.
        """
        self._warn_naive_implementation("frobenius_norm")
        return matrix_norm(self.to_dense())

    ###############################################################################
    #                      Special initialization operations                      #
    ###############################################################################
    @classmethod
    def zeros(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> StructuredMatrix:
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
        cls._warn_naive_implementation("zero")
        return cls.from_dense(zeros((dim, dim), dtype=dtype, device=device))

    @classmethod
    def eye(
        cls,
        dim: int,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
    ) -> StructuredMatrix:
        """Create a structured matrix representing the identity matrix.

        Args:
            dim: Dimension of the (square) matrix.
            dtype: Optional data type of the matrix. If not specified, uses the default
                tensor type.
            device: Optional device of the matrix. If not specified, uses the default
                tensor type.

        Returns:
            A structured matrix representing the identity matrix.
        """
        cls._warn_naive_implementation("eye")
        return cls.from_dense(supported_eye(dim, dtype=dtype, device=device))

    @staticmethod
    def _check_square(t: Tensor, name: str = "tensor"):
        """Make sure the supplied tensor is a square matrix.

        Args:
            t: The tensor to be checked.
            name: Optional name of the tensor to be printed in the error message.
                Default: `"tensor"`.

        Raises:
            ValueError: If the tensor is not a square matrix.
        """
        if t.ndim != 2 or t.shape[0] != t.shape[1]:
            raise ValueError(f"{name} must be square matrix. Got shape {t.shape}.")
