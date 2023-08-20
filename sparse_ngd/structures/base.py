"""Implementations of matrices with structure that form a group under multiplication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union
from warnings import warn

import torch
from torch import Tensor, eye, zeros


class StructuredMatrix(ABC):
    """Base class for structured matrices closed under addition and multiplication.

    This base class defines the functions that need to be implemented to support
    a new structured matrix class with inverse-free KFAC.

    The minimum amount of work to add a new structured matrix class requires
    implementing the ``to_dense`` and ``from_dense`` methods. The other operations
    will then use a naive implementation which internally re-constructs unstructured
    dense matrices. By default, these operations will trigger a warning which can be
    used to identify functions that can be implemented more efficiently using structure.

    Attributes:
        WARN_NAIVE: Warn the user if a method falls back to a naive implementation
            of this base class. This indicates a method that should be implemented to
            save memory and run time by considering the represented structure.
            Default: ``True``.
    """

    WARN_NAIVE: bool = True

    def __matmul__(self, other: StructuredMatrix) -> StructuredMatrix:
        """Multiply with another matrix that has identical structure (@ operator).

        (https://peps.python.org/pep-0465/)

        Args:
            other: Another matrix with same structure which will be multiplied onto.

        Returns:
            A structured matrix resulting from the multiplication.
        """
        self._warn_naive_implementation("__matmul__")
        return self.from_dense(self.to_dense() @ other.to_dense())

    @classmethod
    @abstractmethod
    def from_dense(cls, mat: Tensor) -> StructuredMatrix:
        """Extract the represented structure from a dense matrix.

        This will discard elements that are not part of the structure, even if they
        are non-zero.

        Args:
            mat: A dense matrix which will be converted into a structured one.

        # noqa: DAR202

        Returns:
            Structured matrix.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError

    @abstractmethod
    def to_dense(self) -> Tensor:
        """Return a dense tensor representing the structured matrix.

        # noqa: DAR202

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

    def __add__(self, other: StructuredMatrix) -> StructuredMatrix:
        """Add another matrix of same structure.

        Args:
            other: Another structured matrix which will be added.

        Returns:
            A structured matrix resulting from the addition.
        """
        self._warn_naive_implementation("__add__")
        return self.from_dense(self.to_dense() + other.to_dense())

    def __sub__(self, other: StructuredMatrix) -> StructuredMatrix:
        """Subtract another matrix of same structure.

        Args:
            other: Another structured matrix which will be subtracted.

        Returns:
            A structured matrix resulting from the subtraction.
        """
        return self + (other * (-1.0))

    @abstractmethod
    def transpose(self) -> StructuredMatrix:
        """Create a structured matrix representing the transpose.

        Returns:
            A structured matrix representing the transpose.

        # noqa: DAR202

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError

    @classmethod
    def _warn_naive_implementation(cls, fn_name: str):
        """Warn the user that a naive implementation is called.

        This suggests that a child class does not implement a specialized version
        that is usually more efficient.

        You can turn off the warning by setting the ``WARN_NAIVE`` class attribute.

        Args:
            fn_name: Name of the function whose naive version is being called.
        """
        if cls.WARN_NAIVE:
            cls_name = cls.__name__
            warn(
                f"Calling naive implementation of {cls_name}.{fn_name}."
                + f"Consider implementing {cls_name}.{fn_name} using structure."
            )

    ###############################################################################
    #                        Special operations for IF-KFAC                       #
    ###############################################################################

    def from_inner(self, X: Union[Tensor, None] = None) -> StructuredMatrix:
        """Extract the represented structure from ``self.T @ X @ X^T @ self``.

        We can recycle terms by writing ``self.T @ X @ X^T @ self`` as ``S @ S^T``
        with ``S := self.T @ X``.

        Args:
            X: Optional arbitrary 2d tensor. If ``None``, ``X = I`` will be used.

        Returns:
            The structured matrix extracted from ``self.T @ X @ X^T @ self``.
        """
        self._warn_naive_implementation("from_inner")
        S_dense = self.to_dense().T if X is None else self.to_dense().T @ X
        return self.from_dense(S_dense @ S_dense.T)

    # NOTE This operation should be removed long-term as implementing IF-KFAC
    # with `from_inner` is more efficient. For now, it will exist as it makes
    # integrating this interface into existing implementations of sparse IF-KFAC
    # easier, as they have access to the input/gradient covariance matrices.
    def from_inner2(self, XXT: Tensor) -> StructuredMatrix:
        """Extract the represented structure from ``self.T @ XXT @ self``.

        Args:
            XXT: 2d square symmetric matrix.

        Returns:
            The structured matrix extracted from ``self.T @ XXT @ self``.
        """
        self._warn_naive_implementation("from_inner2")
        self_dense = self.to_dense()
        return self.from_dense(self_dense.T @ XXT @ self_dense)

    def trace(self) -> Tensor:
        """Compute the trace of the represented matrix.

        Returns:
            The trace of the represented matrix.
        """
        self._warn_naive_implementation("trace")
        return self.to_dense().trace()

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
        return cls.from_dense(eye(dim, dtype=dtype, device=device))
