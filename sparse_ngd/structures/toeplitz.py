"""Toeplitz matrix implemented in the ``StructuredMatrix`` interface."""

from __future__ import annotations

from torch import Tensor

from sparse_ngd.structures.base import StructuredMatrix


class ToeplitzMatrix(StructuredMatrix):
    """Toeplitz-structured matrix implemented in the ``StructuredMatrix`` interface.

    We follow the representation of such matrices using the SciPy terminology, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html
    """

    # TODO After the below basic functions are implemented, we can tackle the
    # specialized ones, then eventually remove this line
    WARN_NAIVE: bool = False  # Fall-back to naive base class implementations OK

    def __init__(self, c: Tensor, r: Tensor) -> None:
        """Store the Toeplitz matrix internally.

        Args:
            c: TODO @Wu Describe what c is.
            r: TODO @Wu Describe what r is.
        """
        raise NotImplementedError("TODO @Wu")

    @classmethod
    def from_dense(cls, mat: Tensor) -> ToeplitzMatrix:
        """Construct from a PyTorch tensor.

        Args:
            mat: A dense square matrix which will be represented as ``ToeplitzMatrix``.

        Returns:
            ``ToeplitzMatrix`` representing the passed matrix.
        """
        raise NotImplementedError("TODO @Wu")

    def to_dense(self) -> Tensor:
        """Convert into dense PyTorch tensor.

        Returns:
            The represented matrix as PyTorch tensor.
        """
        raise NotImplementedError("TODO @Wu")
