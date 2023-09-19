"""Utility functions for testing the optimizer."""

from typing import Tuple

import torch

from sparse_ngd.optim.optimizer import SNGD
from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.blockdiagonal import BlockDiagonalMatrixTemplate
from sparse_ngd.structures.dense import DenseMatrix
from sparse_ngd.structures.diagonal import DiagonalMatrix
from sparse_ngd.structures.hierarchical import HierarchicalMatrixTemplate
from sparse_ngd.structures.triltoeplitz import TrilToeplitzMatrix
from sparse_ngd.structures.triutoeplitz import TriuToeplitzMatrix


def check_preconditioner_structures(optim: SNGD, structures: Tuple[str, str]):
    """Verify that the optimizer's pre-conditioner has the correct structure.

    Args:
        optim: The optimizer to check.
        structures: The structure-tuple used to initialize the optimizer.
    """
    K_cls = SNGD.SUPPORTED_STRUCTURES[structures[0]]
    C_cls = SNGD.SUPPORTED_STRUCTURES[structures[1]]

    for module in optim.modules:
        assert isinstance(optim.Ks[module], K_cls)
        assert isinstance(optim.Cs[module], C_cls)
        assert isinstance(optim.m_Ks[module], K_cls)
        assert isinstance(optim.m_Cs[module], C_cls)


def check_preconditioner_dtypes(optim: SNGD):
    """Verify that the optimizer's pre-conditioner has the correct data type.

    Args:
        optim: The optimizer to check.
    """
    for module in optim.modules:
        dtype_K, dtype_C = optim._get_param_group_entry(module, "preconditioner_dtype")
        dtype_K = dtype_K if isinstance(dtype_K, torch.dtype) else module.weight.dtype
        dtype_C = dtype_C if isinstance(dtype_C, torch.dtype) else module.weight.dtype

        verify_dtype(optim.Ks[module], dtype_K)
        verify_dtype(optim.m_Ks[module], dtype_K)
        verify_dtype(optim.Cs[module], dtype_C)
        verify_dtype(optim.m_Cs[module], dtype_C)


def verify_dtype(mat: StructuredMatrix, dtype: torch.dtype):
    """Check whether a structured matrix's  tensors are of the specified type.

    Args:
        mat: The structured matrix to check.
        dtype: The dtype to check for.

    Raises:
        NotImplementedError: If the structure is not supported.
        RuntimeError: If the structure is not of the specified type.
    """
    if isinstance(mat, DenseMatrix):
        tensors_to_check = (mat._mat,)
    elif isinstance(mat, DiagonalMatrix):
        tensors_to_check = (mat._mat_diag,)
    elif issubclass(mat.__class__, BlockDiagonalMatrixTemplate):
        tensors_to_check = (mat._blocks, mat._last)
    elif issubclass(mat.__class__, HierarchicalMatrixTemplate):
        tensors_to_check = (mat.A, mat.B, mat.C, mat.D, mat.E)
    elif isinstance(mat, TriuToeplitzMatrix):
        tensors_to_check = (mat._mat_row,)
    elif isinstance(mat, TrilToeplitzMatrix):
        tensors_to_check = (mat._mat_column,)
    else:
        raise NotImplementedError(f"{mat.__class__.__name__} is not yet supported.")

    dtypes = [tensor.dtype for tensor in tensors_to_check]
    if any(d != dtype for d in dtypes):
        raise RuntimeError(f"Expected dtype {dtype}, got {dtypes}.")
