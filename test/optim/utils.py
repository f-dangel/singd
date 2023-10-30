"""Utility functions for testing the optimizer."""

from typing import Tuple

from torch import Tensor, cat, dtype, stack
from torch.autograd import grad
from torch.nn import Module

from singd.optim.optimizer import SINGD
from singd.structures.base import StructuredMatrix


def check_preconditioner_structures(optim: SINGD, structures: Tuple[str, str]):
    """Verify that the optimizer's pre-conditioner has the correct structure.

    Args:
        optim: The optimizer to check.
        structures: The structure-tuple used to initialize the optimizer.
    """
    K_cls = SINGD.SUPPORTED_STRUCTURES[structures[0]]
    C_cls = SINGD.SUPPORTED_STRUCTURES[structures[1]]

    for name in optim.module_names.values():
        assert isinstance(optim.Ks[name], K_cls)
        assert isinstance(optim.Cs[name], C_cls)
        assert isinstance(optim.m_Ks[name], K_cls)
        assert isinstance(optim.m_Cs[name], C_cls)


def check_preconditioner_dtypes(optim: SINGD):
    """Verify that the optimizer's pre-conditioner has the correct data type.

    Args:
        optim: The optimizer to check.
    """
    for module, name in optim.module_names.items():
        dtype_K, dtype_C = optim._get_param_group_entry(module, "preconditioner_dtype")
        dtype_K = dtype_K if isinstance(dtype_K, dtype) else module.weight.dtype
        dtype_C = dtype_C if isinstance(dtype_C, dtype) else module.weight.dtype

        verify_dtype(optim.Ks[name], dtype_K)
        verify_dtype(optim.m_Ks[name], dtype_K)
        verify_dtype(optim.Cs[name], dtype_C)
        verify_dtype(optim.m_Cs[name], dtype_C)


def verify_dtype(mat: StructuredMatrix, dtype: dtype):
    """Check whether a structured matrix's  tensors are of the specified type.

    Args:
        mat: The structured matrix to check.
        dtype: The dtype to check for.

    Raises:
        RuntimeError: If the structure is not of the specified type.
    """
    dtypes = [tensor.dtype for _, tensor in mat.named_tensors()]
    if any(d != dtype for d in dtypes):
        raise RuntimeError(f"Expected dtype {dtype}, got {dtypes}.")


def jacobians_naive(model: Module, data: Tensor, setting: str) -> Tuple[Tensor, Tensor]:
    """Compute the Jacobians of a model's output w.r.t. its parameters.

    Args:
        model: The model.
        data: The input data.
        setting: The setting to use for the forward pass of the model
            (if appropriate). Possible values are `"expand"` and `"reduce"`.

    Returns:
        A tuple of the Jacobians of the model's output w.r.t. its parameters and
        the model's output, with shapes `(n_loss_terms * out_dim, num_params)`
        and `(n_loss_terms, ..., out_dim)` respectively.
    """
    try:
        f: Tensor = model(data, setting)
    except TypeError:
        f: Tensor = model(data)
    # f: (n_loss_terms, out_dim)
    last_f_dim = f.numel() - 1
    jacs = []
    for i, f_i in enumerate(f.flatten()):
        jac = grad(f_i, model.parameters(), retain_graph=i != last_f_dim)
        jacs.append(cat([j.flatten() for j in jac]))
    # jacs: (n_loss_terms * out_dim, num_params)
    jacs = stack(jacs).flatten(end_dim=-2)
    return jacs.detach(), f.detach()


class Transpose(Module):
    """A module that transposes the input tensor."""

    def __init__(self, dim0: int, dim1: int):
        """Initialize the module.

        Args:
            dim0: The first dimension to transpose.
            dim1: The second dimension to transpose.
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        """Transpose the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The transposed tensor.
        """
        return x.transpose(self.dim0, self.dim1)
