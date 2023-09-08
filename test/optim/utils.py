"""Utility functions for testing the optimizer."""

from sparse_ngd.optim.optimizer import SNGD


def check_preconditioner_dtypes(optim: SNGD):
    """Verify that the optimizer's pre-conditioner has the correct data type.

    Args:
        optim: The optimizer to check.

    Raises:
        NotImplementedError: If the pre-conditioner structure is not supported.
    """
    for module in optim.modules:
        dtype_K, dtype_C = optim._get_param_group_entry(module, "preconditioner_dtype")
        dtype_K = module.weight.dtype if dtype_K is None else dtype_K
        dtype_C = module.weight.dtype if dtype_C is None else dtype_C

        structure_K, structure_C = optim._get_param_group_entry(module, "structures")

        if structure_K == "dense":
            real_K = optim.Ks[module]._mat.dtype
            real_m_K = optim.m_Ks[module]._mat.dtype
        elif structure_K == "diagonal":
            real_K = optim.Ks[module]._mat_diag.dtype
            real_m_K = optim.m_Ks[module]._mat_diag.dtype
        else:
            raise NotImplementedError(
                "Only dense and diagonal structures are supported."
            )

        assert real_K == dtype_K
        assert real_m_K == dtype_K

        if structure_C == "dense":
            real_C = optim.Cs[module]._mat.dtype
            real_m_C = optim.m_Cs[module]._mat.dtype
        elif structure_C == "diagonal":
            real_C = optim.Cs[module]._mat_diag.dtype
            real_m_C = optim.m_Cs[module]._mat_diag.dtype
        else:
            raise NotImplementedError(
                "Only dense and diagonal structures are supported."
            )

        assert real_C == dtype_C
        assert real_m_C == dtype_C
