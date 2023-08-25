"""Utility functions for the tests."""

from typing import Union

from torch import Tensor, allclose, isclose

from sparse_ngd.optim.optimizer import SNGD


def report_nonclose(
    tensor1: Tensor,
    tensor2: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    name: str = "array",
):
    """Compare two tensors, raise exception if nonclose values and print them.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        rtol: Relative tolerance (see ``torch.allclose``). Default: ``1e-5``.
        atol: Absolute tolerance (see ``torch.allclose``). Default: ``1e-8``.
        equal_nan: Whether comparing two NaNs should be considered as ``True``
            (see ``torch.allclose``). Default: ``False``.
        name: Optional name what the compared tensors mean. Default: ``'array'``.

    Raises:
        ValueError: If the two tensors don't match in shape or have nonclose values.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"{name} shapes don't match.")

    if allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan):
        print(f"{name} values match.")
    else:
        mismatch = 0
        for a1, a2 in zip(tensor1.flatten(), tensor2.flatten()):
            if not isclose(a1, a2, atol=atol, rtol=rtol, equal_nan=equal_nan):
                mismatch += 1
                print(f"{a1} ≠ {a2}")
        raise ValueError(f"{name} values don't match ({mismatch} / {tensor1.numel()}).")


def compare_optimizers(
    optim1: SNGD,
    optim2: SNGD,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    rtol_momentum: Union[float, None] = None,
    atol_momentum: Union[float, None] = None,
):
    """Compare the states of two structured inverse-free KFAC optimizers.

    Looks at ``K, m_K, C, m_C``, as well as the momentum buffers and parameters.

    Args:
        optim1: First optimizer.
        optim2: Second optimizer.
        rtol: Relative tolerance (see ``torch.allclose``) for comparison.
            Default: ``1e-5``.
        atol: Absolute tolerance (see ``torch.allclose``). Default: ``1e-8``.
        rtol_momentum: Relative tolerance used for comparing the momentum buffers.
            If not specified, uses the same value as ``rtol``.
        atol_momentum: Absolute tolerance used for comparing the momentum buffers.
            If not specified, uses the same value as ``atol``.
    """
    # compare K, C, m_K, m_C
    assert len(optim1.modules) == len(optim2.modules)
    for m, m_scale in zip(optim1.modules, optim2.modules):
        K = optim1.Ks[m].to_dense()
        K_scale = optim2.Ks[m_scale].to_dense()
        report_nonclose(K, K_scale, atol=atol, rtol=rtol, name="K")

        m_K = optim1.m_Ks[m].to_dense()
        m_K_scale = optim2.m_Ks[m_scale].to_dense()
        report_nonclose(m_K, m_K_scale, atol=atol, rtol=rtol, name="m_K")

        C = optim1.Cs[m].to_dense()
        C_scale = optim2.Cs[m_scale].to_dense()
        report_nonclose(C, C_scale, atol=atol, rtol=rtol, name="C")

        m_C = optim1.m_Cs[m].to_dense()
        m_C_scale = optim2.m_Cs[m_scale].to_dense()
        report_nonclose(m_C, m_C_scale, atol=atol, rtol=rtol, name="m_C")

    # compare momentum buffers
    atol_momentum = atol_momentum if atol_momentum is not None else atol
    rtol_momentum = rtol_momentum if rtol_momentum is not None else rtol

    if optim1.momentum != 0 or optim2.momentum != 0:
        for module1, module2 in zip(optim1.modules, optim2.modules):
            for p, p_scale in zip(module1.parameters(), module2.parameters()):
                mom = optim1.state[p]["momentum_buffer"]
                mom_scale = optim2.state[p_scale]["momentum_buffer"]
                report_nonclose(
                    mom,
                    mom_scale,
                    atol=atol_momentum,
                    rtol=rtol_momentum,
                    name="momentum",
                )

    # compare parameter values
    for module1, module2 in zip(optim1.modules, optim2.modules):
        for p, p_scale in zip(module1.parameters(), module2.parameters()):
            report_nonclose(p, p_scale, atol=atol, rtol=rtol, name="parameters")
