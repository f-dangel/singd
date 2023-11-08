"""Utility functions for the tests."""

from typing import Union

from torch import Tensor, allclose, cuda, device, isclose

from singd.optim.optimizer import SINGD

DEVICE_IDS = ["cpu", "cuda"] if cuda.is_available() else ["cpu"]
DEVICES = [device(name) for name in DEVICE_IDS]

REDUCTIONS = ["mean", "sum"]
REDUCTION_IDS = [f"reduction={reduction}" for reduction in REDUCTIONS]


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
        print(f"Min entries: {tensor1.min()}, {tensor2.min()}")
        print(f"Max entries: {tensor1.max()}, {tensor2.max()}")
        raise ValueError(f"{name} values don't match ({mismatch} / {tensor1.numel()}).")


def compare_optimizers(  # noqa: C901
    optim1: SINGD,
    optim2: SINGD,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    rtol_momentum: Union[float, None] = None,
    atol_momentum: Union[float, None] = None,
    rtol_hook: Union[float, None] = None,
    atol_hook: Union[float, None] = None,
    check_hook_quantities: bool = True,
    check_steps_and_grad_scales: bool = True,
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
        rtol_hook: Relative tolerance used for comparing the hook quantities.
            If not specified, uses the same value as ``rtol``.
        atol_hook: Absolute tolerance used for comparing the hook quantities.
            If not specified, uses the same value as ``atol``.
        check_hook_quantities: Whether to check the quantities stored and computed
            by hooks. Default: ``True``.
        check_steps_and_grad_scales: Whether to compare the ``steps`` and
            ``_grad scales`` attributes. Default: ``True``.
    """
    if check_steps_and_grad_scales:
        assert optim1.steps == optim2.steps
        assert optim1._grad_scales == optim2._grad_scales

    # compare K, C, m_K, m_C
    assert len(optim1.module_names) == len(optim2.module_names)
    assert set(optim1.module_names.values()) == set(optim2.module_names.values())
    for name in optim1.module_names.values():
        K1 = optim1.Ks[name].to_dense()
        K2 = optim2.Ks[name].to_dense()
        report_nonclose(K1, K2, atol=atol, rtol=rtol, name="K")

        m_K1 = optim1.m_Ks[name].to_dense()
        m_K2 = optim2.m_Ks[name].to_dense()
        report_nonclose(m_K1, m_K2, atol=atol, rtol=rtol, name="m_K")

        C1 = optim1.Cs[name].to_dense()
        C2 = optim2.Cs[name].to_dense()
        report_nonclose(C1, C2, atol=atol, rtol=rtol, name="C")

        m_C1 = optim1.m_Cs[name].to_dense()
        m_C2 = optim2.m_Cs[name].to_dense()
        report_nonclose(m_C1, m_C2, atol=atol, rtol=rtol, name="m_C")

    # compare hook quantities
    atol_hook = atol_hook if atol_hook is not None else atol
    rtol_hook = rtol_hook if rtol_hook is not None else rtol

    if check_hook_quantities:
        assert set(optim1.H_Ks.keys()) == set(optim2.H_Ks.keys())
        assert set(optim1.H_Cs.keys()) == set(optim2.H_Cs.keys())

        for name in optim1.H_Ks:
            H_K1 = optim1.H_Ks[name].value.to_dense()
            H_K2 = optim2.H_Ks[name].value.to_dense()
            report_nonclose(H_K1, H_K2, atol=atol_hook, rtol=rtol_hook, name="H_K")

        for name in optim1.H_Cs:
            H_C1 = optim1.H_Cs[name].value.to_dense()
            H_C2 = optim2.H_Cs[name].value.to_dense()
            report_nonclose(H_C1, H_C2, atol=atol_hook, rtol=rtol_hook, name="H_C")

    # compare momentum buffers
    atol_momentum = atol_momentum if atol_momentum is not None else atol
    rtol_momentum = rtol_momentum if rtol_momentum is not None else rtol

    if (
        optim1.param_groups[0]["momentum"] != 0
        or optim2.param_groups[0]["momentum"] != 0
    ):
        for module1, module2 in zip(optim1.module_names, optim2.module_names):
            assert len(list(module1.parameters())) == len(list(module2.parameters()))
            for p1, p2 in zip(module1.parameters(), module2.parameters()):
                mom1 = optim1.state[p1]["momentum_buffer"]
                mom2 = optim2.state[p2]["momentum_buffer"]
                report_nonclose(
                    mom1,
                    mom2,
                    atol=atol_momentum,
                    rtol=rtol_momentum,
                    name="momentum",
                )

    # compare parameter values
    for module1, module2 in zip(optim1.module_names, optim2.module_names):
        assert len(list(module1.parameters())) == len(list(module2.parameters()))
        for p1, p2 in zip(module1.parameters(), module2.parameters()):
            report_nonclose(p1, p2, atol=atol, rtol=rtol, name="parameters")
