"""Utility functions for the tests."""

from torch import Tensor, allclose, isclose


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
