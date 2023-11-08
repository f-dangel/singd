"""Test utility functions of the optimizer."""

from test.utils import report_nonclose
from typing import Any, Dict

from codetiming import Timer
from einops import reduce
from memory_profiler import memory_usage
from pytest import mark
from torch import Tensor, manual_seed, rand

from singd.optim.utils import _extract_averaged_patches, _extract_patches

CASES = [
    {
        "batch_size": 20,
        "in_channels": 10,
        "input_size": (28, 28),
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 2,  # must divide in_channels
        "seed": 0,
    }
]
CASE_IDS = [
    "_".join([f"{k}={v}".replace(" ", "") for k, v in case.items()]) for case in CASES
]


@mark.parametrize("case", CASES, ids=CASE_IDS)
def test_extract_average_patches(case: Dict[str, Any]):
    """Compare averaged patches with the averaged output of patches.

    Args:
        case: Dictionary of test case parameters.
    """
    manual_seed(case["seed"])
    x = rand(case["batch_size"], case["in_channels"], *case["input_size"])

    kernel_size = case["kernel_size"]
    stride = case["stride"]
    padding = case["padding"]
    dilation = case["dilation"]
    groups = case["groups"]

    patches = _extract_patches(x, kernel_size, stride, padding, dilation, groups)
    truth = reduce(patches, "b o1_o2 c_in_k1_k2 -> b c_in_k1_k2", "mean")

    averaged_patches = _extract_averaged_patches(
        x, kernel_size, stride, padding, dilation, groups
    )

    report_nonclose(averaged_patches, truth, rtol=1e-5, atol=1e-7)


PERFORMANCE_CASES = [
    {
        "batch_size": 128,
        "in_channels": 10,
        "input_size": (256, 256),
        "kernel_size": (5, 5),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "groups": 1,  # must divide in_channels
        "seed": 0,
    }
]
PERFORMANCE_CASE_IDS = [
    "_".join([f"{k}={v}".replace(" ", "") for k, v in case.items()])
    for case in PERFORMANCE_CASES
]


@mark.parametrize("case", PERFORMANCE_CASES, ids=PERFORMANCE_CASE_IDS)
def test_performance_extract_average_patches(case: Dict[str, Any]):
    """Compare performance of averaged patches vs averaged output of patches.

    Compares run time and memory consumption

    Args:
        case: Dictionary of test case parameters.
    """
    x_shape = (case["batch_size"], case["in_channels"], *case["input_size"])
    seed = case["seed"]

    kernel_size = case["kernel_size"]
    stride = case["stride"]
    padding = case["padding"]
    dilation = case["dilation"]
    groups = case["groups"]

    def inefficient_fn() -> Tensor:
        """Compute average patches inefficiently.

        Returns:
            Average patches.
        """
        manual_seed(seed)
        x = rand(*x_shape)
        patches = _extract_patches(x, kernel_size, stride, padding, dilation, groups)
        return reduce(patches, "b o1_o2 c_in_k1_k2 -> b c_in_k1_k2", "mean")

    def efficient_fn() -> Tensor:
        """Compute average patches efficiently.

        Returns:
            Average patches.
        """
        manual_seed(seed)
        x = rand(*x_shape)
        return _extract_averaged_patches(
            x, kernel_size, stride, padding, dilation, groups
        )

    # measure memory
    mem_inefficient = memory_usage(inefficient_fn, interval=1e-4, max_usage=True)
    mem_efficient = memory_usage(efficient_fn, interval=1e-4, max_usage=True)
    print(f"Memory used by inefficient function: {mem_inefficient:.1f} MiB.")
    print(f"Memory used by efficient function: {mem_efficient:.1f} MiB.")

    # measure run time
    with Timer(text="Inefficient function took {:.2e} s") as timer:
        inefficient_result = inefficient_fn()
    t_inefficient = timer.last
    with Timer(text="Efficient function took {:.2e} s") as timer:
        efficient_result = efficient_fn()
    t_efficient = timer.last

    # compare all performance specs
    report_nonclose(inefficient_result, efficient_result, rtol=1e-5, atol=1e-7)
    # NOTE This may be break for cases with small unfolded input, or if the built-in
    # version of `unfold` becomes more efficient.
    assert mem_efficient < mem_inefficient
    assert t_efficient < t_inefficient
