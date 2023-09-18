"""Test utility functions of ``sparse_ngd.structures``."""

from pytest import raises
from torch import (
    Tensor,
    allclose,
    bfloat16,
    device,
    eye,
    float16,
    manual_seed,
    rand,
    zeros,
)

from sparse_ngd.structures.utils import all_traces, diag_add_


def test_cpu_float16_matmul_unsupported():
    """Test whether ``@`` between two ``float16`` tensors on CPU is unsupported."""
    cpu = device("cpu")
    mat1 = zeros((2, 2), dtype=float16, device=cpu)
    mat2 = zeros((2, 2), dtype=float16, device=cpu)

    with raises(RuntimeError):
        _ = mat1 @ mat2


def test_cpu_bfloat16_eye_unsupported():
    """Test whether ``eye`` is unsupported in ``bfloat16`` on CPU."""
    cpu = device("cpu")
    with raises(RuntimeError):
        eye(2, dtype=bfloat16, device=cpu)


def test_cpu_half_precision_trace_unsupported():
    """Test whether ``trace`` is unsupported in half precision on CPU."""
    cpu = device("cpu")

    for dtype in [float16, bfloat16]:
        mat = zeros((2, 2), device=cpu).to(dtype)
        with raises(RuntimeError):
            mat.trace()


def test_all_traces():
    """Test the computation of all traces of a matrix."""
    # fat matrix
    A = Tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ]
    )
    traces = Tensor([8.0, 13.0, 15.0, 18.0, 9.0, 3.0])
    assert allclose(all_traces(A), traces)

    # tall matrix
    B = Tensor(
        [
            [0.0, 4.0, 8.0],
            [1.0, 5.0, 9.0],
            [2.0, 6.0, 10.0],
            [3.0, 7.0, 11.0],
        ]
    )
    traces = Tensor([3.0, 9.0, 18.0, 15.0, 13.0, 8.0])
    assert allclose(all_traces(B), traces)


def test_diag_add_():
    """Test in-place addition to the diagonal of a matrix."""
    manual_seed(0)
    dim = 10
    mat = rand(dim, dim)
    value = -1.23

    truth = mat.clone()
    for d in range(dim):
        truth[d, d] += value

    # call in-place op and do not assign to a variable
    result = mat.clone()
    diag_add_(result, value)
    assert allclose(result, truth)

    # call in-place op and assign to a new variable
    result = mat.clone()
    new_result = diag_add_(result, value)
    assert result is new_result  # both point to the same object in memory
    assert allclose(result, truth)
