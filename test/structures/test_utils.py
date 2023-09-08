"""Test utility functions of ``sparse_ngd.structures``."""

from pytest import raises
from torch import bfloat16, device, eye, float16, zeros


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
