"""Profiling script for ``TriuToeplitzMatrix.from_dense``."""

from itertools import product
from timeit import timeit

import numpy as np
import torch
from torch import Tensor, allclose, cuda, device, manual_seed, rand

from singd.structures.triutoeplitz import TriuToeplitzMatrix


def old_from_dense(sym_mat: Tensor) -> TriuToeplitzMatrix:
    """Extract a ``TriuToeplitzMatrix`` from a symmetric dense matrix.

    Uses the approach from
    https://stackoverflow.com/questions/57347896/sum-all-diagonals-in-feature-maps-in-parallel-in-pytorch

    Args:
        sym_mat: A symmetric dense matrix.

    Returns:
        A ``TriuToeplitzMatrix``.
    """
    assert sym_mat.shape[0] == sym_mat.shape[1]
    dim = sym_mat.shape[0]

    x = torch.fliplr(sym_mat)
    digitized = np.sum(np.indices(x.shape), axis=0).ravel()
    digitized_tensor = torch.as_tensor(digitized).to(x.device)
    result = torch.bincount(digitized_tensor, x.view(-1))

    row = result[torch.arange(dim, device=result.device)].flip(0) + result[dim - 1 :]
    row.div_(torch.arange(dim, 0, step=-1, device=row.device))
    row[0] /= 2.0
    row = row.to(sym_mat.device).to(sym_mat.dtype)

    structured = TriuToeplitzMatrix(row)

    if sym_mat.is_cuda:
        cuda.synchronize()

    return structured


def current_from_dense(sym_mat: Tensor) -> TriuToeplitzMatrix:
    """Extract a ``TriuToeplitzMatrix`` from a symmetric dense matrix.

    Uses the current implementation.

    Args:
        sym_mat: A symmetric dense matrix.

    Returns:
        A ``TriuToeplitzMatrix``.
    """
    structured = TriuToeplitzMatrix.from_dense(sym_mat)

    if sym_mat.is_cuda:
        cuda.synchronize()

    return structured


if __name__ == "__main__":
    manual_seed(0)

    dims = [250, 1_000, 2_500]
    num_repeats = 5
    devices = (
        [device("cpu"), device("cuda")] if cuda.is_available() else [device("cpu")]
    )

    print("Benchmarking TrilToeplitzMatrix.from_dense")
    print(50 * "=")

    for dev, dim in product(devices, dims):
        sym_mat = rand(dim, dim).to(dev)
        sym_mat = (sym_mat + sym_mat.T) / 2.0

        # check correctness
        old = old_from_dense(sym_mat).to_dense()
        current = current_from_dense(sym_mat).to_dense()
        assert allclose(old, current, rtol=1e-5, atol=1e-7)

        # obtain timings
        best_old = float("inf")
        best_current = float("inf")

        for _ in range(num_repeats):
            run_time = timeit(lambda: old_from_dense(sym_mat), number=10)  # noqa: B023
            best_old = min(best_old, run_time)
            run_time = timeit(
                lambda: current_from_dense(sym_mat), number=10  # noqa: B023
            )
            best_current = min(best_current, run_time)

        print(f"Dim: {dim}, Device: {str(dev)}")
        print(f"\tOld: {best_old:.3e}")
        print(f"\tCurrent: {best_current:.3e}")
        print(f"\tRatio: {best_old / best_current:.2f}")
