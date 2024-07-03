"""Utility functions for testing the interface of structured matrices."""

from abc import ABC, abstractmethod
from os import makedirs, path
from test.utils import DEVICE_IDS, DEVICES, report_nonclose
from typing import Callable, List, Type, Union

import torch
from imageio import mimsave
from imageio.v2 import imread
from matplotlib import pyplot as plt
from pytest import mark
from torch import Tensor, device, manual_seed, rand, zeros
from torch.linalg import matrix_norm, vector_norm

from singd.structures.base import StructuredMatrix
from singd.structures.utils import is_half_precision, supported_eye

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_IDS = [str(dt).split(".")[-1] for dt in DTYPES]


def _test_matmul(
    sym_mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test `@` operation of any child of `StructuredMatrix`.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        mat2: Another dense matrix which will be (symmetrized then) converted into a
            structured matrix.
        structured_matrix_cls: The class of the structured matrix into which
            `sym_mat1` and a symmetrization of `mat2` will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into
            a dense matrix of the tested structure. Used to establish the ground truth.
    """
    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    sym_mat2 = symmetrize(mat2)
    sym_mat2_structured = structured_matrix_cls.from_dense(sym_mat2)

    tolerances = {
        "rtol": 5e-2 if is_half_precision(sym_mat1.dtype) else 1e-5,
        "atol": 1e-4 if is_half_precision(sym_mat1.dtype) else 1e-7,
    }

    # multiplication with a structured matrix
    truth = project(sym_mat1) @ project(sym_mat2)
    report_nonclose(
        truth,
        (sym_mat1_structured @ sym_mat2_structured).to_dense(),
        **tolerances,
    )

    # multiplication with a PyTorch tensor
    truth = project(sym_mat1) @ mat2
    report_nonclose(truth, sym_mat1_structured @ mat2, **tolerances)


def _test_add(
    sym_mat1: Tensor,
    sym_mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test `+` operation of any child of `StructuredMatrix`.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        sym_mat2: Another symmetric dense matrix which be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which
            `sym_mat1` and `sym_mat2` will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(sym_mat1) + project(sym_mat2)
    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    sym_mat2_structured = structured_matrix_cls.from_dense(sym_mat2)

    report_nonclose(
        truth,
        (sym_mat1_structured + sym_mat2_structured).to_dense(),
        rtol=1e-2 if is_half_precision(sym_mat1.dtype) else 1e-5,
        atol=1e-4 if is_half_precision(sym_mat1.dtype) else 1e-7,
    )


def _test_mul(
    sym_mat: Tensor,
    factor: float,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test `+` operation of any child of `StructuredMatrix`.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        factor: Scalar which will be multiplied onto the structured matrix.
        structured_matrix_cls: The class of the structured matrix into which `sym_mat`
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(factor * sym_mat)
    mat_structured = structured_matrix_cls.from_dense(sym_mat)
    report_nonclose(
        truth,
        (mat_structured * factor).to_dense(),
        rtol=1e-1 if is_half_precision(sym_mat.dtype) else 1e-5,
        atol=1e-4 if is_half_precision(sym_mat.dtype) else 1e-7,
    )


def _test_rmatmat(
    sym_mat1: Tensor,
    mat2: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
):
    """Test `rmatmat` operation of any child of `StructuredMatrix`.

    Args:
        sym_mat1: A symmetric dense matrix which will be converted into a structured
            matrix.
        mat2: A dense matrix onto which `sym_mat1`'s structured matrix transpose
            will be multiplied onto.
        structured_matrix_cls: The class of the structured matrix into which `mat`
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
    """
    truth = project(sym_mat1).T @ mat2

    sym_mat1_structured = structured_matrix_cls.from_dense(sym_mat1)
    report_nonclose(
        truth,
        sym_mat1_structured.rmatmat(mat2),
        rtol=1e-1 if is_half_precision(sym_mat1.dtype) else 1e-5,
        atol=1e-4 if is_half_precision(sym_mat1.dtype) else 1e-7,
    )


def _test_from_inner(
    sym_mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
    X: Union[Tensor, None],
):
    """Test `from_inner` method of any child of `StructuredMatrix`.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which `sym_mat`
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
        X: An optional matrix which will be passed to the `from_inner` method.
    """
    if X is None:
        truth = project(project(sym_mat).T @ project(sym_mat))
    else:
        MTX = project(sym_mat).T @ X
        truth = project(MTX @ MTX.T)

    sym_mat_structured = structured_matrix_cls.from_dense(sym_mat)
    report_nonclose(
        truth,
        sym_mat_structured.from_inner(X=X).to_dense(),
        rtol=5e-2 if is_half_precision(sym_mat.dtype) else 1e-5,
        atol=1e-4 if is_half_precision(sym_mat.dtype) else 1e-7,
    )


def _test_from_inner2(
    sym_mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
    project: Callable[[Tensor], Tensor],
    XXT: Tensor,
):
    """Test `from_inner2` method of any child of `StructuredMatrix`.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which `sym_mat`
            will be converted.
        project: A function which converts an arbitrary symmetric dense matrix into a
            dense matrix of the tested structure. Used to establish the ground truth.
        XXT: An symmetric PSD matrix that will be passed to `from_inner2`.
    """
    truth = project(project(sym_mat).T @ XXT @ project(sym_mat))
    sym_mat_structured = structured_matrix_cls.from_dense(sym_mat)
    report_nonclose(
        truth,
        sym_mat_structured.from_inner2(XXT).to_dense(),
        rtol=1e-2 if is_half_precision(sym_mat.dtype) else 1e-5,
        atol=1e-4 if is_half_precision(sym_mat.dtype) else 1e-7,
    )


def _test_zeros(
    structured_matrix_cls: Type[StructuredMatrix],
    dim: int,
    dtype: Union[torch.dtype, None] = None,
    device: Union[torch.device, None] = None,
):
    """Test initializing a structured matrix representing the zero matrix.

    Args:
        structured_matrix_cls: The class of the structured matrix to be tested.
        dim: Dimension of the (square) zero matrix.
        dtype: Optional data type of the matrix. If not specified, uses the default
            tensor type.
        device: Optional device of the matrix. If not specified, uses the default
            tensor type.
    """
    truth = zeros((dim, dim), dtype=dtype, device=device)
    structured_zero_matrix = structured_matrix_cls.zeros(
        dim, dtype=dtype, device=device
    )
    zero_matrix = structured_zero_matrix.to_dense()
    assert truth.dtype == zero_matrix.dtype
    assert truth.device == zero_matrix.device
    report_nonclose(truth, zero_matrix)


def _test_eye(
    structured_matrix_cls: Type[StructuredMatrix],
    dim: int,
    dtype: Union[torch.dtype, None] = None,
    device: Union[torch.device, None] = None,
):
    """Test initializing a structured matrix representing the identity matrix.

    Args:
        structured_matrix_cls: The class of the structured matrix to be tested.
        dim: Dimension of the (square) zero matrix.
        dtype: Optional data type of the matrix. If not specified, uses the default
            tensor type.
        device: Optional device of the matrix. If not specified, uses the default
            tensor type.
    """
    truth = supported_eye(dim, dtype=dtype, device=device)
    structured_identity_matrix = structured_matrix_cls.eye(
        dim, dtype=dtype, device=device
    )
    identity_matrix = structured_identity_matrix.to_dense()
    assert truth.dtype == identity_matrix.dtype
    assert truth.device == identity_matrix.device
    report_nonclose(truth, identity_matrix)


def _test_average_trace(
    sym_mat: Tensor,
    structured_matrix_cls: Type[StructuredMatrix],
):
    """Test `average_trace` operation of any child of `StructuredMatrix`.

    Args:
        sym_mat: A symmetric dense matrix which will be converted into a structured
            matrix.
        structured_matrix_cls: The class of the structured matrix into which `sym_mat`
            will be converted.
    """
    truth = sym_mat.diag().mean()
    mat_structured = structured_matrix_cls.from_dense(sym_mat)
    report_nonclose(
        truth,
        mat_structured.average_trace(),
        rtol=1e-2 if is_half_precision(sym_mat.dtype) else 1e-5,
        atol=1e-4 if is_half_precision(sym_mat.dtype) else 1e-7,
    )


def symmetrize(mat: Tensor) -> Tensor:
    """Symmetrize a matrix.

    Args:
        mat: A square matrix.

    Returns:
        The symmetrized matrix.
    """
    return (mat + mat.T) / 2.0


class _TestStructuredMatrix(ABC):
    """Abstract class for testing `StructuredMatrix` implementations.

    To test a new structured matrix type, create a new class and specify the class
    attributes, then implement the `project` method.

    `
    class TestDenseMatrix(_TestStructuredMatrix):
        STRUCTURED_MATRIX_CLS = DenseMatrix

        def project(self, mat: Tensor) -> Tensor:):
            ...
    `

    `pytest` will automatically pick up the tests defined for `TestDenseMatrix`
    via the base class.

    Attributes:
        STRUCTURED_MATRIX_CLS: The class of the structured matrix that is tested.
        DIMS: A list of dimensions of the matrices to be tested.
    """

    STRUCTURED_MATRIX_CLS: Type[StructuredMatrix]
    DIMS: List[int] = [1, 10]

    @abstractmethod
    def project(self, sym_mat: Tensor) -> Tensor:
        """Project a symmetric dense matrix onto a structured matrix.

        Args:
            sym_mat: A symmetric dense matrix.

        Returns:
            The same matrix.
        """
        raise NotImplementedError("Must be implemented by a child class.")

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_add(self, dev: device, dtype: torch.dtype):
        """Test matrix addition of two structured matrices.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            sym_mat1 = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            sym_mat2 = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            _test_add(sym_mat1, sym_mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_add_(self, dev: device, dtype: torch.dtype, alpha: float = -0.5):
        """Test in-place addition of a structured matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
            alpha: The value to scale the other matrix before adding. Default: `-0.5`.
        """
        tolerances = {
            "rtol": 2e-2 if is_half_precision(dtype) else 1e-5,
            "atol": 1e-4 if is_half_precision(dtype) else 1e-7,
        }

        for dim in self.DIMS:
            manual_seed(0)
            sym_mat1 = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            sym_mat2 = symmetrize(rand((dim, dim), device=dev, dtype=dtype))

            truth = self.project(sym_mat1.clone()).add_(
                self.project(sym_mat2.clone()), alpha=alpha
            )

            # Call in-place operation without assigning the return to a variable
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat1.clone())
            structured.add_(
                self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat2.clone()), alpha=alpha
            )
            report_nonclose(truth, structured.to_dense(), **tolerances)

            # Call in-place operation and assign the return to a variable
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat1.clone())
            updated_structured = structured.add_(
                self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat2.clone()), alpha=alpha
            )
            assert structured is updated_structured  # point to same object in memory
            report_nonclose(truth, updated_structured.to_dense(), **tolerances)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_matmul(self, dev: device, dtype: torch.dtype):
        """Test matrix multiplication of two structured matrices.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            sym_mat1 = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            mat2 = rand((dim, dim), device=dev, dtype=dtype)
            _test_matmul(sym_mat1, mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_mul(self, dev: device, dtype: torch.dtype):
        """Test multiplication of a structured matrix with a scalar.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            factor = 0.3
            _test_mul(sym_mat, factor, self.STRUCTURED_MATRIX_CLS, self.project)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_mul_(self, dev: device, dtype: torch.dtype, value: float = -1.23):
        """Test in-place multiplication of a structured matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
            value: The value to multiply with. Default: `-1.23`.
        """
        tolerances = {
            "rtol": 2e-2 if is_half_precision(dtype) else 1e-5,
            "atol": 1e-4 if is_half_precision(dtype) else 1e-7,
        }

        for dim in self.DIMS:
            manual_seed(0)
            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))

            truth = self.project(sym_mat.clone()).mul_(value)

            # Call in-place operation without assigning the return to a variable
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat.clone())
            structured.mul_(value)
            report_nonclose(truth, structured.to_dense(), **tolerances)

            # Call in-place operation and assign the return to a variable
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat.clone())
            updated_structured = structured.mul_(value)
            assert structured is updated_structured  # point to same object in memory
            report_nonclose(truth, updated_structured.to_dense(), **tolerances)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_rmatmat(self, dev: device, dtype: torch.dtype):
        """Test multiplication with the transpose of a structured matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            sym_mat1 = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            mat2 = rand((dim, 2 * dim), device=dev, dtype=dtype)
            _test_rmatmat(sym_mat1, mat2, self.STRUCTURED_MATRIX_CLS, self.project)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_from_inner(self, dev: device, dtype: torch.dtype):
        """Test structure extraction after self-inner product w/o intermediate term.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)

            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            X = None
            _test_from_inner(sym_mat, self.STRUCTURED_MATRIX_CLS, self.project, X)

            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            X = rand((dim, 2 * dim), device=dev, dtype=dtype)
            _test_from_inner(sym_mat, self.STRUCTURED_MATRIX_CLS, self.project, X)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_from_mat_inner(self, dev: device, dtype: torch.dtype):
        """Test structure extraction from `X @ X.T`.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            X = rand((dim, 2 * dim), device=dev, dtype=dtype)

            truth = self.project(X @ X.T)
            report_nonclose(
                truth,
                self.STRUCTURED_MATRIX_CLS.from_mat_inner(X).to_dense(),
                rtol=1e-2 if is_half_precision(X.dtype) else 1e-5,
                atol=1e-6 if is_half_precision(X.dtype) else 1e-7,
            )

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_from_inner2(self, dev: device, dtype: torch.dtype):
        """Test structure extraction after self-inner product w/ intermediate matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)

            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            X = rand((dim, 2 * dim), device=dev, dtype=dtype)
            XXT = X @ X.T
            _test_from_inner2(sym_mat, self.STRUCTURED_MATRIX_CLS, self.project, XXT)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_eye(self, dev: device, dtype: torch.dtype):
        """Test initializing a structured matrix representing the identity matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            _test_eye(self.STRUCTURED_MATRIX_CLS, dim, dtype=dtype, device=dev)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_zeros(self, dev: device, dtype: torch.dtype):
        """Test initializing a structured matrix representing the zero matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            _test_zeros(self.STRUCTURED_MATRIX_CLS, dim, dtype=dtype, device=dev)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_average_trace(self, dev: device, dtype: torch.dtype):
        """Test average trace of a structured dense matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)

            mat = rand((dim, dim), device=dev, dtype=dtype)
            _test_average_trace(mat, self.STRUCTURED_MATRIX_CLS)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_diag_add_(self, dev: device, dtype: torch.dtype, value: float = -1.23):
        """Test in-place addition onto the diagonal of a structured dense matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
            value: The value to add to the diagonal. Default: `-1.23`
        """
        tolerances = {
            "rtol": 1e-2 if is_half_precision(dtype) else 1e-5,
            "atol": 1e-4 if is_half_precision(dtype) else 1e-7,
        }

        for dim in self.DIMS:
            manual_seed(0)
            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))

            truth = self.project(sym_mat.clone())
            for d in range(dim):
                truth[d, d] += value

            # Call in-place operation without assigning the return to a variable
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat.clone())
            structured.diag_add_(value)
            report_nonclose(truth, structured.to_dense(), **tolerances)

            # Call in-place operation and assign the return to a variable
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat.clone())
            updated_structured = structured.diag_add_(value)
            assert structured is updated_structured  # point to same object in memory
            report_nonclose(truth, updated_structured.to_dense(), **tolerances)

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_infinity_vector_norm(self, dev: device, dtype: torch.dtype):
        """Test infinity vector norm of a structured matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            truth = vector_norm(self.project(sym_mat), ord=float("inf"))
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat)
            report_nonclose(truth, structured.infinity_vector_norm())

    @mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    @mark.parametrize("dev", DEVICES, ids=DEVICE_IDS)
    def test_frobenius_norm(self, dev: device, dtype: torch.dtype):
        """Test Frobenius norm of a structured matrix.

        Args:
            dev: The device on which to run the test.
            dtype: The data type of the matrices.
        """
        for dim in self.DIMS:
            manual_seed(0)
            sym_mat = symmetrize(rand((dim, dim), device=dev, dtype=dtype))
            truth = matrix_norm(self.project(sym_mat))
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(sym_mat)
            report_nonclose(truth, structured.frobenius_norm())

    @mark.expensive
    def test_visual(self):
        """Create pictures and animations of the structure.

        This serves to verify the edge cases where a matrix is too small to fit
        all the structural components.
        """
        manual_seed(0)
        dims = [1, 2, 4, 8, 16, 32, 64, 128]

        HEREDIR = path.dirname(path.abspath(__file__))
        structure_name = self.STRUCTURED_MATRIX_CLS.__name__
        FIGDIR = path.join(HEREDIR, "fig", structure_name)
        makedirs(FIGDIR, exist_ok=True)

        frames = []

        for d in dims:
            dense = symmetrize(rand(d, d))
            structured = self.STRUCTURED_MATRIX_CLS.from_dense(dense).to_dense()

            # share limits
            vmin = min(dense.min(), structured.min())
            vmax = max(dense.max(), structured.max())

            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.tight_layout()
            fig.suptitle(f"Dimension: {d}")
            ax1.set_title("Dense")
            ax1.imshow(dense, vmin=vmin, vmax=vmax)
            ax2.set_title(structure_name)
            ax2.imshow(structured, vmin=vmin, vmax=vmax)

            savepath = path.join(FIGDIR, f"dim_{d:05d}.png")
            fig.savefig(savepath)
            plt.close(fig)
            frames.append(savepath)

        # create gif
        images = [imread(frame) for frame in frames]
        mimsave(path.join(FIGDIR, "animated.gif"), images, duration=1_000, loop=0)
