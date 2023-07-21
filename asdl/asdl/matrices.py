import os
import copy

import torch
import torch.nn as nn
import torch.distributed as dist

from .symmatrix import SymMatrix

HESSIAN = 'hessian'  # Hessian
FISHER_EXACT = 'fisher_exact'  # exact Fisher
FISHER_MC = 'fisher_mc'  # Fisher estimation by Monte-Carlo sampling
FISHER_EMP = 'fisher_emp'  # no-centered covariance a.k.a. empirical Fisher

SHAPE_FULL = 'full'  # full
SHAPE_LAYER_WISE = 'layer_wise'  # layer-wise block-diagonal
SHAPE_KRON = 'kron'  # Kronecker-factored
SHAPE_SWIFT_KRON = 'swift_kron'  # swift Kronecker-factored
SHAPE_KFE = 'kfe'  # Kronecker-factored eigenbasis
SHAPE_UNIT_WISE = 'unit_wise'  # unit-wise block-diagonal
SHAPE_DIAG = 'diag'  # diagonal

__all__ = [
    'MatrixManager',
    'FISHER_EXACT',
    'FISHER_MC',
    'FISHER_EMP',
    'HESSIAN',
    'SHAPE_FULL',
    'SHAPE_LAYER_WISE',
    'SHAPE_KRON',
    'SHAPE_SWIFT_KRON',
    'SHAPE_KFE',
    'SHAPE_UNIT_WISE',
    'SHAPE_DIAG',
]

_supported_types = [HESSIAN, FISHER_EXACT, FISHER_MC, FISHER_EMP]
_supported_shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG]

_normalizations = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def _requires_matrix(module: torch.nn.Module):
    if not hasattr(module, 'weight'):
        return False
    if module.weight.requires_grad:
        return True
    return hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad


class MatrixManager:
    def __init__(self, model, matrix_types, scale=1., smoothing_weight=None):
        self._model: nn.Module = model
        self._device = next(model.parameters()).device
        if isinstance(matrix_types, str):
            matrix_types = [matrix_types]
        for mat_type in matrix_types:
            if mat_type not in _supported_types:
                raise ValueError(f'Invalid matrix_type: {mat_type}. {_supported_types} are supported.')
        # remove duplicates
        self._matrix_types = set(matrix_types)
        # for updating stats
        self._scale = scale
        self._smoothing_weight = smoothing_weight
        self._stats_names = set()

    @staticmethod
    def _get_save_field(matrix_type, stats_name=None):
        if stats_name is None:
            return matrix_type
        return f'{stats_name}_{matrix_type}'

    def _clear_stats(self, stats_name):
        if stats_name in self._stats_names:
            self._stats_names.remove(stats_name)

    def _check_stats_name(self, stats_name):
        if stats_name is None:
            return
        if stats_name not in self._stats_names:
            raise ValueError(f'stats {stats_name} does not exist.')

    def accumulate_matrices(
        self, stats_name=None, scale=None, smoothing_weight=None
    ):
        """
        Accumulate the latest fisher values to acc_fisher.
        module.{fisher_type} = fisher
        module.{stats_name}_{fisher_type} = acc_fisher
        """
        self._stats_names.add(stats_name)
        if scale is None:
            scale = self._scale
        if smoothing_weight is None:
            smoothing_weight = self._smoothing_weight

        for module in self._model.modules():
            for mat_type in self._matrix_types:
                matrix = getattr(module, mat_type, None)
                if matrix is None:
                    continue
                matrix.scaling(scale)
                stats_attr = self._get_save_field(mat_type, stats_name)
                stats = getattr(module, stats_attr, None)
                if stats is None:
                    setattr(module, stats_attr, copy.deepcopy(matrix))
                    continue
                if smoothing_weight:
                    w = smoothing_weight
                    stats_ema = stats.scaling(1 - w) + matrix.scaling(w)
                    setattr(module, stats_attr, stats_ema)
                else:
                    stats = stats + matrix
                    setattr(module, stats_attr, stats)

    def save_matrices(self, root, relative_dir='', stats_name=None):
        """
        Save fisher for each fisher_type and for each module.
        module.{stats_name}_{fisher_type} = fisher
        """
        self._check_stats_name(stats_name)

        # save all fisher and collect relative_paths
        relative_paths = {}
        for mat_type in self._matrix_types:
            relative_paths[mat_type] = {}
            for mname, module in self._model.named_modules():
                stats_attr = self._get_save_field(mat_type, stats_name)
                stats = getattr(module, stats_attr, None)
                # if module does not have computed matrices, skip
                if stats is None:
                    continue
                _relative_dir = os.path.join(relative_dir, mat_type, mname)
                rst = stats.save(root, _relative_dir)
                if module is self._model:
                    relative_paths[mat_type].update(rst)
                else:
                    relative_paths[mat_type][mname] = rst

        return relative_paths

    def load_matrices(self, root, relative_paths, matrix_shapes):
        for mat_shape in matrix_shapes:
            if mat_shape not in _supported_shapes:
                raise ValueError(f'Invalid matrix_shape: {mat_shape}. {_supported_shapes} are supported.')

        def root_join(path_or_dict):
            if isinstance(path_or_dict, dict):
                rst = {}
                for k, v in path_or_dict.items():
                    rst[k] = root_join(v)
                return rst
            else:
                return os.path.join(root, path_or_dict)

        paths = root_join(relative_paths)

        for mat_type in self._matrix_types:
            mat_paths = paths.get(mat_type, None)
            if mat_paths is None:
                raise ValueError(f'matrix type {mat_type} does not exist.')

            def _load_path(mat_shape, load_key, path_key, module_name=None):
                try:
                    if module_name:
                        kwargs = {load_key: mat_paths[module_name][path_key]}
                    else:
                        kwargs = {load_key: mat_paths[path_key]}
                    matrix.load(**kwargs)
                except (KeyError, FileNotFoundError):
                    if module_name:
                        raise ValueError(
                            f'{mat_type}.{mat_shape} for module {module_name} does not exist.'
                        )
                    else:
                        raise ValueError(
                            f'{mat_type}.{mat_shape} does not exist.'
                        )

            # load layer-wise matrices
            for mname, module in self._model.named_modules():
                if module is self._model:
                    continue
                if not _requires_matrix(module):
                    continue
                matrix = SymMatrix()
                if SHAPE_LAYER_WISE in matrix_shapes:
                    _load_path(SHAPE_LAYER_WISE, 'path', 'tril', mname)
                if SHAPE_KRON in matrix_shapes:
                    if isinstance(module, _normalizations):
                        _load_path(
                            'unit_wise', 'unit_path', 'unit_wise', mname
                        )
                    else:
                        _load_path(SHAPE_KRON, 'kron_path', 'kron', mname)
                if SHAPE_DIAG in matrix_shapes:
                    _load_path(SHAPE_DIAG, 'diag_path', 'diag', mname)
                setattr(module, mat_type, matrix)

            # full matrix
            if SHAPE_FULL in matrix_shapes:
                matrix = SymMatrix()
                _load_path(SHAPE_FULL, 'path', 'tril')
                setattr(self._model, mat_type, matrix)

    def matrices_exist(self, root, relative_paths, matrix_shapes):
        try:
            self.load_matrices(root, relative_paths, matrix_shapes)
            return True
        except ValueError:
            return False

    def clear_matrices(self, stats_name):
        """
        Clear fisher for each fisher_type and for each module.
        module.{stats_name}_{fisher_type} = fisher
        """
        self._check_stats_name(stats_name)

        # save all fisher and collect relative_paths
        for mat_type in self._matrix_types:
            stats_attr = self._get_save_field(mat_type, stats_name)
            for module in self._model.modules():
                if hasattr(module, stats_attr):
                    delattr(module, stats_attr)

        self._clear_stats(stats_name)

    def matrices_to_vector(self, stats_name):
        """
        Flatten all fisher values.
        module.{stats_name}_{fisher_type} = fisher
        fisher = {
            'diag': {'weight': torch.Tensor, 'bias': torch.Tensor},
            'kron': {'A': torch.Tensor, 'B': torch.Tensor},
            'block_diag': {'F': torch.Tensor},
        }
        """
        self._check_stats_name(stats_name)
        vec = []
        for mat_type in self._matrix_types:
            stats_attr = self._get_save_field(mat_type, stats_name)
            for module in self._model.modules():
                stats = getattr(module, stats_attr, None)
                if stats is None:
                    continue
                vec.extend(stats.to_vector())

        vec = [v.flatten() for v in vec]
        return torch.cat(vec)

    def vector_to_matrices(self, vec, stats_name):
        """
        Unflatten vector like fisher.
        module.{stats_name}_{fisher_type} = fisher
        fisher = {
            'diag': {'weight': torch.Tensor, 'bias': torch.Tensor},
            'kron': {'A': torch.Tensor, 'B': torch.Tensor},
            'block_diag': {'F': torch.Tensor},
        }
        """
        self._check_stats_name(stats_name)

        pointer = 0
        for mat_type in self._matrix_types:
            stats_attr = self._get_save_field(mat_type, stats_name)
            for module in self._model.modules():
                stats = getattr(module, stats_attr, None)
                if stats is None:
                    continue
                pointer = stats.to_matrices(vec, pointer)

        if pointer != torch.numel(vec):
            raise ValueError(f'pointer has to be {torch.numel(vec)}. Got {pointer}.')

    def reduce_matrices(
        self, stats_name=None, is_master=True, all_reduce=False
    ):
        # pack
        packed_tensor = self.matrices_to_vector(stats_name)
        # reduce
        if all_reduce:
            dist.all_reduce(packed_tensor)
        else:
            dist.reduce(packed_tensor, dst=0)
        # unpack
        if is_master or all_reduce:
            self.vector_to_matrices(
                packed_tensor.div_(dist.get_world_size()), stats_name
            )
        dist.barrier()

    def _collect_metrics(
        self,
        matrix_type,
        matrix_shape,
        stats_name,
        metrics_fn,
        reduce_fn,
        init
    ):
        stats_attr = self._get_save_field(matrix_type, stats_name)
        if matrix_shape == SHAPE_FULL:
            matrix = getattr(self._model, stats_attr, None)
            if matrix is None or not matrix.has_data:
                raise ValueError(f'{matrix_type}.{matrix_shape} does not exist.')
            return getattr(matrix, metrics_fn)()

        rst = init
        for mname, module in self._model.named_modules():
            if module is self._model:
                continue
            if not _requires_matrix(module):
                continue
            matrix = getattr(module, stats_attr, None)
            if matrix is None:
                raise ValueError(f'{matrix_type} for {mname} does not exist.')
            if matrix_shape == SHAPE_LAYER_WISE:
                if not matrix.has_data:
                    raise ValueError(f'{matrix_type}.{matrix_shape} for {mname} does not exist.')
                rst = reduce_fn(rst, getattr(matrix, metrics_fn)())
            elif matrix_shape == SHAPE_KRON:
                if isinstance(module, _normalizations):
                    if not matrix.has_unit:
                        raise ValueError(f'{matrix_type}.unit_wise for {mname} does not exist.')
                    rst = reduce_fn(rst, getattr(matrix.unit, metrics_fn)())
                else:
                    if not matrix.has_kron:
                        raise ValueError(f'{matrix_type}.{matrix_shape} for {mname} does not exist.')
                    rst = reduce_fn(rst, getattr(matrix.kron, metrics_fn)())
            elif matrix_shape == SHAPE_DIAG:
                if not matrix.has_diag:
                    raise ValueError(f'{matrix_type}.{matrix_shape} for {mname} does not exist.')
                rst = reduce_fn(rst, getattr(matrix.diag, metrics_fn)())
            else:
                raise ValueError(f'Invalid matrix_shape: {matrix_shape}.')
        return rst

    def get_eigenvalues(self, matrix_type, matrix_shape, stats_name=None):
        def reduce(val1, val2):
            val1.append(val2)
            return val1

        rst = self._collect_metrics(
            matrix_type,
            matrix_shape,
            stats_name,
            metrics_fn='eigenvalues',
            reduce_fn=reduce,
            init=[]
        )
        if not isinstance(rst, torch.Tensor):
            rst = torch.sort(torch.cat(rst), descending=True)[0]
        return rst

    def get_top_eigenvalue(self, matrix_type, matrix_shape, stats_name=None):
        def reduce(val1, val2):
            val1 = max(val1, val2)
            return val1

        rst = self._collect_metrics(
            matrix_type,
            matrix_shape,
            stats_name,
            metrics_fn='top_eigenvalue',
            reduce_fn=reduce,
            init=-1
        )
        return rst

    def get_trace(self, matrix_type, matrix_shape, stats_name=None):
        def reduce(val1, val2):
            val1 += val2
            return val1

        rst = self._collect_metrics(
            matrix_type,
            matrix_shape,
            stats_name,
            metrics_fn='trace',
            reduce_fn=reduce,
            init=0
        )
        return rst

    def get_effective_dim(
        self, matrix_type, matrix_shape, reg, stats_name=None
    ):
        eigs = self.get_eigenvalues(
            matrix_type, matrix_shape, stats_name=stats_name
        )

        return torch.sum(eigs / (eigs + reg))
