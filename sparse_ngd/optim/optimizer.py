"""Implements structured inverse-free KFAC."""

from math import sqrt
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union
from warnings import warn

import torch.distributed as dist
from torch import Tensor, cat, dtype, is_grad_enabled, zeros_like
from torch.nn import Conv2d, Linear, Module, Parameter
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from sparse_ngd.optim.accumulator import BatchAccumulator
from sparse_ngd.optim.utils import process_grad_output, process_input
from sparse_ngd.structures.base import StructuredMatrix
from sparse_ngd.structures.dense import DenseMatrix
from sparse_ngd.structures.diagonal import DiagonalMatrix


class SNGD(Optimizer):
    """Structured inverse-free KFAC based on the empirical Fisher.

    Extends the inverse-free KFAC algorithm from

    - Lin et al. (ICML 2023), 'Simplifying Momentum-based Riemannian Submanifold
      Optimization'

    by allowing for structured matrices. We use their notation in the code.

    Note:
        The optimizer installs forward and backward hooks on known modules.
        These hooks compute quantities required for the pre-conditioner and are
        compatible with gradient accumulation. During `.step`, these quantities
        will be flushed to update the pre-conditioner, compute the approximate
        natural gradient, and update the neural network parameters.

    Attributes:
        SUPPORTED_STRUCTURES: A string-to-class mapping of supported structures.
        SUPPORTED_MODULES: Supported layers.
        _step_supports_amp_scaling: Indicate that `step` handles gradient scaling
            internally if the optimizer is used together with a
            ``torch.cuda.amp.GradScaler``. Before calling this class's ``.step()``,
            the gradient scaler will store the current gradient scale inside
            ``.grad_scale``, and whether ``infs`` occur in the gradients in
            ``.found_inf``. For details, see the implementation of
            ``torch.cuda.amp.GradScaler.step`` at
            https://pytorch.org/docs/stable/_modules/torch/cuda/amp/grad_scaler.html.
    """

    SUPPORTED_STRUCTURES: Dict[str, Type[StructuredMatrix]] = {
        "dense": DenseMatrix,
        "diagonal": DiagonalMatrix,
    }
    SUPPORTED_MODULES: Tuple[Type[Module], ...] = (Linear, Conv2d)
    _step_supports_amp_scaling = True  # do not modify this name (PyTorch convention)!

    def __init__(
        self,
        model: Module,
        params: Union[None, Iterable[Parameter], List[Dict[str, Any]]] = None,
        lr: float = 0.001,  # β₂ in the paper
        momentum: float = 0.9,  # α₂ in the paper
        damping: float = 0.001,  # λ in the paper
        alpha1: float = 0.5,  # α₁ in the paper
        weight_decay: float = 0.0,  # γ in the paper
        T: int = 10,  # T in the paper
        batch_averaged: bool = True,
        lr_cov: Union[float, Callable[[int], float]] = 1e-2,  # β₁ in the paper
        structures: Tuple[str, str] = ("diagonal", "dense"),
        warn_unsupported: bool = True,
        kfac_like: bool = False,
        preconditioner_dtype: Tuple[Union[dtype, None], Union[dtype, None]] = (
            None,
            None,
        ),
    ):
        """Structured inverse-free KFAC optimizer.

        Uses the empirical Fisher. See Lin et al. (2023) for the notation.

        Args:
            model: The neural network whose parameters (or a subset thereof) will be
                trained.
            params: Used to specify the trainable parameters or parameter groups.
                If unspecified, all parameters of ``model`` which are supported by the
                optimizer will be trained. If a list of ``Parameters`` is passed,
                only these parameters will be trained. If a list of dictionaries is
                passed, these will be used as parameter groups.
            lr: (β₂ in the paper) Learning rate for the parameter updates.
                Default: ``0.001``.
            momentum: (α₂ in the paper) Momentum on the parameter updates.
                Default: ``0.9``.
            damping: (λ in the paper) Damping strength used in the updates of
                ``m_K, m_C``. Default: ``0.001``.
            alpha1: (α₁ in the paper) Momentum used in the update of ``m_K, m_C``.
                Default: ``0.5``.
            weight_decay: (γ in the paper) Weight decay on the parameters.
                Default: ``0.0``.
            T: Pre-conditioner update frequency. Default: ``10``.
            batch_averaged: Whether the loss function is a mean over per-sample
                losses. Default is ``True``.
            lr_cov: (β₁ in the paper) Learning rate for the updates of ``m_K, m_C``.
                Default is ``1e-2``. Also allows for a callable which takes the current
                step and returns the current value for ``lr_cov``.
            structures: A 2-tuple of strings specifying the structure of the
                factorizations of ``K`` and ``C``. Possible values are
                (``'dense'``, ``'diagonal'``). Default is (``'dense'``, ``'dense'``).
            warn_unsupported: Only relevant if ``params`` is unspecified. Whether to
                warn if ``model`` contains parameters of layers that are not supported.
                These parameters will not be trained by the optimizer.
            kfac_like: Whether to use an update rule which results in an update close
                to the KFAC optimizer. Default: ``False``. Please see the theorem in
                the paper for more details.
            preconditioner_dtype: Data types used to store the structured
                pre-conditioner matrices (``K`` and ``C``). If ``None``, will use the
                same data type as the parameter for both pre-conditioner matrices. If
                ``(float32, None)``, will use ``float32`` for ``K`` and the same data
                type as the weight for ``C``. Default: ``(None, None)``.

        Raises:
            TypeError: If DataParallel or DistributedDataParallel model wrappers
                are used.
            ValueError: If any of the learning rate and momentum parameters
                (``lr, lr_cov, alpha1, momentum, weight_decay``) are non-positive.
        """
        if isinstance(model, (DP, DDP)):
            raise TypeError(
                "DataParallel and DistributedDataParallel wrappers are not supported. "
                "Use the normal DDP setup without the wrapper for distributed training."
            )

        for x, name in [
            (lr, "lr"),
            (lr_cov, "lr_cov"),
            (alpha1, "alpha1"),
            (momentum, "momentum"),
            (weight_decay, "weight_decay"),
        ]:
            if isinstance(x, float) and x < 0.0:
                raise ValueError(f"{name} must be positive. Got {x}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            damping=damping,
            alpha1=alpha1,
            weight_decay=weight_decay,
            T=T,
            batch_averaged=batch_averaged,
            lr_cov=lr_cov,
            structures=structures,
            kfac_like=kfac_like,
            preconditioner_dtype=preconditioner_dtype,
        )
        if params is None:
            params = self._get_trainable_parameters(
                model, warn_unsupported=warn_unsupported
            )
        super().__init__(params, defaults)
        self.steps = 0

        # for mapping modules to their groups
        self.param_to_group_idx = self._check_param_groups(model)
        # layers whose parameters will be updated
        self.modules, self.hook_handles = self._install_hooks(model)

        # temporarily stores layer inputs during a forward-backward pass
        self.inputs: Dict[Module, Tensor] = {}

        # store matrices for the pre-conditioner
        self.Ks: Dict[Module, StructuredMatrix] = {}
        self.Cs: Dict[Module, StructuredMatrix] = {}

        # store momentum terms for the pre-conditioner matrices
        self.m_Ks: Dict[Module, StructuredMatrix] = {}
        self.m_Cs: Dict[Module, StructuredMatrix] = {}

        # accumulators for H_K and H_C
        self.H_Ks: Dict[Module, BatchAccumulator] = {}
        self.H_Cs: Dict[Module, BatchAccumulator] = {}

        self._initialize_buffers()

    def _get_param_group_entry(self, module: Module, entry: str) -> Any:
        """Get the parameter group that contains the layer's parameters.

        Args:
            module: A supported layer whose parameter group will be returned.
            entry: The entry in the parameter group that will be returned.

        Returns:
            The entry of the parameter group that contains the layer's parameters.
        """
        assert isinstance(module, self.SUPPORTED_MODULES)
        group_idx = self.param_to_group_idx[module.weight.data_ptr()]
        return self.param_groups[group_idx][entry]

    def _check_param_groups(self, model: Module) -> Dict[int, int]:
        """Check parameter groups for conflicts.

        For all supported layers, the parameters must be in the same group because
        we compute and update the pre-conditioner per layer.

        Args:
            model: The model whose layers will be checked.

        Raises:
            ValueError: If parameters in a supported layer are in different groups.

        Returns:
            A dictionary mapping parameter IDs (``.data_ptr()``) to group indices.
        """
        # if KFAC-like update is employed, alpha1 will be ignored
        for idx, group in enumerate(self.param_groups):
            if group["kfac_like"] and group["alpha1"] != 0.0:
                warn(
                    f"Parameter group {idx} has kfac_like=True but was initialized "
                    + f"with non-zero Riemannian momentum (alpha1={group['alpha1']}). "
                    + "Setting alpha' to zero."
                )
                group["alpha1"] = 0.0

        # Find out which parameter is in which group
        param_to_group_idx = {}
        for group_idx, group in enumerate(self.param_groups):
            for param in group["params"]:
                param_to_group_idx[param.data_ptr()] = group_idx

        param_ids = [
            param.data_ptr() for group in self.param_groups for param in group["params"]
        ]
        # check that parameters in a supported module are in the same group
        # all layers that are not containers
        modules = [
            m
            for m in model.modules()
            if len(list(m.modules())) == 1 and isinstance(m, self.SUPPORTED_MODULES)
        ]
        for m in modules:
            m_param_ids = [
                p.data_ptr() for p in m.parameters() if p.data_ptr() in param_ids
            ]
            m_groups = [param_to_group_idx[p_id] for p_id in m_param_ids]
            if len(set(m_groups)) not in [0, 1]:
                raise ValueError(
                    "Parameters of a layer are in different parameter groups. "
                    + f"Layer: {m}. Group index of parameters: {m_groups}."
                )

        return param_to_group_idx

    def _get_trainable_parameters(
        self, model: Module, warn_unsupported: bool
    ) -> List[Parameter]:
        """Return a list containing the model parameters that can be trained.

        Args:
            model: The model whose parameters should be trained.
            warn_unsupported: Whether to warn if ``model`` contains parameters of
                layers that are not supported.

        Returns:
            A list of parameters that can be trained.
        """
        # all layers that are not containers
        named_modules = [
            (name, mod)
            for name, mod in model.named_modules()
            if len(list(mod.modules())) == 1
        ]

        trainable = []
        for name, mod in named_modules:
            mod_trainable = [param for param in mod.parameters() if param.requires_grad]
            if isinstance(mod, self.SUPPORTED_MODULES):
                trainable.extend(mod_trainable)
            elif mod_trainable and warn_unsupported:
                warn(
                    "Found un-supported parameter(s) that will not be trained in "
                    + f"layer {name}: {mod}. To disable this warning, construct the "
                    "optimizer with `warn_unsupported=False`."
                )

        return trainable

    def _initialize_buffers(self):
        """Initialize buffers for ``K, C, m_K, m_C``.

        Writes to ``self.Ks, self.Cs, self.m_Ks, self.m_Cs``.

        ``K, C`` are initialized to identity matrices, their momentum terms
        ``m_K, m_C`` to zero.
        """
        for module in self.modules:
            dim_K, dim_C = self.preconditioner_dims(module)

            dtype = self._get_param_group_entry(module, "preconditioner_dtype")
            dtype_K = dtype[0] if dtype[0] is not None else module.weight.dtype
            dtype_C = dtype[1] if dtype[1] is not None else module.weight.dtype
            device = module.weight.device

            # use the structure specified for the parameter group
            structures = self._get_param_group_entry(module, "structures")
            K_cls = self.SUPPORTED_STRUCTURES[structures[0]]
            C_cls = self.SUPPORTED_STRUCTURES[structures[1]]

            self.Ks[module] = K_cls.eye(dim_K, dtype=dtype_K, device=device)
            self.Cs[module] = C_cls.eye(dim_C, dtype=dtype_C, device=device)

            alpha1 = self._get_param_group_entry(module, "alpha1")
            if alpha1 != 0.0:
                self.m_Ks[module] = K_cls.zeros(dim_K, dtype=dtype_K, device=device)
                self.m_Cs[module] = C_cls.zeros(dim_C, dtype=dtype_C, device=device)

    @staticmethod
    def preconditioner_dims(module: Module) -> Tuple[int, int]:
        """Return the dimensions of the pre-conditioner matrices for a layer.

        Args:
            module: Layer whose pre-conditioner dimensions are returned.

        Returns:
            Tuple of the form ``(dim_K, dim_C)``.

        Raises:
            NotImplementedError: If the module is not supported.
        """
        if isinstance(module, Linear):
            dim_C, dim_K = module.weight.shape
            if module.bias is not None:
                dim_K += 1
        elif isinstance(module, Conv2d):
            dim_K = module.weight.shape[1:].numel()
            if module.bias is not None:
                dim_K += 1
            dim_C = module.weight.shape[0]
        else:
            raise NotImplementedError(f"Initialization not implemented for {module}.")
        return dim_K, dim_C

    def _save_input(self, module: Module, inputs: Tuple[Tensor]):
        """Internally store input of a layer if triggered by update frequency.

        Saves the input to ``self.inputs``.

        Args:
            module: Layer whose input is stored.
            inputs: Inputs to the layer.
        """
        T = self._get_param_group_entry(module, "T")
        if is_grad_enabled() and self.steps % T == 0:
            self.inputs[module] = inputs[0].data

    def _update_preconditioner(self, module: Module, grad_scale: float = 1.0):
        """Update the pre-conditioner matrices and their momenta for a layer.

        Only updates for steps matched by the specified update frequency.
        Flushes the accumulated quantities ``H_K, H_C``.
        Updates internal quantities ``K, C, m_K, m_C``.

        Args:
            module: Layer whose pre-conditioner matrices are updated.
            grad_scale: Scaling factor for the gradients that were used to compute.
                ``H_C``. Default is ``1.0``.
        """
        T = self._get_param_group_entry(module, "T")
        if self.steps % T != 0:
            return

        K, C = self.Ks[module], self.Cs[module]
        # NOTE: Pop such that they will be freed after
        H_K: StructuredMatrix = self.H_Ks.pop(module).value
        H_C: StructuredMatrix = self.H_Cs.pop(module).value

        # un-scale ``H_C = structure(C.T @ (grad_scale * g) @ (grad_scale * g).T @ C)``
        if grad_scale != 1.0:
            # In total we have to divide by ``grad_scale ** 2``. The ``H_C`` computed
            # in the backward pass was already divided by ``grad_scale`` to avoid
            # overflows. Here, we apply the remaining un-scaling
            H_C *= 1 / grad_scale

        # 1) COMPUTE UPDATE
        K_tK = K.from_inner()
        C_tC = C.from_inner()

        p, d = self.preconditioner_dims(module)

        # TODO These don't need to be computed for KFAC-like, but at the moment we
        # need their device and dtype to create the ``eye_like``s below
        tr_H_C = H_C.trace()
        tr_H_K = H_K.trace()

        # hyper-parameters for parameter group of module
        kfac_like = self._get_param_group_entry(module, "kfac_like")
        damping = self._get_param_group_entry(module, "damping")
        structures = self._get_param_group_entry(module, "structures")
        K_cls = self.SUPPORTED_STRUCTURES[structures[0]]
        C_cls = self.SUPPORTED_STRUCTURES[structures[1]]

        # step for m_K
        if kfac_like:
            first_term = H_K
            second_term = K_tK * damping
        else:
            first_term = H_K * (tr_H_C / d)
            c_squared = damping * C_tC.trace()
            second_term = K_tK * (c_squared / d)
        third_term = K_cls.eye(p, dtype=tr_H_K.dtype, device=tr_H_K.device)
        new_m_K = (first_term + second_term - third_term) * 0.5

        # step for m_C
        if kfac_like:
            first_term = H_C
            second_term = C_tC
        else:
            first_term = H_C * (tr_H_K / p)
            kappa_squared = damping * K_tK.trace()
            second_term = C_tC * (kappa_squared / p)
        third_term = C_cls.eye(d, dtype=tr_H_C.dtype, device=tr_H_C.device)
        new_m_C = (first_term + second_term - third_term) * 0.5

        # 2) APPLY UPDATE
        alpha1 = self._get_param_group_entry(module, "alpha1")
        if alpha1 != 0.0:
            new_m_C += self.m_Cs[module] * alpha1
            new_m_K += self.m_Ks[module] * alpha1
            self.m_Cs[module] = new_m_C
            self.m_Ks[module] = new_m_K

        beta1 = self._get_param_group_entry(module, "lr_cov")
        if isinstance(beta1, Callable):  # scheduled
            beta1 = beta1(self.steps)
        self.Ks[module] = K - (K @ new_m_K) * beta1
        self.Cs[module] = C - (C @ new_m_C) * beta1

    def _accumulate_H_terms(
        self, module: Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]
    ):
        """Accumulate the current mini-batch's contribution to ``H_K, H_C`` for a layer.

        Updates the ``H_K, H_C`` buffers for the module.

        Only updates for steps matched by the specified update frequency.
        Requires that the layer inputs have been stored in ``self.inputs``.

        Args:
            module: Layer whose pre-conditioner is updated.
            grad_input: Gradients w.r.t. the input.
            grad_output: Gradients w.r.t. the output.
        """
        T = self._get_param_group_entry(module, "T")
        if self.steps % T != 0:
            return

        batch_averaged = self._get_param_group_entry(module, "batch_averaged")

        # 1) PROCESS INPUTS AND GRAD_OUTPUTS
        a = self.inputs.pop(module)
        batch_size = a.shape[0]
        # For convolutions, unfold the input, for modules with bias terms, append a 1
        a = process_input(a, module)

        g = grad_output[0].data
        # Flatten into matrix, add scaling from batch average
        g = process_grad_output(g, module, batch_averaged)

        # 2) Update H_K, H_C
        K, C = self.Ks[module], self.Cs[module]
        H_K = K.from_inner(X=a.T)

        grad_scale = self._get_grad_scale()
        if grad_scale != 1.0:
            # In total we have to divide by ``grad_scale ** 2``. Here, we only divide
            # by ``grad_scale`` and apply the remaining un-scaling later when updating
            # the pre-conditioner
            g = g / sqrt(grad_scale)
        H_C = C.from_inner(X=g.T)

        # If DDP is used.
        if dist.is_initialized():
            # all-reduce across devices (computes average by default).
            H_K.all_reduce()
            H_C.all_reduce()

        # maybe set up fresh accumulators (they get flushed in `.step`)
        if module not in self.H_Ks:
            self.H_Ks[module] = BatchAccumulator(batch_averaged=batch_averaged)
        if module not in self.H_Cs:
            self.H_Cs[module] = BatchAccumulator(batch_averaged=batch_averaged)

        self.H_Ks[module].update(H_K, batch_size)
        self.H_Cs[module].update(H_C, batch_size)

    def _install_hooks(
        self, model: Module
    ) -> Tuple[List[Module], List[RemovableHandle]]:
        """Install hooks on supported modules to accumulate pre-conditioner quantities.

        Args:
            model: Model whose modules are hooked.

        Returns:
            List of modules that have been hooked and list of the installed hooks'
            handles.
        """
        param_ids = [
            p.data_ptr() for group in self.param_groups for p in group["params"]
        ]
        modules = [
            m
            for m in model.modules()
            if isinstance(m, self.SUPPORTED_MODULES)
            and any(p.data_ptr() in param_ids for p in m.parameters())
        ]
        handles = []
        for module in modules:
            handles.extend(
                (
                    module.register_forward_pre_hook(self._save_input),
                    module.register_full_backward_hook(self._accumulate_H_terms),
                )
            )
        return modules, handles

    def _compute_natural_gradient(
        self, module: Module, grad_scale: float = 1.0
    ) -> Tuple[Tensor, ...]:
        """Compute the natural gradient with the current pre-conditioner for a layer.

        Uses the current value in ``.grad`` to compute the natural gradient.

        Args:
            module: The layer whose natural gradient will be computed.
            grad_scale: Scaling factor for the gradients stored in ``.grad``.
                Default is ``1.0`` (no gradient scaling used).

        Returns:
            The natural gradient for the parameters (weights only or weights + bias)
            of the layer in tuple format.

        Raises:
            NotImplementedError: If the layer is not supported.
        """
        # 1) CONCATENATE GRADIENTS OF WEIGHT AND BIAS AND RESHAPE INTO MATRIX
        if isinstance(module, Conv2d):
            grad_mat = module.weight.grad.data.flatten(start_dim=1)
        elif isinstance(module, Linear):
            grad_mat = module.weight.grad.data
        else:
            raise NotImplementedError(f"Can't get matrix gradient of {module}")

        if module.bias is not None:
            grad_mat = cat([grad_mat, module.bias.grad.data.unsqueeze(1)], dim=1)

        # un-scale gradients
        if grad_scale != 1.0:
            grad_mat = grad_mat / grad_scale

        # 2) COMPUTE THE NATURAL GRADIENT IN CONCATENATED MATRIX FORM

        # We need to compute ``W @ K @ K^T`` where ``W`` is the weight gradient
        # ``K`` supports ``K @ ...`` and ``K^T @ ...``. Hence, we rewrite into
        # ``W @ K @ K^T = ( K @ (K^T @ W^T) )^T``.
        K = self.Ks[module]
        nat_grad = (K @ K.rmatmat(grad_mat.T)).T

        C = self.Cs[module]
        nat_grad = C @ (C.rmatmat(nat_grad))

        # If DDP is used.
        if dist.is_initialized():
            # all-reduce across devices.
            dist.all_reduce(nat_grad, op=dist.ReduceOp.AVG)

        # 3) UN-CONCATENATE, UN-RESHAPE, AND COPY THE NATURAL GRADIENT TO ``.GRAD``
        if module.bias is not None:
            # bias term is stored in last column
            nat_grad_weight, nat_grad_bias = nat_grad[:, :-1], nat_grad[:, -1]
            nat_grad_weight = nat_grad_weight.reshape_as(module.weight)
            nat_grad_bias = nat_grad_bias.reshape_as(module.bias)
            return nat_grad_weight, nat_grad_bias
        else:
            return (nat_grad.reshape_as(module.weight),)

    def step(self, closure: Union[None, Callable[[], Tensor]] = None):
        """Compute natural gradients and update parameters.

        Args:
            closure: Optional closure that evaluates the loss. Not supported.

        Raises:
            NotImplementedError: If a closure is supplied or if ``inf``s are
                encountered by a gradient scaler that is used jointly with the
                optimizer.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")

        grad_scale = self._get_grad_scale()
        found_inf = getattr(self, "found_inf", False)
        if found_inf:
            raise NotImplementedError("Encountered inf in .grad. No policy defined.")

        for module in self.modules:
            self._update_preconditioner(module, grad_scale=grad_scale)
            natural_gradients = self._compute_natural_gradient(
                module, grad_scale=grad_scale
            )
            parameters = (
                [module.weight] if module.bias is None else [module.weight, module.bias]
            )

            # hyper-parameters for group containing module
            weight_decay = self._get_param_group_entry(module, "weight_decay")
            momentum = self._get_param_group_entry(module, "momentum")
            lr = self._get_param_group_entry(module, "lr")

            for p, p_nat_grad in zip(parameters, natural_gradients):
                p_step = p_nat_grad

                # add weight decay
                if weight_decay != 0.0:
                    p_step.add_(p.data, alpha=weight_decay)

                # momentum on previous updates
                if momentum != 0.0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = zeros_like(p.data)

                    p_momentum = param_state["momentum_buffer"]
                    p_momentum.mul_(momentum).add_(p_step)
                    p_step = p_momentum

                p.data.add_(p_step, alpha=-lr)

        self.steps += 1

    def _get_grad_scale(self) -> float:
        """Get the current gradient scale.

        Get current gradient scale if used with ``torch.cuda.amp.GradScaler``,
        see the comment on the class attribute ``_step_supports_amp_scaling`` how
        gradient scales are stored inside an optimizer

        Returns:
            The current gradient scale.
        """
        grad_scale = getattr(self, "grad_scale", None)
        return grad_scale if grad_scale is not None else 1.0
