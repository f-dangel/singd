"""Implements structured inverse-free KFAC."""

from typing import Callable, Dict, Iterable, List, Tuple, Type, Union

from torch import Tensor, cat, is_grad_enabled, zeros_like
from torch.nn import Conv2d, Linear, Module, Parameter
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

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
        These hooks compute the approximate natural gradient and store it in the
        ``.grad`` field of their parameters.

    Attributes:
        SUPPORTED_STRUCTURES: A string-to-class mapping of supported structures.
        SUPPORTED_MODULES: Supported layers.
    """

    SUPPORTED_STRUCTURES: Dict[str, Type[StructuredMatrix]] = {
        "dense": DenseMatrix,
        "diagonal": DiagonalMatrix,
    }
    SUPPORTED_MODULES: Tuple[Type[Module], ...] = (Linear, Conv2d)

    def __init__(
        self,
        model: Module,
        lr: float = 0.001,  # β₂ in the paper
        momentum: float = 0.9,  # α₂ in the paper
        damping: float = 0.001,  # λ in the paper
        alpha1: float = 0.5,  # α₁ in the paper
        weight_decay: float = 0.0,  # γ in the paper
        T: int = 10,  # T in the paper
        batch_averaged: bool = True,
        model_params: Union[None, Iterable[Parameter]] = None,
        lr_cov: float = 1e-2,  # β₁ in the paper
        structures: Tuple[str, str] = ("diagonal", "dense"),
    ):
        """Structured inverse-free KFAC optimizer.

        Uses the empirical Fisher. See Lin et al. (2023) for the notation.

        Args:
            model: The neural network whose parameters (or a subset thereof) will be
                trained.
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
            model_params: Optional sequence of parameters that should be trained.
                If unspecified, all parameters of ``model`` will be trained.
            lr_cov: (β₁ in the paper) Learning rate for the updates of ``m_K, m_C``.
                Default is ``1e-2``.
            structures: A 2-tuple of strings specifying the structure of the
                factorizations of ``K`` and ``C``. Possible values are
                (``'dense'``, ``'diagonal'``). Default is (``'dense'``, ``'dense'``).

        Raises:
            ValueError: If any of the learning rate and momentum parameters
                (``lr, lr_cov, alpha1, momentum, weight_decay``) are non-positive.
        """
        for x, name in [
            (lr, "lr"),
            (lr_cov, "lr_cov"),
            (alpha1, "alpha1"),
            (momentum, "momentum"),
            (weight_decay, "weight_decay"),
        ]:
            if x < 0.0:
                raise ValueError(f"{name} must be positive. Got {x}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(
            model.parameters() if model_params is None else model_params, defaults
        )
        self.lr = lr
        self.momentum = momentum
        self.lr_cov = lr_cov
        self.weight_decay = weight_decay
        self.damping = damping
        self.batch_averaged = batch_averaged
        self.alpha1 = alpha1
        self.T = T
        self.steps = 0

        # layers whose parameters will be updated
        self.modules, self.hook_handles = self._install_hooks(model)

        # temporarily stores layer inputs during a forward-backward pass
        self.inputs: Dict[Module, Tensor] = {}

        self.K_cls = self.SUPPORTED_STRUCTURES[structures[0]]
        self.C_cls = self.SUPPORTED_STRUCTURES[structures[1]]

        # store matrices for the pre-conditioner
        self.Ks: Dict[Module, StructuredMatrix] = {}
        self.Cs: Dict[Module, StructuredMatrix] = {}

        # store momentum terms for the pre-conditioner matrices
        self.m_Ks: Dict[Module, StructuredMatrix] = {}
        self.m_Cs: Dict[Module, StructuredMatrix] = {}

        self._initialize_buffers()

    def _initialize_buffers(self):
        """Initialize buffers for ``K, C, m_K, m_C``.

        Writes to ``self.Ks, self.Cs, self.m_Ks, self.m_Cs``.

        ``K, C`` are initialized to identity matrices, their momentum terms
        ``m_K, m_C`` to zero.

        Raises:
            NotImplementedError: If the initialization is not implemented for a module.
        """
        for module in self.modules:
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
                raise NotImplementedError(
                    f"Initialization not implemented for {module}."
                )

            kwargs = {"dtype": module.weight.dtype, "device": module.weight.device}

            self.Ks[module] = self.K_cls.eye(dim_K, **kwargs)
            self.Cs[module] = self.C_cls.eye(dim_C, **kwargs)

            self.m_Ks[module] = self.K_cls.zeros(dim_K, **kwargs)
            self.m_Cs[module] = self.C_cls.zeros(dim_C, **kwargs)

    def _save_input(self, module: Module, inputs: Tuple[Tensor]):
        """Internally store input of a layer if triggered by update frequency.

        Saves the input to ``self.inputs``.

        Args:
            module: Layer whose input is stored.
            inputs: Inputs to the layer.
        """
        if is_grad_enabled() and self.steps % self.T == 0:
            self.inputs[module] = inputs[0].data

    def _update_preconditioner(
        self, module: Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]
    ):
        """Maybe update the pre-conditioner for a layer.

        Updates the ``K, C, m_K, m_C`` buffers for the module.

        Only updates for steps matched by the specified update frequency.
        Requires that the layer inputs have been stored in ``self.inputs``.

        Args:
            module: Layer whose pre-conditioner is updated.
            grad_input: Gradients w.r.t. the input.
            grad_output: Gradients w.r.t. the output.
        """
        if self.steps % self.T != 0:
            return

        # 1) PROCESS INPUTS AND GRAD_OUTPUTS
        a = self.inputs.pop(module)
        # For convolutions, unfold the input, for modules with bias terms, append a 1
        a = process_input(a, module)

        g = grad_output[0].data
        # Flatten into matrix, add scaling from batch average
        g = process_grad_output(g, module, self.batch_averaged)

        # 2) COMPUTE UPDATE
        K, C = self.Ks[module], self.Cs[module]
        m_K, m_C = self.m_Ks[module], self.m_Cs[module]

        H_K = K.from_inner(X=a.T)
        H_C = C.from_inner(X=g.T)
        K_tK = K.from_inner()
        C_tC = C.from_inner()

        d, p = g.shape[1], a.shape[1]

        c_squared = self.damping * C_tC.trace()
        step_m_K = (
            H_K * (H_C.trace() / d)
            + K_tK * (c_squared / d)
            - self.K_cls.eye(p, dtype=a.dtype, device=a.device)
        ) * 0.5

        kappa_squared = self.damping * K_tK.trace()
        step_m_C = (
            H_C * (H_K.trace() / p)
            + C_tC * (kappa_squared / p)
            - self.C_cls.eye(d, dtype=g.dtype, device=g.device)
        ) * 0.5

        # 3) APPLY UPDATE
        beta1 = self.lr_cov

        self.m_Ks[module] = m_K * self.alpha1 + step_m_K
        self.Ks[module] = K - (K @ self.m_Ks[module]) * beta1

        self.m_Cs[module] = m_C * self.alpha1 + step_m_C
        self.Cs[module] = C - (C @ self.m_Cs[module]) * beta1

    def _install_hooks(
        self, model: Module
    ) -> Tuple[List[Module], List[RemovableHandle]]:
        """Install hooks on supported modules to update the pre-conditioner.

        Args:
            model: Model whose modules are hooked.

        Returns:
            List of modules that have been hooked and list of the installed hooks'
            handles.
        """
        modules = [
            m
            for m in model.modules()
            if isinstance(m, self.SUPPORTED_MODULES)
            and all(p.requires_grad for p in m.parameters())
        ]
        handles = []
        for module in modules:
            handles.extend(
                (
                    module.register_forward_pre_hook(self._save_input),
                    module.register_full_backward_hook(self._update_preconditioner),
                )
            )
        return modules, handles

    def _compute_natural_gradient(self, module: Module) -> Tuple[Tensor, ...]:
        """Compute the natural gradient with the current pre-conditioner for a layer.

        Uses the current value in ``.grad`` to compute the natural gradient.

        Args:
            module: The layer whose natural gradient will be computed.

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

        # 2) COMPUTE THE NATURAL GRADIENT IN CONCATENATED MATRIX FORM

        # We need to compute ``W @ K @ K^T`` where ``W`` is the weight gradient
        # ``K`` supports ``K @ ...`` and ``K^T @ ...``. Hence, we rewrite into
        # ``W @ K @ K^T = ( K @ (K^T @ W^T) )^T``.
        K = self.Ks[module]
        nat_grad = (K @ K.rmatmat(grad_mat.T)).T

        C = self.Cs[module]
        nat_grad = C @ (C.rmatmat(nat_grad))

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
            NotImplementedError: If a closure is supplied.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")

        for module in self.modules:
            natural_gradients = self._compute_natural_gradient(module)
            parameters = (
                [module.weight] if module.bias is None else [module.weight, module.bias]
            )
            for p, p_nat_grad in zip(parameters, natural_gradients):
                p_step = p_nat_grad

                # add weight decay
                if self.weight_decay != 0.0:
                    p_step.add_(p.data, alpha=self.weight_decay)

                # momentum on previous updates
                if self.momentum != 0.0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = zeros_like(p.data)

                    p_momentum = param_state["momentum_buffer"]
                    p_momentum.mul_(self.momentum).add_(p_nat_grad)
                    p_step = p_momentum

                p.data.add_(p_step, alpha=-self.lr)

        self.steps += 1
