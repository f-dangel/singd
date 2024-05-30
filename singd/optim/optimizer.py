"""Implements structured inverse-free KFAC."""

from functools import partial
from math import sqrt
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type, Union
from warnings import simplefilter, warn

import torch.distributed as dist
from torch import Tensor, cat, device, dtype, is_grad_enabled, zeros_like
from torch.nn import Conv2d, Linear, Module, Parameter
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from singd.optim.utils import process_grad_output, process_input
from singd.structures.base import StructuredMatrix
from singd.structures.blockdiagonal import Block30DiagonalMatrix
from singd.structures.dense import DenseMatrix
from singd.structures.diagonal import DiagonalMatrix
from singd.structures.hierarchical import Hierarchical15_15Matrix
from singd.structures.triltoeplitz import TrilToeplitzMatrix
from singd.structures.triutoeplitz import TriuToeplitzMatrix


class SINGD(Optimizer):
    """Structured inverse-free natural gradient descent.

    The algorithm is introduced in [this paper](http://arxiv.org/abs/2312.05705) and
    extends the inverse-free KFAC algorithm from [Lin et al. (ICML
    2023)](https://arxiv.org/abs/2302.09738) with structured pre-conditioner
    matrices.

    Note:
        Uses the empirical Fisher.

    Note:
        (Implementation concept) The optimizer installs a single forward hook on known
        modules. During a forward pass, this hook installs a tensor hook on the layer's
        output which computes the quantities required for the pre-conditioner.
        During `.step`, these quantities will be flushed to update the pre-conditioner,
        compute the approximate natural gradient, and update the network parameters.

    Attributes:
        SUPPORTED_STRUCTURES: A string-to-class mapping of supported structures.
        SUPPORTED_MODULES: Supported layers.
        STATE_ATTRIBUTES: Attributes that belong to the optimizer's state but are
            not stored inside the `self.state` attribute. They will be saved
            and restored when the optimizer is check-pointed (by calling
            [`.state_dict()`](\
https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html) and
            [`.load_state_dict()`](\
https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html)).
        SUPPORTED_LOSS_AVERAGE: Supported loss averaging schemes.
        _step_supports_amp_scaling: Indicates that `step` handles gradient scaling
            internally if the optimizer is used together with a
            [`torch.cuda.amp.GradScaler`](\
https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler).
            Before calling this class's `.step()`, the gradient scaler will store the
            current gradient scale inside `.grad_scale`, and whether `infs` occur in
            the gradients in `.found_inf`. For details, see the implementation of
            [`torch.cuda.amp.GradScaler.step`](\
https://pytorch.org/docs/stable/_modules/torch/cuda/amp/grad_scaler.html).
    """

    SUPPORTED_STRUCTURES: Dict[str, Type[StructuredMatrix]] = {
        "dense": DenseMatrix,
        "diagonal": DiagonalMatrix,
        "block30diagonal": Block30DiagonalMatrix,
        "hierarchical15_15": Hierarchical15_15Matrix,
        "triltoeplitz": TrilToeplitzMatrix,
        "triutoeplitz": TriuToeplitzMatrix,
    }
    SUPPORTED_MODULES: Tuple[Type[Module], ...] = (Linear, Conv2d)
    SUPPORTED_LOSS_AVERAGE: Tuple[Union[None, str], ...] = (
        None,
        "batch",
        "batch+sequence",
    )
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
        loss_average: Union[None, str] = "batch",
        lr_cov: Union[float, Callable[[int], float]] = 1e-2,  # β₁ in the paper
        structures: Tuple[str, str] = ("dense", "dense"),
        kfac_approx: str = "expand",
        warn_unsupported: bool = True,
        kfac_like: bool = False,
        preconditioner_dtype: Tuple[Union[dtype, None], Union[dtype, None]] = (
            None,
            None,
        ),
        init_grad_scale: float = 1.0,
        normalize_lr_cov: bool = False,
    ):  # noqa: D301
        """Structured inverse-free natural gradient descent optimizer.

        Uses the empirical Fisher. See the [paper](http://arxiv.org/abs/2312.05705) for
        the notation.

        Args:
            model: The neural network whose parameters (or a subset thereof) will be
                trained.
            params: Used to specify the trainable parameters or parameter groups.
                If unspecified, all parameters of `model` which are supported by the
                optimizer will be trained. If a list of `Parameters` is passed,
                only these parameters will be trained. If a list of dictionaries is
                passed, these will be used as [parameter groups](\
https://pytorch.org/docs/stable/optim.html#per-parameter-options).
            lr: (\\(\\beta_2\\) in the paper) Learning rate for the parameter updates.
                Default: `0.001`.
            momentum: (\\(\\alpha_2\\) in the paper) Momentum on the parameter updates.
                Default: `0.9`.
            damping: (\\(\\lambda\\) in the paper) Damping strength used in the updates
                of the pre-conditioner momenta \\(\\mathbf{m}_\\mathbf{K}\\) and
                \\(\\mathbf{m}_\\mathbf{C}\\). Default: `0.001`.
            alpha1: (\\(\\alpha_1\\) in the paper) Momentum used in the updates
                of the pre-conditioner momenta \\(\\mathbf{m}_\\mathbf{K}\\) and
                \\(\\mathbf{m}_\\mathbf{C}\\). Default: `0.5`.
            weight_decay: (\\(\\gamma\\) in the paper) Weight decay on the parameters.
                Default: `0.0`.
            T: Pre-conditioner update frequency. Default: `10`.
            loss_average: Whether the loss function is a mean over per-sample
                losses and if yes, over which dimensions the mean is taken.
                If `"batch"`, the loss function is a mean over as many terms as
                the size of the mini-batch. If `"batch+sequence"`, the loss
                function is a mean over as many terms as the size of the
                mini-batch times the sequence length, e.g. in the case of
                language modeling. If `None`, the loss function is a sum. This
                argument is used to ensure that the preconditioner is scaled
                consistently with the loss and the gradient. Default: `"batch"`.
            lr_cov: (β₁ in the paper) Learning rate for the updates of the pre-
                conditioner momenta \\(\\mathbf{m}_\\mathbf{K}\\) and
                \\(\\mathbf{m}_\\mathbf{C}\\). Default is `1e-2`. Also allows for a
                callable which takes the current step and returns the current value for
                `lr_cov`. Using a too large value during the first few steps might lead
                to instabilities because the pre-conditioner is still warming up. In
                that case, try using a schedule which gradually ramps up `lr_cov`. Or
                use a constant value and turn on `normalize_lr_cov` which will at most
                use `lr_cov` during training.
            structures: A 2-tuple of strings specifying the structure of the
                pre-conditioner matrices \\(\\mathbf{K}, \\mathbf{C}\\) and their
                momenta \\(\\mathbf{m}_\\mathbf{K}, \\mathbf{m}_\\mathbf{C}\\).
                Possible values for each entry are `'dense'`, `'diagonal'`,
                `'block30diagonal'`, `'hierarchical15_15'`, `'triltoeplitz'`, and
                `'triutoeplitz'`. Default is (`'dense'`, `'dense'`).
            kfac_approx: A string specifying the KFAC approximation that should
                be used for linear weight-sharing layers, e.g. `Conv2d` modules
                or `Linear` modules that process matrix- or higher-dimensional
                features.
                Possible values are `'expand'` and `'reduce'`.
                See [Eschenhagen et al., 2023](https://arxiv.org/abs/2311.00636)
                for an explanation of the two approximations.
            warn_unsupported: Only relevant if `params` is unspecified. Whether to
                warn if `model` contains parameters of layers that are not supported.
                These parameters will not be trained by the optimizer. Default: `True`
            kfac_like: Whether to use the modified update rule which results in an
                update close to the KFAC optimizer (IKFAC). Default: `False`.
                Please see the theorem in the paper for more details.
            preconditioner_dtype: Data types used to store the structured
                pre-conditioner matrices \\(\\mathbf{K}, \\mathbf{C}\\) and their
                momenta \\(\\mathbf{m}_\\mathbf{K}, \\mathbf{m}_\\mathbf{C}\\).
                If `None`, will use the same data type as the parameter for both
                pre-conditioner matrices and momenta. If `(float32, None)`, will use
                `float32` for \\(\\mathbf{K}, \\mathbf{m}_\\mathbf{K}\\) and the same
                data type as the weight for \\(\\mathbf{C}, \\mathbf{m}_\\mathbf{C}\\).
                Default: `(None, None)`.
            init_grad_scale: Only relevant if using a [`torch.amp.GradScaler`](\
https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler). Initial gradient
                scale of the scaler or a number of similar magnitude. If unspecified,
                the optimizer will still work correctly but the pre-conditioner compu-
                tation in the first backpropagation might be numerically unstable.
                Default: `1.0`.
            normalize_lr_cov: Use [normalized gradient descent](\
https://arxiv.org/abs/1711.05224) to update the pre-conditioner factors. Enabling this
                is a good alternative to scheduling `lr_cov` as we found it to improve
                SINGD's stability in the early phase where the pre-conditioners are
                still warming up. Default: `False`. Requires an additional matrix norm
                computation which will be used to adapt `lr_cov`.
                (Details: To update the pre-conditioner, SINGD performs Riemannian
                gradient descent (RGD) on the pre-conditioner factors. Since it uses
                Riemannian normal coordinates RGD reduces to GD. This allows to apply
                the idea of normalized gradient descent.)

        Raises:
            TypeError: If `DataParallel` or `DistributedDataParallel` model wrappers
                are used.
            ValueError: If any of the learning rate and momentum parameters
                (`lr, lr_cov, alpha1, momentum, weight_decay`) are non-positive.
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
            loss_average=loss_average,
            lr_cov=lr_cov,
            structures=structures,
            kfac_approx=kfac_approx,
            kfac_like=kfac_like,
            preconditioner_dtype=preconditioner_dtype,
            normalize_lr_cov=normalize_lr_cov,
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
        self.module_names, self.hook_handles = self._install_hooks(model)

        # NOTE We use the module names (strings) as keys as they don't change when a
        # model is loaded from a checkpoint (unlike the module objects themselves).

        # store matrices for the pre-conditioner
        self.Ks: Dict[str, StructuredMatrix] = {}
        self.Cs: Dict[str, StructuredMatrix] = {}

        # store momentum terms for the pre-conditioner matrices
        self.m_Ks: Dict[str, StructuredMatrix] = {}
        self.m_Cs: Dict[str, StructuredMatrix] = {}

        # store accumulated H_Ks and H_Cs from one/multiple backward passes
        self.H_Ks: Dict[str, StructuredMatrix] = {}
        self.H_Cs: Dict[str, StructuredMatrix] = {}

        self._initialize_buffers()

        # Book-keeping of `grad_scale`s. We need to keep track of scales of two
        # consecutive steps because we do not have access to the scale at step `t`
        # during its backpropagation, but only when updating the pre-conditioner. Our
        # solution is to un-scale the gradient with the scale from step `t-1` in the
        # backward hook computations for step `t`, then undo and use the scale from
        # step `t` in the pre-conditioner update.
        self._grad_scales: Dict[int, float] = {-1: init_grad_scale}

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
            ValueError: If `kfac_approx` for any param group is not
                `'expand'` or `'reduce'`.
            ValueError: If parameters in a supported layer are in different groups.
            ValueError: If `loss_average` for any param group is not in
                self.SUPPORTED_LOSS_AVERAGE.

        Returns:
            A dictionary mapping parameter IDs (`.data_ptr()`) to group indices.
        """
        for idx, group in enumerate(self.param_groups):
            # if KFAC-like update is employed, alpha1 will be ignored
            if group["kfac_like"] and group["alpha1"] != 0.0:
                warn(
                    f"Parameter group {idx} has kfac_like=True but was initialized "
                    + f"with non-zero Riemannian momentum (alpha1={group['alpha1']}). "
                    + "Setting alpha' to zero."
                )
                group["alpha1"] = 0.0
            if group["kfac_approx"] not in ["expand", "reduce"]:
                raise ValueError(
                    "kfac_approx has to be set to either 'expand' or 'reduce', "
                    f"but was set to {group['kfac_approx']}."
                )
            if group["loss_average"] not in self.SUPPORTED_LOSS_AVERAGE:
                raise ValueError(
                    "loss_average has to be set to one out of "
                    f"{self.SUPPORTED_LOSS_AVERAGE}, but was set to "
                    f"{group['loss_average']}."
                )

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
            warn_unsupported: Whether to warn if `model` contains parameters of
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

    def _get_preconditioner_dtypes_and_device(
        self, module: Module
    ) -> Tuple[Tuple[dtype, dtype], device]:
        """Get the data types and device of the preconditioner matrices.

        Args:
            module: The layer whose preconditioner data types and device will be
                returned.

        Returns:
            A tuple containing the data types of both preconditioner matrices, and
            their device.
        """
        dtype = self._get_param_group_entry(module, "preconditioner_dtype")
        dtype_K = dtype[0] if dtype[0] is not None else module.weight.dtype
        dtype_C = dtype[1] if dtype[1] is not None else module.weight.dtype
        dev = module.weight.device

        return (dtype_K, dtype_C), dev

    def _initialize_buffers(self):
        """Initialize buffers for `K, C, m_K, m_C`.

        Writes to `self.Ks, self.Cs, self.m_Ks, self.m_Cs`.

        `K, C` are initialized to identity matrices, their momentum terms
        `m_K, m_C` to zero.
        """
        for module, name in self.module_names.items():
            dim_K, dim_C = self.preconditioner_dims(module)
            (dtype_K, dtype_C), dev = self._get_preconditioner_dtypes_and_device(module)

            # use the structure specified for the parameter group
            structures = self._get_param_group_entry(module, "structures")
            K_cls = self.SUPPORTED_STRUCTURES[structures[0]]
            C_cls = self.SUPPORTED_STRUCTURES[structures[1]]

            self.Ks[name] = K_cls.eye(dim_K, dtype=dtype_K, device=dev)
            self.Cs[name] = C_cls.eye(dim_C, dtype=dtype_C, device=dev)

            alpha1 = self._get_param_group_entry(module, "alpha1")
            if alpha1 != 0.0:
                self.m_Ks[name] = K_cls.zeros(dim_K, dtype=dtype_K, device=dev)
                self.m_Cs[name] = C_cls.zeros(dim_C, dtype=dtype_C, device=dev)

    @staticmethod
    def preconditioner_dims(module: Module) -> Tuple[int, int]:
        """Return the dimensions of the pre-conditioner matrices for a layer.

        Args:
            module: Layer whose pre-conditioner dimensions are returned.

        Returns:
            Tuple of the form `(dim_K, dim_C)`.

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

    def _update_preconditioner(self, module: Module):
        """Update the pre-conditioner matrices and their momenta for a layer.

        Only updates for steps matched by the specified update frequency.
        Flushes the accumulated quantities `H_K, H_C`.
        Updates internal quantities `K, C, m_K, m_C`.

        Args:
            module: Layer whose pre-conditioner matrices are updated.
        """
        T = self._get_param_group_entry(module, "T")
        if self.steps % T != 0:
            return

        module_name = self.module_names[module]
        K, C = self.Ks[module_name], self.Cs[module_name]
        # NOTE: Pop such that they will be freed after
        H_K: StructuredMatrix = self.H_Ks.pop(module_name)
        H_C: StructuredMatrix = self.H_Cs.pop(module_name)

        # Define `grad_unscaling` to later un-scale
        # `H_C = structure(C.T @ (grad_scale * g) @ (grad_scale * g).T @ C)`.
        prev_grad_scale = self._get_grad_scale(self.steps - 1)
        grad_scale = self._get_grad_scale(self.steps)
        # In total we have to divide by `grad_scale ** 2`. The `H_C` computed
        # in the backward pass was already divided by `prev_grad_scale` to avoid
        # overflows. So we apply the remaining un-scaling. For increased
        # numerical stability we do not scale `H_C` directly but instead
        # include the un-scaling in the update of `m_K` and `m_C`.
        grad_unscaling = prev_grad_scale / grad_scale**2

        # 1) COMPUTE UPDATE
        K_tK = K.from_inner()
        C_tC = C.from_inner()

        # hyper-parameters for parameter group of module
        kfac_like = self._get_param_group_entry(module, "kfac_like")
        damping = self._get_param_group_entry(module, "damping")
        alpha1 = self._get_param_group_entry(module, "alpha1")

        normalize_lr_cov = self._get_param_group_entry(module, "normalize_lr_cov")
        # NOTE If we normalize `lr_cov`, we need to multiply with `1.0 - alpha1`
        # to avoid that the update leads to a strictly increasing largest value
        # in `m_K, m_C`
        scale = 0.5 * (1.0 - alpha1 if normalize_lr_cov else 1.0)

        dim_K, dim_C = self.preconditioner_dims(module)
        (dtype_K, dtype_C), dev = self._get_preconditioner_dtypes_and_device(module)

        # step for m_K
        new_m_K = K.zeros(dim_K, dtype=dtype_K, device=dev)
        new_m_K.add_(
            H_K, alpha=1.0 if kfac_like else grad_unscaling * H_C.average_trace()
        )
        new_m_K.add_(K_tK, alpha=damping * (1.0 if kfac_like else C_tC.average_trace()))
        new_m_K.diag_add_(-1.0).mul_(scale)

        # step for m_C
        new_m_C = C.zeros(dim_C, dtype=dtype_C, device=dev)
        new_m_C.add_(
            H_C,
            alpha=grad_unscaling if kfac_like else grad_unscaling * H_K.average_trace(),
        )
        new_m_C.add_(C_tC, alpha=damping * (1.0 if kfac_like else K_tK.average_trace()))
        new_m_C.diag_add_(-1.0).mul_(scale)

        # 2) APPLY UPDATE
        if alpha1 != 0.0:
            new_m_C.add_(self.m_Cs[module_name], alpha=alpha1)
            new_m_K.add_(self.m_Ks[module_name], alpha=alpha1)
            self.m_Cs[module_name] = new_m_C
            self.m_Ks[module_name] = new_m_K

        # learning rates
        beta1_K = self._get_param_group_entry(module, "lr_cov")
        if isinstance(beta1_K, Callable):  # scheduled
            beta1_K = beta1_K(self.steps)
        beta1_C = self._get_param_group_entry(module, "lr_cov")
        if isinstance(beta1_C, Callable):  # scheduled
            beta1_C = beta1_C(self.steps)

        # perform normalized gradient descent on `K, C` if enabled
        # NOTE We clip the norms below from `1.0` so that the maximum possible
        # learning rate is the `lr_cov` value specified by the user, but no larger
        # to avoid numerical instabilities.
        if normalize_lr_cov:
            beta1_K /= max(1.0, new_m_K.infinity_vector_norm())
            beta1_C /= max(1.0, new_m_C.infinity_vector_norm())

        self.Ks[module_name].add_(K @ new_m_K, alpha=-beta1_K)
        self.Cs[module_name].add_(C @ new_m_C, alpha=-beta1_C)

    def _register_tensor_hook_on_output_to_accumulate_H_terms(
        self, module: Module, inputs: Tuple[Tensor], output: Tensor
    ):
        """Register a tensor hook on the module's output that accumulates the H terms.

        This function can be used as a `forward_hook`.

        Only installs the hook for steps matching the specified update frequency.

        Note:
            The easier way to compute `H_K` and `H_C` would be via a full backward hook
            on the module itself which performs the computation. However, this approach
            breaks down if the output of a layer feeds into an activation with
            `inplace=True` (see https://github.com/pytorch/pytorch/issues/61519). Hence
            we use the workaround
            https://github.com/pytorch/pytorch/issues/61519#issuecomment-883524237, and
            install a module hook which installs a tensor hook on the module's output
            tensor, which performs the accumulation of `H_K` and `H_C`.

        Args:
            module: Layer onto whose output a tensor hook to compute `H_K` and `H_C`
                will be installed.
            inputs: The layer's input tensors.
            output: The layer's output tensor.
        """
        T = self._get_param_group_entry(module, "T")
        if is_grad_enabled() and self.steps % T == 0:
            tensor_hook = partial(self._accumulate_H_terms, module, inputs)
            output.register_hook(tensor_hook)

    def _accumulate_H_terms(
        self, module: Module, inputs: Tuple[Tensor], grad_output: Tensor
    ):
        """Accumulate the current mini-batch's contribution to `H_K, H_C` for a layer.

        Updates the `H_K, H_C` buffers for the module.

        Args:
            module: Layer whose pre-conditioner is updated.
            inputs: The layer's input tensors.
            grad_output: The gradient w.r.t. the output.
        """
        loss_average = self._get_param_group_entry(module, "loss_average")
        kfac_approx = self._get_param_group_entry(module, "kfac_approx")
        module_name = self.module_names[module]

        # load inputs and gradients to the same precision as the pre-conditioner
        (dtype_K, dtype_C), _ = self._get_preconditioner_dtypes_and_device(module)

        # 1) PROCESS INPUTS AND GRAD_OUTPUTS
        a = inputs[0].data.to(dtype_K)
        # Process into matrix according to kfac_approx
        # For convolutions, unfold the input, for modules with bias terms, append a 1
        a = process_input(a, module, kfac_approx)

        g = grad_output.data.to(dtype_C)
        # Process into matrix according to kfac_approx, add scaling from batch average
        g = process_grad_output(g, module, loss_average, kfac_approx)

        # 2) Update H_K, H_C
        K, C = self.Ks[module_name], self.Cs[module_name]
        H_K = K.from_inner(X=a.T)

        # use the gradient scale from the previous step because we do not have access
        # to the actual one during backpropagation
        prev_grad_scale = self._get_grad_scale(self.steps - 1)
        if prev_grad_scale != 1.0:
            # In total we have to divide by `grad_scale ** 2`. Here, we divide
            # by `prev_grad_scale` and apply the remaining un-scaling by dividing
            # by `grad_scale **2 / prev_grad_scale` later when updating the
            # pre-conditioner
            g = g / sqrt(prev_grad_scale)
        H_C = C.from_inner(X=g.T)

        # If DDP is used.
        if dist.is_initialized():
            # all-reduce across devices (computes average by default).
            op = dist.ReduceOp.AVG if loss_average else dist.ReduceOp.SUM
            H_K.all_reduce(op=op)
            H_C.all_reduce(op=op)

        # store or update existing quantities (they get flushed in `.step`)
        self.H_Ks[module_name] = (
            self.H_Ks[module_name].add_(H_K) if module_name in self.H_Ks else H_K
        )
        self.H_Cs[module_name] = (
            self.H_Cs[module_name].add_(H_C) if module_name in self.H_Cs else H_C
        )

    def _install_hooks(
        self, model: Module
    ) -> Tuple[Dict[Module, str], List[RemovableHandle]]:
        """Install hooks on supported modules to accumulate pre-conditioner quantities.

        Args:
            model: Model whose modules are hooked.

        Returns:
            Mapping from hooked modules to their names and list of the installed hooks'
            handles.
        """
        param_ids = [
            p.data_ptr() for group in self.param_groups for p in group["params"]
        ]
        module_names = {
            mod: name
            for (name, mod) in model.named_modules()
            if isinstance(mod, self.SUPPORTED_MODULES)
            and any(p.data_ptr() in param_ids for p in mod.parameters())
        }
        handles = [
            module.register_forward_hook(
                self._register_tensor_hook_on_output_to_accumulate_H_terms
            )
            for module in module_names
        ]
        return module_names, handles

    def _compute_natural_gradient(self, module: Module) -> Tuple[Tensor, ...]:
        """Compute the natural gradient with the current pre-conditioner for a layer.

        Uses the current value in `.grad` to compute the natural gradient.

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

        # un-scale gradients
        grad_scale = self._get_grad_scale(self.steps)
        if grad_scale != 1.0:
            grad_mat = grad_mat / grad_scale

        # 2) COMPUTE THE NATURAL GRADIENT IN CONCATENATED MATRIX FORM
        module_name = self.module_names[module]

        # load the gradient to the pre-conditioner precision while multiplying
        dtype_K, dtype_C = self._get_param_group_entry(module, "preconditioner_dtype")

        # We need to compute `W @ K @ K^T` where `W` is the weight gradient
        # `K` supports `K @ ...` and `K^T @ ...`. Hence, we rewrite into
        # `W @ K @ K^T = ( K @ (K^T @ W^T) )^T`.
        K = self.Ks[module_name]
        nat_grad = (K @ K.rmatmat(grad_mat.T.to(dtype_K))).T

        C = self.Cs[module_name]
        nat_grad = C @ (C.rmatmat(nat_grad.to(dtype_C)))

        # load the pre-conditioned gradient back to the original precision
        nat_grad = nat_grad.to(grad_mat.dtype)

        # If DDP is used.
        if dist.is_initialized():
            # all-reduce across devices.
            loss_average = self._get_param_group_entry(module, "loss_average")
            op = dist.ReduceOp.AVG if loss_average else dist.ReduceOp.SUM
            dist.all_reduce(nat_grad, op=op)

        # 3) UN-CONCATENATE, UN-RESHAPE, AND COPY THE NATURAL GRADIENT TO `.GRAD`
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

        Warn the user when the step is skipped because `inf`s were found in
        the gradients.

        Args:
            closure: Optional closure that evaluates the loss. Not supported.

        Raises:
            NotImplementedError: If a closure is supplied or if `inf`s are
                encountered by a gradient scaler that is used jointly with the
                optimizer.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")

        # Set current gradient scale if used with `torch.cuda.amp.GradScaler`
        # and store it internally. See the comment on the class attribute
        # `_step_supports_amp_scaling` how gradient scales are communicated to
        # an optimizer.
        try:
            # see if `grad_scale` was externally supplied by the user (e.g. when
            # using gradient clipping via `torch.cuda.amp.GradScaler.unscale_`)
            self._get_grad_scale(self.steps)
        except KeyError:
            # `grad_scale` was supplied to the optimizer via `scaler.step`,
            # or no gradient scaler is used.
            grad_scale = getattr(self, "grad_scale", Tensor([1.0])).item()
            self.set_current_grad_scale(grad_scale)

        found_inf = getattr(self, "found_inf", False)
        if found_inf:
            # Skip the update step if `inf`s were encountered in the gradients.
            # The `GradScaler` will adjust the scale in this case.
            simplefilter("always", UserWarning)  # Warn every time this happens.
            warn("Encountered inf in gradients. Skipping update.")
            # Also, empty the accumulators because this step will be skipped.
            for name in self.module_names.values():
                if name in self.H_Ks:
                    del self.H_Ks[name]
                if name in self.H_Cs:
                    del self.H_Cs[name]
        else:
            self._step()

        self.steps += 1

        # remove `grad_scale`s that are not required anymore
        self._remove_used_grad_scales()

    def _step(self):
        """Compute natural gradients and update parameters."""
        for module in self.module_names:
            self._update_preconditioner(module)
            natural_gradients = self._compute_natural_gradient(module)
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

    def _get_grad_scale(self, t: int) -> float:
        """Get the gradient scale used in the backpropagation of step `t`.

        Args:
            t: The step for which the gradient scale is requested.

        Returns:
            The gradient scale at step `t`.
        """
        return self._grad_scales[t]

    def _remove_used_grad_scales(self):
        """Remove gradient scales that are not required anymore.

        Modifies `self._grad_scales` in-place.
        """
        drop = [t for t in self._grad_scales if t < self.steps - 1]
        for t in drop:
            del self._grad_scales[t]

    def set_current_grad_scale(self, grad_scale: float):
        """Store the current gradient scale internally.

        Warn the user if `init_grad_scale` was not specified but a scaler is used.

        Args:
            grad_scale: The current gradient scale.
        """
        self._grad_scales[self.steps] = grad_scale

        if self.steps == 0:
            init_scale, this_scale = self._get_grad_scale(-1), self._get_grad_scale(0)
            if init_scale == 1.0 and this_scale != 1.0:
                warn(
                    f"Detected non-zero gradient scaling ({this_scale} at step 0). "
                    + f"Consider passing a value similar to {this_scale} to "
                    "`init_grad_scale` when initializing the optimizer as this will "
                    f"improve numerical stability (was initialized to {init_scale})."
                )

    STATE_ATTRIBUTES: List[str] = [
        "steps",
        "_grad_scales",
        "Ks",
        "Cs",
        "m_Ks",
        "m_Cs",
        "H_Ks",
        "H_Cs",
    ]

    def state_dict(self) -> Dict[str, Any]:
        """Return a save-able state of the optimizer.

        Returns:
            A dictionary containing the optimizer state.
        """
        state_dict = super().state_dict()

        for name in self.STATE_ATTRIBUTES:
            assert name not in state_dict.keys()
            state_dict[name] = getattr(self, name)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load an optimizer state.

        Args:
            state_dict: A dictionary containing a valid state obtained from this
                class's `.state_dict()` method.
        """
        attributes = {name: state_dict.pop(name) for name in self.STATE_ATTRIBUTES}
        super().load_state_dict(state_dict)

        for name, value in attributes.items():
            setattr(self, name, value)
