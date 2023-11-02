"""Tests for the KFAC approximation of the Fisher/GGN."""

from test.optim.utils import Transpose, jacobians_naive
from test.utils import DEVICE_IDS, DEVICES
from typing import Callable, List, Tuple, Union

from einops import rearrange, reduce
from pytest import mark
from torch import (
    Tensor,
    allclose,
    block_diag,
    cat,
    device,
    float64,
    kron,
    manual_seed,
    randn,
)
from torch.nn import AdaptiveAvgPool2d, Conv2d, Flatten, Linear, Module, Sequential
from torch.utils.hooks import RemovableHandle

from singd.optim.utils import process_grad_output, process_input

IN_DIM = 3
HID_DIM = 5
REP_DIM = 2  # weight-sharing dimension
OUT_DIM = 2
N_SAMPLES = 4
C_in = 3  # input channels
C_out = 2  # output channels
H_in = W_in = 16  # input height and width
H_out, W_out = H_in + 1, W_in + 1  # output height and width
K = 4  # kernel size
# Use double dtype to avoid numerical issues.
DTYPE = float64


def conv2d_model(setting: str, bias: bool) -> Sequential:
    """Model with a `Conv2d` module in the expand or the reduce setting.

    Args:
        setting: KFAC approximation setting. Possible values are `"expand"`
            and `"reduce"`.
        bias: Whether to use a bias term for `Conv2d` and `Linear` modules.

    Raises:
        ValueError: If `setting` is neither `"expand"` nor `"reduce"`.

    Returns:
        Model with a `Conv2d` module in the expand or the reduce setting.
    """
    if setting == "expand":
        return Sequential(
            Conv2d(C_in, C_out, K, padding=K // 2, bias=bias),
            Flatten(start_dim=2),
            Transpose(dim0=1, dim1=2),
            Linear(C_out, OUT_DIM, bias=bias),
            Flatten(start_dim=0, end_dim=-2),
        )
    elif setting == "reduce":
        return Sequential(
            Conv2d(C_in, C_out, K, padding=K // 2, bias=bias),
            AdaptiveAvgPool2d(1),
            Flatten(start_dim=1),
            Linear(C_out, OUT_DIM, bias=bias),
        )
    else:
        raise ValueError(f"Unknown setting {setting}.")


# Models for tests.
MODELS = {
    # Single linear layer.
    "single_linear": lambda bias: WeightShareModel(
        Linear(in_features=IN_DIM, out_features=OUT_DIM, bias=bias),
    ),
    # 3-layer deep linear network.
    "deep_linear": lambda bias: WeightShareModel(
        Linear(in_features=IN_DIM, out_features=HID_DIM, bias=bias),
        Linear(in_features=HID_DIM, out_features=HID_DIM, bias=bias),
        Linear(in_features=HID_DIM, out_features=OUT_DIM, bias=bias),
    ),
    # Single convolutional layer + linear output layer.
    "conv2d": conv2d_model,
}


@mark.parametrize("model", MODELS.items(), ids=MODELS.keys())
@mark.parametrize("setting", ["expand", "reduce"])
@mark.parametrize("averaged", [True, False], ids=["loss_average", "not_averaged"])
@mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@mark.parametrize("device", DEVICES, ids=DEVICE_IDS)
def test_kfac(
    model: Tuple[str, Callable],
    setting: str,
    averaged: bool,
    bias: bool,
    device: device,
):
    """Test KFAC using (deep) linear models with weight sharing and the MSE loss.

    See [Eschenhagen et al., 2023](https://arxiv.org/abs/2311.00636) for more
    details on the conditions for KFAC-expand and KFAC-reduce being exact.

    Args:
        model: Tuple of model name and function takes `bias` as input and
            returns the model.
        setting: KFAC approximation setting. Either `"expand"` or `"reduce"`.
        averaged: Whether the loss uses a mean reduction.
        bias: Whether to use a bias term.
        device: Device to run the test on.

    Raises:
        AssertionError: If the KFAC approximation is not exact.
    """
    # Fix random seed.
    manual_seed(711)

    # Setup model and inputs x.
    model_name, model_fn = model

    # Set appropriate loss_average argument based on averaged and setting.
    if averaged:
        if setting == "expand":
            loss_average = "batch+sequence"
        else:
            loss_average = "batch"
    else:
        loss_average = None

    if model_name == "conv2d":
        model: Module = model_fn(setting, bias)
        x = randn((N_SAMPLES, C_in, H_in, W_in), dtype=DTYPE, device=device)
        n_loss_terms = N_SAMPLES * H_out * W_out if setting == "expand" else N_SAMPLES
    else:
        model: Module = model_fn(bias)
        x = randn((N_SAMPLES, REP_DIM, IN_DIM), dtype=DTYPE, device=device)
        n_loss_terms = N_SAMPLES * REP_DIM if setting == "expand" else N_SAMPLES
    model.to(device, DTYPE)

    num_params_per_layer = [
        sum(p.numel() for p in module.parameters())
        for module in model.modules()
        if isinstance(module, (Linear, Conv2d))
    ]
    num_params = sum(num_params_per_layer)

    # Jacobians.
    Js, f = jacobians_naive(model, x, setting)
    assert f.shape == (n_loss_terms, OUT_DIM)
    assert Js.shape == (n_loss_terms * OUT_DIM, num_params)

    # Exact Fisher/GGN.
    exact_F = Js.T @ Js  # regression
    assert exact_F.shape == (num_params, num_params)
    if loss_average:
        exact_F /= n_loss_terms

    # K-FAC Fisher/GGN.
    kfac = KFACMSE(model, loss_average, setting)
    kfac.forward_and_backward(x)
    F = kfac.get_full_kfac_matrix()
    assert F.shape == (num_params, num_params)

    # Compare true Fisher/GGN against K-FAC Fisher/GGN (should be exact).
    assert allclose(F.diag(), exact_F.diag())  # diagonal comparison
    # Layer-wise comparison.
    for i, n_params in enumerate(num_params_per_layer):
        prev_n_params = sum(num_params_per_layer[:i])
        layer_start = prev_n_params
        layer_stop = prev_n_params + n_params
        assert allclose(
            F[layer_start:layer_stop, layer_start:layer_stop],
            exact_F[layer_start:layer_stop, layer_start:layer_stop],
        )


class WeightShareModel(Sequential):
    """Sequential model with processing of the weight-sharing dimension.

    Wraps a `Sequential` model, but processes the weight-sharing dimension based
    on the `setting` before it returns the output of the sequential model.
    Assumes that the output of the sequential model is of shape
    (N_SAMPLES, REP_DIM, OUT_DIM).
    """

    def forward(self, x: Tensor, setting: str) -> Tensor:
        """Forward pass with processing of the weight-sharing dimension.

        Args:
            x: Input to the forward pass.
            setting: KFAC approximation setting. Possible values are `'expand'`
                and `'reduce'`.

        Returns:
            Output of the sequential model with processed weight-sharing dimension.

        Raises:
            AssertionError: If `setting` is neither `'expand'` nor `'reduce'`.
        """
        assert setting in {"expand", "reduce"}
        x = super().forward(x)
        if setting == "expand":
            # Example: Transformer for translation
            # (REP_DIM = sequence length).
            # (N_SAMPLES, REP_DIM, OUT_DIM) -> (N_SAMPLES * REP_DIM, OUT_DIM)
            return rearrange(x, "batch shared features -> (batch shared) features")
        # Example: Vision transformer for image classification
        # (REP_DIM = image patches).
        # (N_SAMPLES, REP_DIM, OUT_DIM) -> (N_SAMPLES, OUT_DIM)
        return reduce(x, "batch shared features -> batch features", "mean")


class KFACMSE:
    """Class for computing the KFAC approximation with the MSE loss."""

    def __init__(self, model: Module, loss_average: Union[None, str], setting: str):
        """Initialize the KFAC approximation class.

        Installs forward and backward hooks to the model.

        Args:
            model: The model.
            loss_average: Whether the loss function is a mean over per-sample
                losses and if yes, over which dimensions the mean is taken.
                If `"batch"`, the loss function is a mean over as many terms as
                the size of the mini-batch. If `"batch+sequence"`, the loss
                function is a mean over as many terms as the size of the
                mini-batch times the sequence length, e.g. in the case of
                language modeling. If `None`, the loss function is a sum. This
                argument is used to ensure that the preconditioner is scaled
                consistently with the loss and the gradient. Default: `"batch"`.
            setting: KFAC approximation setting. Possible values are `'expand'`
                and `'reduce'`.
        """
        self.model = model
        self.loss_average = loss_average
        self.setting = setting

        # Install forward and backward hooks.
        self.hook_handles = self._install_hooks()

    def forward_and_backward(self, x: Tensor):
        """Forward and backward pass.

        Args:
            x: Input tensor for the forward pass.
        """
        # Forward pass.
        kwargs = (
            {"setting": self.setting}
            if isinstance(self.model, WeightShareModel)
            else {}
        )
        logits: Tensor = self.model(x, **kwargs)
        # Since we only consider the MSE loss, we do not need to explicitly
        # consider the loss function or labels, as the MSE loss Hessian w.r.t.
        # the logits is the precision matrix of the Gaussian likelihood.
        # With other words, we only need to compute the Jacobian of the logits
        # w.r.t. the parameters. This requires one backward pass per output
        # dimension.
        n_dims = logits.size(-1)
        for i in range(n_dims):
            logits_i = logits[:, i]
            # Mean or sum reduction over `n_loss_terms`.
            loss = logits_i.mean() if self.loss_average else logits_i.sum()
            loss.backward(retain_graph=i < n_dims - 1)

    def get_kfac_blocks(self) -> List[Tensor]:
        """Computes the KFAC approximation blocks for each layer.

        Returns:
            A list of KFAC approximation blocks for each layer.

        Raises:
            ValueError: If `forward_and_backward()` has not been called before.
        """
        # Compute the K-FAC approximation blocks for each layer.
        blocks = []
        for module in self.model.modules():
            if not isinstance(module, (Linear, Conv2d)):
                continue
            if not hasattr(module, "kfac_g"):
                raise ValueError("forward_and_backward() has to be called first.")
            # Get Kronecker factor ingredients stored as module attributes.
            a: Tensor = module.kfac_a
            g: Tensor = cat(module.kfac_g)
            # Compute Kronecker product of both factors.
            block = kron(g.T @ g, a.T @ a)
            # When a bias is used we have to reorder the rows and columns of the
            # block to match the order of the parameters in the naive Jacobian
            # implementation.
            if module.bias is not None:
                num_params = sum(p.numel() for p in module.parameters())
                in_dim_w = (
                    module.weight.shape[1]
                    if isinstance(module, Linear)
                    else module.weight.shape[1:].numel()  # in_dim_w = C_in * K * K
                )
                # Get indices of bias parameters.
                idx_b = list(range(in_dim_w, num_params, in_dim_w + 1))
                # Get indices of weight parameters.
                idx_w = [n for n in range(num_params) if n not in idx_b]
                # Reorder rows and columns.
                idx = idx_w + idx_b
                block = block[idx][:, idx]
            blocks.append(block)
        return blocks

    def get_full_kfac_matrix(self) -> Tensor:
        """Computes the full block-diagonal KFAC approximation matrix.

        Returns:
            The full block-diagonal KFAC approximation matrix.
        """
        blocks = self.get_kfac_blocks()
        return block_diag(*blocks)

    def _install_hooks(self) -> List[RemovableHandle]:
        """Installs forward and backward hooks to the model.

        Returns:
            A list of handles to the installed hooks.
        """
        handles = []
        for module in self.model.modules():
            if not isinstance(module, (Linear, Conv2d)):
                continue
            handles.extend(
                (
                    module.register_forward_pre_hook(self._set_a),
                    module.register_full_backward_hook(self._set_g),
                )
            )
        return handles

    def _set_a(self, module: Module, inputs: Tuple[Tensor]):
        """Computes and stores the Kronecker factor ingredient `a`.

        Args:
            module: The current layer.
            inputs: Inputs to the layer.
        """
        a = inputs[0].data.detach()
        a = process_input(a, module, kfac_approx=self.setting)
        module.kfac_a = a

    def _set_g(
        self, module: Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]
    ):
        """Computes and stores the Kronecker factor ingredient `g`.

        Args:
            module: The current layer.
            grad_input: Gradients w.r.t. the layer inputs.
            grad_output: Gradients w.r.t. the layer outputs.
        """
        g = grad_output[0].data.detach()
        g = process_grad_output(
            g, module, loss_average=self.loss_average, kfac_approx=self.setting
        )
        if hasattr(module, "kfac_g"):
            module.kfac_g.append(g)
        else:
            module.kfac_g = [g]
