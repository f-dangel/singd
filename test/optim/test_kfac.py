from test.optim.utils import jacobians_naive
from typing import List, Tuple

import torch
from pytest import mark
from torch import Tensor, device, is_grad_enabled
from torch.nn import AdaptiveAvgPool2d, Conv2d, Flatten, Linear, Module, Sequential
from torch.nn.utils import parameters_to_vector
from torch.utils.hooks import RemovableHandle

from singd.optim.utils import process_grad_output, process_input

# Set double dtype as default to avoid numerical issues.
torch.set_default_dtype(torch.float64)

IN_DIM = 3
HID_DIM = 5
REP_DIM = 2
OUT_DIM = 1
N_SAMPLES = 4
C_in = 3
C_out = 2
H_in = W_in = 16
K = 4


@mark.parametrize("setting", ["expand", "reduce"])
@mark.parametrize(
    "batch_averaged", [True, False], ids=["batch_averaged", "not_averaged"]
)
@mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@mark.parametrize("device", [device("cpu"), device("cuda:0")], ids=["cpu", "gpu"])
def test_kfac_single_linear_module(
    setting: str, batch_averaged: bool, bias: bool, device: device
):
    # Fix random seed.
    torch.manual_seed(711)
    # Set up inputs x.
    x = torch.randn((N_SAMPLES, REP_DIM, IN_DIM), device=device)
    n_loss_terms = N_SAMPLES * REP_DIM if setting == "expand" else N_SAMPLES

    # Set up one-layer linear network for inputs with additional REP_DIM.
    model = WeightShareModel(
        Linear(in_features=IN_DIM, out_features=OUT_DIM, bias=bias),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    # Jacobians.
    Js, f = jacobians_naive(model, x, setting)
    assert f.shape == (n_loss_terms, OUT_DIM)
    assert Js.shape == (n_loss_terms, OUT_DIM, num_params)
    Js = Js.flatten(start_dim=1)

    # Exact Fisher/GGN.
    exact_F = Js.T @ Js  # regression
    assert exact_F.shape == (num_params, num_params)
    if batch_averaged:
        exact_F /= n_loss_terms

    # K-FAC Fisher/GGN.
    kfac = KFACMSE(model, batch_averaged, setting)
    kfac.forward_and_backward(x)
    F = kfac.get_full_kfac_matrix()
    assert F.shape == (num_params, num_params)

    # Compare true Fisher/GGN against K-FAC Fisher/GGN (should be exact).
    assert torch.allclose(F.diag(), exact_F.diag())  # diagonal comparison
    assert torch.allclose(F, exact_F)  # full comparison


@mark.parametrize("setting", ["expand", "reduce"])
@mark.parametrize(
    "batch_averaged", [True, False], ids=["batch_averaged", "not_averaged"]
)
@mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@mark.parametrize("device", [device("cpu"), device("cuda:0")], ids=["cpu", "gpu"])
def test_kfac_deep_linear(
    setting: str, batch_averaged: bool, bias: bool, device: device
):
    # Fix random seed.
    torch.manual_seed(711)
    # Set up inputs x.
    x = torch.randn((N_SAMPLES, REP_DIM, IN_DIM), device=device)
    n_loss_terms = N_SAMPLES * REP_DIM if setting == "expand" else N_SAMPLES

    # Set up two-layer linear network for inputs with additional REP_DIM.
    model = WeightShareModel(
        Linear(in_features=IN_DIM, out_features=HID_DIM, bias=bias),
        Linear(in_features=HID_DIM, out_features=OUT_DIM, bias=bias),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_layer1 = sum(p.numel() for p in model[0].parameters())

    # Jacobians.
    Js, f = jacobians_naive(model, x, setting)
    assert f.shape == (n_loss_terms, OUT_DIM)
    assert Js.shape == (n_loss_terms, OUT_DIM, num_params)
    Js = Js.flatten(start_dim=1)

    # Exact Fisher/GGN.
    exact_F = Js.T @ Js  # regression
    assert exact_F.shape == (num_params, num_params)
    if batch_averaged:
        exact_F /= n_loss_terms

    # K-FAC Fisher/GGN.
    kfac = KFACMSE(model, batch_averaged, setting)
    kfac.forward_and_backward(x)
    F = kfac.get_full_kfac_matrix()
    assert F.shape == (num_params, num_params)

    # Compare true Fisher/GGN against K-FAC Fisher/GGN block diagonal (should be exact).
    assert torch.allclose(F.diag(), exact_F.diag())  # diagonal comparison
    assert torch.allclose(
        F[:num_params_layer1, :num_params_layer1],
        exact_F[:num_params_layer1, :num_params_layer1],
    )  # full comparison layer 1.
    assert torch.allclose(
        F[num_params_layer1:, num_params_layer1:],
        exact_F[num_params_layer1:, num_params_layer1:],
    )  # full comparison layer 2.


@mark.parametrize("setting", ["expand", "reduce"])
@mark.parametrize(
    "batch_averaged", [True, False], ids=["batch_averaged", "not_averaged"]
)
@mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@mark.parametrize("device", [device("cpu"), device("cuda:0")], ids=["cpu", "gpu"])
def test_kfac_single_conv2d_module(
    setting: str, batch_averaged: bool, bias: bool, device: device
):
    # Fix random seed.
    torch.manual_seed(711)
    # Set up inputs x.
    x = torch.randn((N_SAMPLES, C_in, H_in, W_in), device=device)
    n_loss_terms = N_SAMPLES  # Only reduce setting.

    # Set up model with conv layer, average pooling, and linear output layer.
    model = Sequential(
        Conv2d(C_in, C_out, K, padding=K // 2, bias=bias),
        AdaptiveAvgPool2d(1),
        Flatten(start_dim=1),
        Linear(C_out, OUT_DIM, bias=bias),
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_conv_params = sum(p.numel() for p in model[0].parameters())

    # Jacobians.
    Js, f = jacobians_naive(model, x, setting)
    assert f.shape == (n_loss_terms, OUT_DIM)
    assert Js.shape == (n_loss_terms, OUT_DIM, num_params)
    Js = Js.flatten(start_dim=1)

    # Exact Fisher/GGN.
    exact_F = Js.T @ Js  # regression
    assert exact_F.shape == (num_params, num_params)
    if batch_averaged:
        exact_F /= n_loss_terms

    # K-FAC Fisher/GGN.
    kfac = KFACMSE(model, batch_averaged, setting)
    kfac.forward_and_backward(x)
    F = kfac.get_full_kfac_matrix()
    assert F.shape == (num_params, num_params)

    if setting == "reduce":
        # KFAC-reduce should be exact for this setting.
        # Compare true Fisher/GGN against K-FAC Fisher/GGN diagonal.
        assert torch.allclose(F.diag(), exact_F.diag())
        # Compare true Fisher/GGN against K-FAC Fisher/GGN for the Conv2d layer.
        assert torch.allclose(
            F[:num_conv_params, :num_conv_params],
            exact_F[:num_conv_params, :num_conv_params],
        )
    else:
        # KFAC-expand should not be exact for this setting.
        # Compare true Fisher/GGN against K-FAC Fisher/GGN diagonal.
        assert not torch.allclose(F.diag(), exact_F.diag())


class WeightShareModel(Sequential):
    def forward(self, x: Tensor, setting: str):
        assert setting in ["expand", "reduce"]
        x = super().forward(x)
        if setting == "expand":
            # Example: Transformer for translation
            # (REP_DIM = sequence length).
            # (N_SAMPLES, REP_DIM, OUT_DIM) -> (N_SAMPLES * REP_DIM, OUT_DIM)
            return x.view(-1, OUT_DIM)
        # Example: Vision transformer for image classification
        # (REP_DIM = image patches).
        # (N_SAMPLES, REP_DIM, OUT_DIM) -> (N_SAMPLES, OUT_DIM)
        return x.mean(dim=1)


class KFACMSE:
    def __init__(self, model: Module, batch_averaged: bool, setting: str):
        self.model = model
        self.batch_averaged = batch_averaged
        self.setting = setting

        # Install forward and backward hooks.
        self.hook_handles = self._install_hooks()

    def forward_and_backward(self, x: Tensor):
        # Forward pass.
        kwargs = (
            {"setting": self.setting}
            if isinstance(self.model, WeightShareModel)
            else {}
        )
        logits: Tensor = self.model(x, **kwargs)
        # Backward pass for each output dimension.
        n_dims = logits.size(-1)
        for i in range(n_dims):
            logits_i = logits[:, i]
            loss = logits_i.mean() if self.batch_averaged else logits_i.sum()
            loss.backward(retain_graph=i < n_dims - 1)

    def get_kfac_blocks(self) -> List[Tensor]:
        # Compute the K-FAC approximation blocks for each layer.
        blocks = []
        for module in self.model.modules():
            if not isinstance(module, (Linear, Conv2d)):
                continue
            if not hasattr(module, "kfac_g"):
                raise ValueError("forward_and_backward() has to be called first.")
            # Get Kronecker factor ingredients stored as module attributes.
            a: Tensor = module.kfac_a
            g: Tensor = module.kfac_g
            # Compute Kronecker product of both factors.
            block = torch.kron(g.T @ g, a.T @ a)
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
        blocks = self.get_kfac_blocks()
        return torch.block_diag(*blocks)

    def _install_hooks(self) -> List[RemovableHandle]:
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
        a = inputs[0].data.detach()
        a = process_input(a, module, kfac_approx=self.setting)
        module.kfac_a = a

    def _set_g(
        self, module: Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]
    ):
        del grad_input
        g = grad_output[0].data.detach()
        g = process_grad_output(
            g, module, batch_averaged=self.batch_averaged, kfac_approx=self.setting
        )
        module.kfac_g = g
