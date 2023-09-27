# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import torch.optim as optim
from torch.nn import Conv2d, Linear
from torch.optim import Optimizer
# from .myadamw import MyAdamW as AdamW

from sparse_ngd.optim.optimizer import SNGD
from sparse_ngd.experimental.kfac_helpers import My_KFAC


class My_SNGD(Optimizer):
    def __init__(
        self,
        model,
        lr,
        momentum,
        damping,
        alpha1,
        weight_decay,
        T,
        lr_cov,
        preconditioner_dtype=(torch.float32, torch.float32),
        structures=("dense", "dense"),
        kfac_like=False,
        batch_averaged=True,
        using_adamw: bool = False,
        adamw_eps = 1e-8,
        adamw_beta1 = 0.9,
        adamw_beta2 = 0.999,
        warmup_factor = 10,
        using_constant_adamw_lr = False,
    ):
        conv_params = [
            p for m in model.modules() if isinstance(m, Conv2d) for p in m.parameters()
        ]
        linear_params = [
            p for m in model.modules() if isinstance(m, Linear) for p in m.parameters()
        ]
        sngd_params = [p.data_ptr() for p in conv_params + linear_params]
        other_params = [
            p for p in model.parameters() if p.data_ptr() not in sngd_params
        ]

        assert len(other_params) + len(conv_params) + len(linear_params) == len(
            list(model.parameters())
        )

        sngd_hyperparams = {
            "lr": lr,
            "momentum": momentum,
            "damping": damping,
            "alpha1": alpha1,
            "weight_decay": weight_decay,
            "T": T,
            "batch_averaged": batch_averaged,
            "lr_cov": lr_cov,
            "structures": structures,
            "kfac_like": kfac_like,
            "preconditioner_dtype": preconditioner_dtype,
        }
        self.weight_decay = weight_decay
        self.lr_cov = lr_cov

        conv_group = {"params": conv_params, **sngd_hyperparams}
        linear_group = {"params": linear_params, **sngd_hyperparams}

        param_groups = [conv_group, linear_group]
        self._sngd_opt = SNGD(model, params=param_groups, init_grad_scale=65536.0)

        if using_adamw:
            self.using_adamw = using_adamw
            # param_others = [{'params': other_params},]
            self._other_opt = optim.AdamW(
                other_params,
                eps=adamw_eps,
                betas=(adamw_beta1, adamw_beta2),
                lr=lr,
                weight_decay=weight_decay,
            )
            self.using_constant_adamw_lr = using_constant_adamw_lr
        else:
            self._other_opt = optim.SGD(
                other_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )

        # self.param_groups = self._other_opt.param_groups
        self.param_groups = self._sngd_opt.param_groups
        self.warmup_factor = warmup_factor
        print('max lr_cov', self.lr_cov)
        print('warmup_factor:', warmup_factor)

    def zero_grad(self, set_to_none: bool = True):
        self._other_opt.zero_grad(set_to_none)
        self._sngd_opt.zero_grad(set_to_none)

        if self._sngd_opt.steps <= 50*self.warmup_factor:
            step_lr_cov = self.lr_cov/10000.0
        elif self._sngd_opt.steps <= 100*self.warmup_factor:
            step_lr_cov = self.lr_cov/100.0
        elif self._sngd_opt.steps <= 150*self.warmup_factor:
            step_lr_cov = self.lr_cov/10.0
        elif self._sngd_opt.steps <= 200*self.warmup_factor:
            step_lr_cov = self.lr_cov/10.0
        else:
            step_lr_cov = self.lr_cov

        for group in self._sngd_opt.param_groups:
            group["lr_cov"] = step_lr_cov

        if self._sngd_opt.steps < 20 * self._sngd_opt.param_groups[0]["T"]:
            step_weight_decay = 0.0
        else:
            step_weight_decay = self.weight_decay

        for group in self._sngd_opt.param_groups:
            group["weight_decay"] = step_weight_decay

        # for group in self._other_opt.param_groups:
            # group["weight_decay"] = step_weight_decay

    def step(self, closure=None):
        assert closure is None
        self._other_opt.step()
        self._sngd_opt.step()


class My_LRScheduler:
    def __init__(self, optimizer, scheduler_class, **kwargs):
        if isinstance(optimizer, My_SNGD):
            sngd_lr = scheduler_class(optimizer._sngd_opt, **kwargs)

            if optimizer.using_adamw and optimizer.using_constant_adamw_lr:
                self._lr = [sngd_lr,] #use a constant learning rate for adamw
            else:
                other_lr = scheduler_class(optimizer._other_opt, **kwargs) #use learning rate scheduling for adamw
                self._lr = [sngd_lr, other_lr]


        elif isinstance(optimizer, My_KFAC):
            self._lr = [scheduler_class(optimizer._opt, **kwargs)]
        else:
            self._lr = [
                scheduler_class(optimizer, **kwargs),
            ]

    def _get_lr(self):
        ss = 0.0
        for idx, lr in enumerate(self._lr):
            ss = lr._get_lr()
        return ss

    def _initial_step(self):
        for lr in enumerate(self._lr):
            lr._initial_step()

    def get_last_lr(self):
        ss = 0.0
        for idx, lr in enumerate(self._lr):
            ss = lr.get_last_lr()
        return ss

    def step(self, epoch=None, metric=None):
        for idx, lr in enumerate(self._lr):
            if metric is not None:
                lr.step(epoch, metric)
            else:
                lr.step(epoch)

    def get_cycle_length(self):
        ss = 0
        for idx, lr in enumerate(self._lr):
            ss = lr.get_cycle_length()
        return ss


    def step_update(self, num_updates, metric=None):
        for idx, lr in enumerate(self._lr):
            lr.step_update(num_updates, metric)


class My_Scaler:
    def __init__(self, optimizer, scaler_class):
        if isinstance(optimizer, My_SNGD):
            self._scaler = [scaler_class(), scaler_class()]
        else:
            self._scaler = [scaler_class()]

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        **kwargs,
    ):
        assert clip_grad is None

        if isinstance(optimizer, My_SNGD):
            loss2 = self._scaler[0].scale(loss)
            # grad_scale = self._scaler[0].get_scale()
            # optimizer._sngd_opt.grad_scale = grad_scale
            loss2.backward()
            self._scaler[1].scale(torch.zeros_like(loss))

            self._scaler[0].step(optimizer._sngd_opt)
            self._scaler[1].step(optimizer._other_opt)

            for sl in self._scaler:
                sl.update()

        elif isinstance(optimizer, My_KFAC):
            loss2 = self._scaler[0].scale(loss)
            grad_scale = self._scaler[0].get_scale()
            optimizer._opt.grad_scale = grad_scale
            loss2.backward()
            self._scaler[0].unscale_(optimizer._opt)
            self._scaler[0].step(optimizer._opt)
            self._scaler[0].update()

        else:
            self._scaler[0](loss, optimizer, clip_grad=clip_grad, **kwargs)

    def state_dict(self):
        my_dict = {}
        if len(self._scaler)>1:
            my_dict['sngd_scaler'] = self._scaler[0].state_dict()
            my_dict['other_scaler'] = self._scaler[1].state_dict()
        else:
            my_dict['other_scaler'] = self._scaler[0].state_dict()
        return my_dict

    def load_state_dict(self, state_dict):
        assert len(state_dict) == len(self._scaler)
        if len(self._scaler)>1:
            self._scaler[0].load_state_dict(my_dict['sngd_scaler'])
            self._scaler[1].load_state_dict(my_dict['other_scaler'])
        else:
            self._scaler[0].load_state_dict(my_dict['other_scaler'])
