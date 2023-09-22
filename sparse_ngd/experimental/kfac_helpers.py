import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../wang2019kfac/'))
from optimizers import KFACOptimizer

import torch
import torch.optim as optim
from torch.nn import Conv2d, Linear
from torch.optim import Optimizer


class My_KFAC(Optimizer):
    def __init__(
        self,
        model,
        lr,
        momentum,
        damping,
        weight_decay,
        T,
        lr_cov,
        use_eign = False,
        warmup_factor = 10,
        using_adamw = False,
        cast_dtype = torch.float32,
        adamw_eps = 1e-8,
        adamw_beta1 = 0.9,
        adamw_beta2 = 0.999,
        using_constant_adamw_lr = False,
    ):
        if use_eign:
            print('using eign')
        else:
            print('using inv')
        print('my kfac helper')
        self.weight_decay = weight_decay
        self.lr_cov = lr_cov

        self._opt = KFACOptimizer(model,
                         lr=lr,
                         momentum=momentum,
                         stat_decay=1.0 - lr_cov,
                         damping=damping,
                         weight_decay=weight_decay,
                         TCov=T,
                         TInv=T,
                         use_eign = use_eign,
                         using_adamw = using_adamw,
                         cast_dtype=cast_dtype,
                         adamw_eps = adamw_eps,
                         adamw_beta1 = adamw_beta1,
                         adamw_beta2 = adamw_beta2,
                         using_constant_adamw_lr = using_constant_adamw_lr,
                         )
        self.warmup_factor = warmup_factor
        self.param_groups = self._opt.param_groups
        print('max lr_cov:', self.lr_cov)
        print('warmup_factor:', warmup_factor)

    def zero_grad(self, set_to_none: bool = True):
        self._opt.zero_grad(set_to_none)

        if self._opt.steps <= 50*self.warmup_factor:
            step_lr_cov = self.lr_cov/10000.0
        elif self._opt.steps <= 100*self.warmup_factor:
            step_lr_cov = self.lr_cov/1000.0
        elif self._opt.steps <= 150*self.warmup_factor:
            step_lr_cov = self.lr_cov/100.0
        elif self._opt.steps <= 200*self.warmup_factor:
            step_lr_cov = self.lr_cov/10.0
        else:
            step_lr_cov = self.lr_cov

        self._opt.stat_decay =  1.0 - step_lr_cov

        if self._opt.steps < 20 * self._opt.TCov:
            step_weight_decay = 0.0
        else:
            step_weight_decay = self.weight_decay

        for group in self._opt.param_groups:
            group["weight_decay"] = step_weight_decay

    def set_cast_dtype(self, cast_dtype):
        self._opt.cast_dtype = cast_dtype


    def step(self, closure=None):
        assert closure is None
        self._opt.step()
