import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../baselines/KFAC-Pytorch/'))
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
        # use_eign = True,
        use_eign = False,
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
                         )
        print('max lr_cov:', self.lr_cov)

    def zero_grad(self, set_to_none: bool = True):
        self._opt.zero_grad(set_to_none)

        if self._opt.steps <= 500:
            step_lr_cov = 1e-6
        elif self._opt.steps <= 1000:
            step_lr_cov = 1e-5
        elif self._opt.steps <= 1500:
            step_lr_cov = 1e-4
        elif self._opt.steps <= 2000:
            step_lr_cov = 1e-3
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
