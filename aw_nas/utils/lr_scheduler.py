import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler

__all__ = ["CosineWithRestarts", "get_scheduler_cls"]

class WarmupCosineWithRestarts(_LRScheduler):  # pylint: disable=protected-access

    """
    Cosine annealing with restarts.
     This is decribed in the paper https://arxiv.org/abs/1608.03983.
     Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``
     T_0 : ``int``
        The maximum number of iterations within the first cycle.
     eta_min : ``float``, optional (default=0)
        The minimum learning rate.
     last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.
     factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.
     """
    def __init__(self,
                 optimizer,
                 T_0,
                 warmup_epochs=0,
                 eta_min=0.,
                 last_epoch=-1,
                 factor=1.,
                 base_lr_factor=1.):
        self.t_max = T_0
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.factor = factor
        self.base_lr_factor = base_lr_factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = T_0
        self._initialized = False
        self._base_lrs = None
        super(WarmupCosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            self._base_lrs = self.base_lrs
            return self._base_lrs
        step = self.last_epoch
        self._cycle_counter = step - self._last_restart
        lrs = [
            self.eta_min + ((lr - self.eta_min) / 2) * (
                np.cos(
                    np.pi *
                    (self._cycle_counter % self._updated_cycle_len) /
                    self._updated_cycle_len
                ) + 1
            ) if (self._cycle_counter % self._updated_cycle_len) >= self.warmup_epochs 
            else self._base_lrs[0] * (self._cycle_counter % self._updated_cycle_len + 1) / self.warmup_epochs 
            for lr in self._base_lrs
        ]
        if self._cycle_counter != 0 and self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step
            self._base_lrs = [lr * self.base_lr_factor for lr in self._base_lrs]
        return lrs


class CosineWithRestarts(_LRScheduler):  # pylint: disable=protected-access

    """
    Cosine annealing with restarts.
     This is decribed in the paper https://arxiv.org/abs/1608.03983.
     Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``
     t_max : ``int``
        The maximum number of iterations within the first cycle.
     eta_min : ``float``, optional (default=0)
        The minimum learning rate.
     last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.
     factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.
     """
    def __init__(self,
                 optimizer,
                 t_0,
                 eta_min=0.,
                 last_epoch=-1,
                 factor=1.,
                 base_lr_factor=1.):
        self.t_max = t_0
        self.eta_min = eta_min
        self.factor = factor
        self.base_lr_factor = base_lr_factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = t_0
        self._initialized = False
        self._base_lrs = None
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            self._base_lrs = self.base_lrs
            return self._base_lrs
        step = self.last_epoch
        self._cycle_counter = step - self._last_restart
        lrs = [
            self.eta_min + ((lr - self.eta_min) / 2) * (
                np.cos(
                    np.pi *
                    (self._cycle_counter % self._updated_cycle_len) /
                    self._updated_cycle_len
                ) + 1
            )
            for lr in self._base_lrs
        ]
        if self._cycle_counter != 0 and self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step
            self._base_lrs = [lr * self.base_lr_factor for lr in self._base_lrs]
        return lrs

class ExpDecay(_LRScheduler):
    """
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        every (int): Exp decay every this number of epoch. Default:10.
        start_epoch (int): The epoch that the exp decay starts. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        eta_min (float): Minimum lr. Default: None.
    """

    def __init__(self, optimizer, gamma, every=1, start_epoch=0, eta_min=None, last_epoch=-1):
        self.gamma = gamma
        self.every = every
        self.start_epoch = start_epoch
        self.eta_min = eta_min
        super(ExpDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(0 if self.eta_min is None else self.eta_min,
                    base_lr * self.gamma ** ((self.last_epoch - self.start_epoch) // self.every))
                for base_lr in self.base_lrs]

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs, every=1, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.every = every
        self.warmup_steps = int(self.warmup_epochs / self.every)
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [0. for _ in self.base_lrs]
        if self.last_epoch <= self.warmup_epochs:
            return [lr * float(int(self.last_epoch / self.every)) / self.warmup_steps
                    for lr in self.base_lrs]
        last_epoch = self.last_epoch - self.warmup_epochs
        if (last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

def get_scheduler_cls(type_):
    if type_ in {"WarmupCosineAnnealingLR", "WarmupCosineWithRestarts", "CosineWithRestarts", "ExpDecay"}:
        return globals()[type_]
    return getattr(torch.optim.lr_scheduler, type_)
