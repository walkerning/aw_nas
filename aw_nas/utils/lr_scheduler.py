import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler

__all__ = ["CosineWithRestarts", "get_scheduler_cls"]

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
                 factor=1.):
        assert t_0 > 0
        assert eta_min >= 0
        # if t_max == 1 and factor == 1:
        #     logger.warning("Cosine annealing scheduler will have no effect on the learning "
        #                    "rate since T_max = 1 and factor = 1.")

        self.t_max = t_0
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = t_0
        self._initialized = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs
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
            for lr in self.base_lrs
        ]
        if self._cycle_counter != 0 and self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.t_max)
            self._last_restart = step
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
        return [max(self.eta_min,
                    base_lr * self.gamma ** ((self.last_epoch - self.start_epoch) // self.every))
                for base_lr in self.base_lrs]

def get_scheduler_cls(type_):
    if type_ in {"CosineWithRestarts", "ExpDecay"}:
        return globals()[type_]
    return getattr(lr_scheduler, type_)
