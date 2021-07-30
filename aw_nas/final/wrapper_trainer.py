import os
import numpy as np
import torch
from torch import nn

from aw_nas import utils
from aw_nas.final.cnn_trainer import CNNFinalTrainer #, _warmup_update_lr
from aw_nas.utils.common_utils import nullcontext
from aw_nas.utils.exception import expect


def _warmup_update_lr(optimizer, epoch, init_lr, warmup_epochs, warmup_ratio=0.0):
    k = (1 - epoch / warmup_epochs) * (1 - warmup_ratio)
    lr = init_lr * (1 - k)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class WrapperFinalTrainer(CNNFinalTrainer):  # pylint: disable=too-many-instance-attributes
    NAME = "wrapper_final_trainer"

    def __init__(
            self,
            model,
            dataset,
            device,
            gpus,
            objective,  # pylint: disable=dangerous-default-value
            multiprocess=False,
            epochs=600,
            batch_size=96,
            optimizer_type="SGD",
            optimizer_kwargs=None,
            learning_rate=0.025,
            momentum=0.9,
            freeze_base_net=False,
            base_net_lr=1e-4,
            warmup_epochs=0,
            warmup_steps=0,
            warmup_ratio=0.,
            optimizer_scheduler={
                "type": "CosineAnnealingLR",
                "T_max": 600,
                "eta_min": 0.001
            },
            weight_decay=3e-4,
            no_bias_decay=False,
            grad_clip=5.0,
            auxiliary_head=False,
            auxiliary_weight=0.4,
            add_regularization=False,
            save_as_state_dict=False,
            workers_per_queue=2,
            eval_every=10,
            eval_batch_size=1,
            eval_no_grad=True,
            eval_dir=None,
            calib_bn_setup=False,
            seed=None,
            schedule_cfg=None):

        self.freeze_base_net = freeze_base_net
        self.base_net_lr = base_net_lr
        super(WrapperFinalTrainer,
              self).__init__(model, dataset, device, gpus, objective,
                             multiprocess, epochs, batch_size, optimizer_type,
                             optimizer_kwargs, learning_rate, momentum,
                             warmup_epochs, optimizer_scheduler, weight_decay,
                             no_bias_decay, grad_clip, auxiliary_head,
                             auxiliary_weight, add_regularization,
                             save_as_state_dict, workers_per_queue,
                             eval_no_grad, eval_every, eval_batch_size, calib_bn_setup, seed, schedule_cfg)

        self._criterion = self.objective._criterion
        self._acc_func = self.objective.get_acc
        self._perf_func = self.objective.get_perfs

        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio


    def _init_optimizer(self):
        optim_cls = getattr(torch.optim, self.optimizer_type)
        optim_kwargs = {
            "lr": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay
        }
        backbone = self.model.backbone
        neck = self.model.neck
        head = self.model.head
        if not self.freeze_base_net:
            params = [{"params": self.model.parameters()}]
        else:
            params = [{
                "params": backbone.parameters(),
                "lr": self.base_net_lr
            }, {
                "params": head.parameters()
            }]
            if neck is not None:
                params += [{"params": neck.parameters()}]
        optim_kwargs.update(self.optimizer_kwargs or {})
        optimizer = optim_cls(params, **optim_kwargs)

        return optimizer

    def evaluate_split(self, split):
        # if len(self.gpus) >= 2:
        #     self._forward_once_for_flops(self.model)
        rank = os.environ.get("LOCAL_RANK")
        if rank is not None and rank != '0':
            return 0., 0.
        assert split in {"train", "test"}
        if split == "test":
            queue = self.valid_queue
        else:
            queue = self.train_queue
        acc, obj, perfs = self.infer_epoch(queue, self.parallel_model,
                                           self._criterion, self.device)
        self.logger.info(
            "acc %f ; obj %f ; performance: %s", acc, obj,
            "; ".join(["{}: {:.3f}".format(n, v) for n, v in perfs.items()]))
        return acc, obj

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def train_epoch(self, train_queue, model, criterion, optimizer, device,
                    epoch):
        expect(self._is_setup, "trainer.setup should be called first")
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        losses_obj = utils.OrderedStats()
        model.train()
        self.objective.set_mode("train")

        for step, (inputs, targets) in enumerate(train_queue):
            cur_step = step + len(train_queue) * (epoch - 1)
            if 0 <= cur_step <= self.warmup_steps:
                lr = _warmup_update_lr(optimizer, cur_step,
                                       self.learning_rate, self.warmup_steps, self.warmup_ratio)
                if cur_step % self.report_every == 0:
                    self.logger.info("Step {} LR: {}".format(step, lr))

            inputs = inputs.to(self.device)
            optimizer.zero_grad()
            predictions = model.forward(inputs)
            losses = criterion(inputs, predictions, targets, model)
            loss = sum(losses.values())
            loss.backward()

            #for p in optimizer.param_groups[0]['params']:
            #    if p.grad is not None:
            #        print(p.grad.sum())

            if isinstance(self.grad_clip, (int, float)) and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            prec1, prec5 = self._acc_func(inputs, predictions, targets, model)
            n = inputs.size(0)
            losses_obj.update(losses)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_every == 0:
                self.logger.info("train %03d %.2f%%; %.2f%%; %s",
                                 step, top1.avg, top5.avg, "; ".join(
                                     ["{}: {:.3f}".format(perf_n, v)
                                      for perf_n, v in losses_obj.avgs().items()]))
        return top1.avg, sum(losses_obj.avgs().values())

    def infer_epoch(self, valid_queue, model, criterion, device):
        expect(self._is_setup, "trainer.setup should be called first")
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        losses_obj = utils.OrderedStats()
        all_perfs = []
        model.eval()
        self.objective.set_mode("eval")

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for step, (inputs, targets) in enumerate(valid_queue):
                inputs = inputs.to(device)
                #targets = targets.to(device)

                predictions = model(inputs)
                losses = criterion(inputs, predictions, targets, model)
                prec1, prec5 = self._acc_func(inputs, predictions, targets,
                                              model)
                perfs = self._perf_func(inputs, predictions, targets, model)
                all_perfs.append(perfs)
                n = inputs.size(0)
                objective_perfs.update(dict(zip(self._perf_names, perfs)), n=n)
                losses_obj.update(losses, n=n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.report_every == 0:
                    self.logger.info(
                        "valid %03d; %s, %s", step, ";".join(
                            ["%s: %.3f" % l for l in losses.items()]),
                        "; ".join([
                            "{}: {:.3f}".format(perf_n, v) for perf_n, v in
                            list(objective_perfs.avgs().items())]))
        all_perfs = list(zip(*all_perfs))
        obj_perfs = {
            k: self.objective.aggregate_fn(k, False)(v)
            for k, v in zip(self._perf_names, all_perfs)
        }
        return top1.avg, sum(losses_obj.avgs().values()), obj_perfs
