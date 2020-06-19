# -*- coding: utf-8 -*-

import os
import six

import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from aw_nas import utils
from aw_nas.final.base import FinalTrainer
from aw_nas.utils.common_utils import nullcontext
from aw_nas.utils.exception import expect
from aw_nas.utils import DataParallel, DistributedDataParallel

from aw_nas.utils.torch_utils import calib_bn


try:
    from aw_nas.utils.SynchronizedBatchNormPyTorch.sync_batchnorm import (
        convert_model as convert_sync_bn,
    )
except ImportError:
    convert_sync_bn = lambda m: m


def _warmup_update_lr(optimizer, epoch, init_lr, warmup_epochs):
    """
    update learning rate of optimizers
    """
    lr = init_lr * epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class OFAFinalTrainer(FinalTrainer): #pylint: disable=too-many-instance-attributes
    NAME = "ofa_final_trainer"

    def __init__(self, model, dataset, device, gpus, objective,#pylint: disable=dangerous-default-value
                 state_dict_path=None,
                 multiprocess=False, use_gn=False,
                 epochs=600, batch_size=96,
                 optimizer_type="SGD", optimizer_kwargs=None,
                 learning_rate=0.025, momentum=0.9,
                 warmup_epochs=0,
                 optimizer_scheduler={
                     "type": "CosineAnnealingLR",
                     "T_max": 600,
                     "eta_min": 0.001
                 },
                 weight_decay=3e-4, no_bias_decay=False,
                 grad_clip=5.0,
                 auxiliary_head=False, auxiliary_weight=0.4,
                 add_regularization=False,
                 save_as_state_dict=False,
                 workers_per_queue=2,
                 eval_no_grad=True,
                 calib_bn_setup=True,
                 schedule_cfg=None):
        super(OFAFinalTrainer, self).__init__(schedule_cfg)

        self.model = model
        self.parallel_model = None
        self.dataset = dataset
        self.device = device
        self.gpus = gpus
        self.objective = objective
        self._perf_func = self.objective.get_perfs
        self._perf_names = self.objective.perf_names()
        self._obj_loss = self.objective.get_loss
        self.multiprocess = multiprocess
        self.use_gn = use_gn

        self.state_dict_path = state_dict_path

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.auxiliary_head = auxiliary_head
        self.auxiliary_weight = auxiliary_weight
        self.add_regularization = add_regularization
        self.save_as_state_dict = save_as_state_dict
        self.eval_no_grad = eval_no_grad
        self.calib_bn_setup = calib_bn_setup

        # for optimizer
        self.weight_decay = weight_decay
        self.no_bias_decay = no_bias_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_scheduler_cfg = optimizer_scheduler

        self._criterion = nn.CrossEntropyLoss().to(self.device)

        _splits = self.dataset.splits()

        train_kwargs = getattr(_splits["train"], "kwargs", {})
        test_kwargs = getattr(_splits["test"], "kwargs", train_kwargs)
        if self.multiprocess:
            self.train_queue = torch.utils.data.DataLoader(
                _splits["train"], batch_size=batch_size, pin_memory=True,
                num_workers=workers_per_queue,
                sampler=DistributedSampler(_splits["train"], shuffle=True), **train_kwargs)
            self.valid_queue = torch.utils.data.DataLoader(
                _splits["test"], batch_size=batch_size, pin_memory=True,
                num_workers=workers_per_queue, shuffle=False, **test_kwargs)
        else:
            self.train_queue = torch.utils.data.DataLoader(
                _splits["train"], batch_size=batch_size, pin_memory=True,
                num_workers=workers_per_queue, shuffle=True, **train_kwargs)
            self.valid_queue = torch.utils.data.DataLoader(
                _splits["test"], batch_size=batch_size, pin_memory=True,
                num_workers=workers_per_queue, shuffle=False, **test_kwargs)

        if self.model is not None:
            self.optimizer = self._init_optimizer()
            self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)

        # states of the trainer
        self.last_epoch = 0
        self.epoch = 0
        self.save_every = None
        self.report_every = None
        self.train_dir = None
        self._is_setup = False

        if self.calib_bn_setup:
            self.model = calib_bn(self.model, self.train_queue)

    def setup(self, load=None, load_state_dict=None,
              save_every=None, train_dir=None, report_every=50):
        expect(not (load is not None and load_state_dict is not None),
               "`load` and `load_state_dict` cannot be passed simultaneously.")
        if load is not None:
            self.load(load)
        else:
            assert self.model is not None
            if load_state_dict is not None:
                self._load_state_dict(load_state_dict)

            self.logger.info("param size = %f M",
                             utils.count_parameters(self.model)/1.e6)
            self._parallelize()

        self.save_every = save_every
        self.train_dir = train_dir
        self.report_every = report_every

        expect(self.save_every is None or self.train_dir is not None,
               "when `save_every` is not None, make sure `train_dir` is not None")

        self._is_setup = True


    def save(self, path):
        rank = (os.environ.get("LOCAL_RANK"))
        if rank is not None and rank != '0':
            return
        path = utils.makedir(path)
        if self.save_as_state_dict:
            torch.save(self.model.state_dict(), os.path.join(path, "model_state.pt"))
        else:
            # save the model directly instead of the state_dict,
            # so that it can be loaded and run directly, without specificy configuration
            torch.save(self.model, os.path.join(path, "model.pt"))
        torch.save({
            "epoch": self.epoch,
            "optimizer":self.optimizer.state_dict()
        }, os.path.join(path, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        self.logger.info("Saved checkpoint to %s", path)

    def load(self, path):
        # load the model
        m_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        if not os.path.exists(m_path):
            m_path = os.path.join(path, "model_state.pt")
            self._load_state_dict(m_path)
        else:
            res = self.model.load_state_dict(torch.load(
                m_path, map_location=torch.device("cpu")), strict=False)

        self.model.to(self.device)
        self._parallelize()
        log_strs = ["model from {}".format(m_path)]

        # init/load the optimzier
        self.optimizer = self._init_optimizer()
        o_path = os.path.join(path, "optimizer.pt") if os.path.isdir(path) else None
        if o_path and os.path.exists(o_path):
            checkpoint = torch.load(o_path, map_location=torch.device("cpu"))
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log_strs.append("optimizer from {}".format(o_path))
            self.last_epoch = checkpoint["epoch"]

        # init/load the scheduler
        self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)
        if self.scheduler is not None:
            s_path = os.path.join(path, "scheduler.pt") if os.path.isdir(path) else None
            if s_path and os.path.exists(s_path):
                self.scheduler.load_state_dict(torch.load(s_path, map_location=torch.device("cpu")))
                log_strs.append("scheduler from {}".format(s_path))

        self.logger.info("param size = %f M",
                         utils.count_parameters(self.model)/1.e6)
        self.logger.info("Loaded checkpoint from %s: %s", path, ", ".join(log_strs))
        self.logger.info("Last epoch: %d", self.last_epoch)

    def train(self):
        self._forward_once_for_flops(self.model)
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.epoch = epoch
            self.on_epoch_start(epoch)

            if epoch < self.warmup_epochs:
                _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
            else:
                if self.scheduler is not None:
                    self.scheduler.step()
            self.logger.info("epoch %d lr %e", epoch, self.optimizer.param_groups[0]["lr"])

            train_acc, train_obj = self.train_epoch(self.train_queue, self.parallel_model,
                                                    self._criterion, self.optimizer,
                                                    self.device, epoch)
            self.logger.info("train_acc %f ; train_obj %f", train_acc, train_obj)

            valid_acc, valid_obj, valid_perfs = self.infer_epoch(self.valid_queue,
                                                                 self.parallel_model,
                                                                 self._criterion, self.device)
            self.logger.info("valid_acc %f ; valid_obj %f ; valid performances: %s",
                             valid_acc, valid_obj,
                             "; ".join(
                                 ["{}: {:.3f}".format(n, v) for n, v in valid_perfs.items()]))

            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch))
                self.save(path)
            self.on_epoch_end(epoch)

        self.save(os.path.join(self.train_dir, "final"))


    def evaluate_split(self, split):
        if len(self.gpus) >= 2:
            self._forward_once_for_flops(self.model)
        assert split in {"train", "test"}
        if split == "test":
            queue = self.valid_queue
        else:
            queue = self.train_queue
        acc, obj, perfs = self.infer_epoch(queue, self.parallel_model,
                                           self._criterion, self.device)
        self.logger.info("acc %f ; obj %f ; performance: %s", acc, obj,
                         "; ".join(
                             ["{}: {:.3f}".format(n, v) for n, v in perfs.items()]))
        return acc, obj

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _load_state_dict(self, path):
        # load state dict
        # TODO: while training, we save sorted channels instead of sorting every step.
        # How to pick channels while loading?
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        extra_keys = set(checkpoint.keys()).difference(set(self.model.state_dict().keys()))
        if extra_keys:
            self.logger.error("%d extra keys in checkpoint! "
                              "Make sure the genotype match", len(extra_keys))
        mismatch = self.model.load_state_dict(checkpoint, strict=False)
        self.logger.info(mismatch)

    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self.model).to(self.device)
            self.parallel_model = DistributedDataParallel(net, self.gpus, find_unused_parameters=True)
        elif len(self.gpus) >= 2:
            self.parallel_model = DataParallel(self.model, self.gpus).to(self.device)
        else:
            self.parallel_model = self.model

    def _init_optimizer(self):
        group_weight = []
        group_bias = []
        for name, param in self.model.named_parameters():
            if "bias" in name:
                group_bias.append(param)
            else:
                group_weight.append(param)
        assert len(list(self.model.parameters())) == len(group_weight) + len(group_bias)
        optim_cls = getattr(torch.optim, self.optimizer_type)
        optim_kwargs = {
            "lr": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay
        }
        optim_kwargs.update(self.optimizer_kwargs or {})
        optimizer = optim_cls(
            [{"params": group_weight},
             {"params": group_bias,
              "weight_decay": 0 if self.no_bias_decay else self.weight_decay}],
            **optim_kwargs)

        return optimizer

    @staticmethod
    def _init_scheduler(optimizer, cfg):
        if cfg:
            cfg = {k:v for k, v in six.iteritems(cfg)}
            sch_cls = utils.get_scheduler_cls(cfg.pop("type"))
            return sch_cls(optimizer, **cfg)
        return None


    def train_epoch(self, train_queue, model, criterion, optimizer, device, epoch):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.train()

        for step, (inputs, target) in enumerate(train_queue):
            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            if self.auxiliary_head: # assume model return two logits in train mode
                logits, logits_aux = model(inputs)
                loss = self._obj_loss(inputs, logits, target, model,
                                      add_evaluator_regularization=self.add_regularization)
                loss_aux = criterion(logits_aux, target)
                loss += self.auxiliary_weight * loss_aux
            else:
                logits = model(inputs)
                loss = self._obj_loss(inputs, logits, target, model,
                                      add_evaluator_regularization=self.add_regularization)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_every == 0:
                self.logger.info("train %03d %.3f; %.2f%%; %.2f%%",
                                 step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def infer_epoch(self, valid_queue, model, criterion, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        model.eval()

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for step, (inputs, target) in enumerate(valid_queue):
                inputs = inputs.to(device)
                target = target.to(device)

                logits = model(inputs)
                loss = criterion(logits, target)
                perfs = self._perf_func(inputs, logits, target, model)
                objective_perfs.update(dict(zip(self._perf_names, perfs)))
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.report_every == 0:
                    self.logger.info("valid %03d %e %f %f %s", step, objs.avg, top1.avg, top5.avg,
                                     "; ".join(["{}: {:.3f}".format(perf_n, v) \
                                                for perf_n, v in objective_perfs.avgs().items()]))

        return top1.avg, objs.avg, objective_perfs.avgs()

    def on_epoch_start(self, epoch):
        super(OFAFinalTrainer, self).on_epoch_start(epoch)
        self.model.on_epoch_start(epoch)
        self.objective.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        super(OFAFinalTrainer, self).on_epoch_end(epoch)
        self.model.on_epoch_end(epoch)
        self.objective.on_epoch_end(epoch)

    def _forward_once_for_flops(self, model):
        # forward the model once to get the flops calculated
        self.logger.info("Training parallel: Forward one batch for the flops information")
        inputs, _ = next(iter(self.train_queue))
        model(inputs.to(self.device))
