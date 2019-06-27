# -*- coding: utf-8 -*-

import os
import six

import torch
from torch import nn

from aw_nas import utils
from aw_nas.final.base import FinalTrainer
from aw_nas.utils.exception import expect

def _warmup_update_lr(optimizer, epoch, init_lr, warmup_epochs):
    """
    update learning rate of optimizers
    """
    lr = init_lr * epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CNNFinalTrainer(FinalTrainer):
    NAME = "cnn_trainer"

    def __init__(self, model, dataset, device, gpus, objective,#pylint: disable=dangerous-default-value
                 epochs=600, batch_size=96,
                 learning_rate=0.025, momentum=0.9,
                 warmup_epochs=0,
                 optimizer_scheduler={
                     "type": "CosineAnnealingLR",
                     "eta_min": 0.001
                 },
                 weight_decay=3e-4, no_bias_decay=False,
                 grad_clip=5.0,
                 auxiliary_head=False, auxiliary_weight=0.4,
                 schedule_cfg=None):
        super(CNNFinalTrainer, self).__init__(schedule_cfg)

        self.model = model
        self.parallel_model = None
        self.dataset = dataset
        self.device = device
        self.gpus = gpus
        self.objective = objective
        self._perf_func = self.objective.get_perfs
        self._perf_names = self.objective.perf_names()

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.auxiliary_head = auxiliary_head
        self.auxiliary_weight = auxiliary_weight

        # for optimizer
        self.weight_decay = weight_decay
        self.no_bias_decay = no_bias_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_scheduler_cfg = optimizer_scheduler

        self._criterion = nn.CrossEntropyLoss().to(self.device)

        _splits = self.dataset.splits()

        self.train_queue = torch.utils.data.DataLoader(
            _splits["train"], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        self.valid_queue = torch.utils.data.DataLoader(
            _splits["test"], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)

        # states of the trainer
        self.last_epoch = 0
        self.epoch = 0
        self.save_every = None
        self.report_every = None
        self.train_dir = None
        self._is_setup = False

    def setup(self, load=None, save_every=None, train_dir=None, report_every=50):
        if load is not None:
            self.load(load)
        else:
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
        path = utils.makedir(path)
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
        self.model = torch.load(m_path, map_location=torch.device("cpu"))
        self.model.to(self.device)
        self._parallelize()
        log_strs = ["model from {}".format(m_path)]

        # init/load the optimzier
        self.optimizer = self._init_optimizer()
        o_path = os.path.join(path, "optimizer.pt") if os.path.isdir(path) else None
        if o_path and os.path.exists(o_path):
            checkpoint = torch.load(o_path)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log_strs.append("optimizer from {}".format(o_path))
            self.last_epoch = checkpoint["epoch"]

        # init/load the scheduler
        self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)
        if self.scheduler is not None:
            s_path = os.path.join(path, "scheduler.pt") if os.path.isdir(path) else None
            if s_path and os.path.exists(s_path):
                self.scheduler.load_state_dict(torch.load(s_path))
                log_strs.append("scheduler from {}".format(s_path))

        self.logger.info("param size = %f M",
                         utils.count_parameters(self.model)/1.e6)
        self.logger.info("Loaded checkpoint from %s: %s", path, ", ".join(log_strs))
        self.logger.info("Last epoch: %d", self.last_epoch)

    def train(self):
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.epoch = epoch
            self.on_epoch_start(epoch)

            if epoch < self.warmup_epochs:
                _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
            else:
                self.scheduler.step()
            self.logger.info('epoch %d lr %e', epoch, self.scheduler.get_lr()[0])

            train_acc, train_obj = self.train_epoch(self.train_queue, self.parallel_model,
                                                    self._criterion, self.optimizer,
                                                    self.device, epoch)
            self.logger.info('train_acc %f ; train_obj %f', train_acc, train_obj)

            valid_acc, valid_obj, valid_perfs = self.infer_epoch(self.valid_queue, self.parallel_model,
                                                    self._criterion, self.device)
            self.logger.info('valid_acc %f ; valid_obj %f ; valid performances: %s', valid_acc, valid_obj,
                             "; ".join(
                                 ["{}: {:.3f}".format(n, v) for n, v in valid_perfs.items()]))

            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch))
                self.save(path)
            self.on_epoch_end(epoch)

        self.save(os.path.join(self.train_dir, "final"))

    def evaluate_split(self, split):
        assert split in {"train", "test"}
        if split == "test":
            queue = self.valid_queue
        else:
            queue = self.train_queue
        acc, obj, perfs = self.infer_epoch(queue, self.parallel_model,
                                           self._criterion, self.device)
        self.logger.info('acc %f ; obj %f ; performance: %s', acc, obj,
                         "; ".join(
                             ["{}: {:.3f}".format(n, v) for n, v in perfs.items()]))
        return acc, obj

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def _parallelize(self):
        if len(self.gpus) >= 2:
            self.parallel_model = torch.nn.DataParallel(self.model, self.gpus).to(self.device)
        else:
            self.parallel_model = self.model

    def _init_optimizer(self):
        group_weight = []
        group_bias = []
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                group_bias.append(param)
            else:
                group_weight.append(param)
        assert len(list(self.model.parameters())) == len(group_weight) + len(group_bias)
        optimizer = torch.optim.SGD(
            [{'params': group_weight},
             {'params': group_bias,
              'weight_decay': 0 if self.no_bias_decay else self.weight_decay}],
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay)
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
                loss = criterion(logits, target)
                loss_aux = criterion(logits_aux, target)
                loss += self.auxiliary_weight * loss_aux
            else:
                logits = model(inputs)
                loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_every == 0:
                self.logger.info('train %03d %.3f; %.2f%%; %.2f%%',
                                 step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def infer_epoch(self, valid_queue, model, criterion, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        model.eval()

        for step, (inputs, target) in enumerate(valid_queue):
            inputs = inputs.to(device)
            target = target.to(device)
            with torch.no_grad():
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
                    self.logger.info('valid %03d %e %f %f %s', step, objs.avg, top1.avg, top5.avg,
                                     "; ".join(["{}: {:.3f}".format(n, v) \
                                                for n, v in objective_perfs.avgs().items()]))

        return top1.avg, objs.avg, objective_perfs.avgs()


    def on_epoch_start(self, epoch):
        super(CNNFinalTrainer, self).on_epoch_start(epoch)
        self.model.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        super(CNNFinalTrainer, self).on_epoch_end(epoch)
        self.model.on_epoch_end(epoch)
