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

    def __init__(self, model, dataset, device, gpus, #pylint: disable=dangerous-default-value
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
        self.dataset = dataset
        self.device = device
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        self.auxiliary_head = auxiliary_head
        self.auxiliary_weight = auxiliary_weight

        if len(gpus) >= 2:
            self.model = torch.nn.DataParallel(self.model, gpus).to(device)

        self.logger.info("param size = %f MB (%d elements)",
                         utils.count_parameters_in_MB(self.model),
                         utils.count_parameters(self.model))

        self._criterion = nn.CrossEntropyLoss().to(self.device)

        group_weight = []
        group_bias = []
        for name, param in model.named_parameters():
            if 'bias' in name:
                group_bias.append(param)
            else:
                group_weight.append(param)
        assert len(list(model.parameters())) == len(group_weight) + len(group_bias)
        self.optimizer = torch.optim.SGD([{'params': group_weight},
                                          {'params': group_bias,
                                           'weight_decay': 0 if no_bias_decay else weight_decay}],
                                         lr=learning_rate,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        _splits = self.dataset.splits()

        self.train_queue = torch.utils.data.DataLoader(
            _splits["train"], batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        self.valid_queue = torch.utils.data.DataLoader(
            _splits["test"], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

        self.scheduler = self._init_scheduler(self.optimizer, optimizer_scheduler)

        self.save_every = None
        self.report_every = None
        self.train_dir = None

    def setup(self, load=None, save_every=None, train_dir=None, report_every=50):
        if load is not None:
            self.load(load)
        self.save_every = save_every
        self.train_dir = train_dir
        self.report_every = report_every

        if self.save_every is not None:
            expect(self.train_dir is not None)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info("Saved checkpoint to %s", path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.logger.info("Loaded checkpoint from %s", path)

    def train(self):
        for epoch in range(1, self.epochs+1):
            self.on_epoch_start(epoch)
            if epoch < self.warmup_epochs:
                _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
            else:
                self.scheduler.step()
            self.logger.info('epoch %d lr %e', epoch, self.scheduler.get_lr()[0])

            train_acc, train_obj = self.train_epoch(self.train_queue, self.model, self._criterion,
                                                    self.optimizer, self.device, epoch)
            self.logger.info('train_acc %f ; train_obj %f', train_acc, train_obj)

            valid_acc, valid_obj = self.infer_epoch(self.valid_queue, self.model,
                                                    self._criterion, self.device)
            self.logger.info('valid_acc %f ; valid_obj %f', valid_acc, valid_obj)

            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch) + ".pt")
                self.save(path)
            self.on_epoch_end(epoch)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @staticmethod
    def _init_scheduler(optimizer, cfg):
        if cfg:
            cfg = {k:v for k, v in six.iteritems(cfg)}
            sch_cls = utils.get_scheduler_cls(cfg.pop("type"))
            return sch_cls(optimizer, **cfg)
        return None


    def train_epoch(self, train_queue, model, criterion, optimizer, device, epoch):
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
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.eval()

        for step, (inputs, target) in enumerate(valid_queue):
            inputs = inputs.to(device)
            target = target.to(device)
            with torch.no_grad():
                logits = model(inputs)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.report_every == 0:
                    self.logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


    def on_epoch_start(self, epoch):
        super(CNNFinalTrainer, self).on_epoch_start(epoch)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.on_epoch_start(epoch)
        else:
            self.model.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        super(CNNFinalTrainer, self).on_epoch_end(epoch)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.on_epoch_end(epoch)
        else:
            self.model.on_epoch_end(epoch)
