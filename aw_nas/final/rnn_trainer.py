# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from torch import nn

from aw_nas import utils
from aw_nas.utils.exception import expect
from aw_nas.final.base import FinalTrainer
from aw_nas.utils import DataParallel

class RNNFinalTrainer(FinalTrainer):
    NAME = "rnn_trainer"

    def __init__(self, model, dataset, device, gpus,
                 epochs=600, batch_size=256, eval_batch_size=10,
                 learning_rate=25., momentum=0.,
                 optimizer_scheduler=None,
                 weight_decay=3e-4,
                 bptt_steps=35, reset_hidden_prob=None,
                 rnn_act_reg=0., rnn_slowness_reg=0.,
                 random_bptt=True,
                 valid_decay_window=5,
                 schedule_cfg=None):
        super(RNNFinalTrainer, self).__init__(schedule_cfg)

        self.model = model
        self.parallel_model = None
        self.dataset = dataset
        self.device = device
        self.gpus = gpus

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.bptt_steps = bptt_steps
        self.reset_hidden_prob = reset_hidden_prob
        self.random_bptt = random_bptt
        self.rnn_act_reg = rnn_act_reg
        self.rnn_slowness_reg = rnn_slowness_reg
        self.valid_decay_window = valid_decay_window
        self.weight_decay = weight_decay
        self.optimizer_scheduler_cfg = optimizer_scheduler

        _splits = self.dataset.splits()

        self.train_data = utils.batchify_sentences(_splits["ori_train"], batch_size)
        self.valid_data = utils.batchify_sentences(_splits["ori_valid"], eval_batch_size)
        self.test_data = utils.batchify_sentences(_splits["test"], eval_batch_size)

        self._criterion = nn.CrossEntropyLoss().to(self.device)
        if self.model is not None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                             weight_decay=weight_decay)
            self.scheduler = self._init_scheduler(self.optimizer, optimizer_scheduler)

        # states of the trainer
        self.last_epoch = 0
        self.epoch = 0
        self.report_every = None
        self.save_every = None
        self.train_dir = None
        self._is_setup = False
        self.best_valid_obj = None
        self.valid_objs = []

    def setup(self, load=None, load_state_dict=None,
              save_every=None, train_dir=None, report_every=50):
        assert load_state_dict is None, "Currently not supported and tested."
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
        torch.save(self.model, os.path.join(path, "model.pt"))
        torch.save({
            "epoch": self.epoch,
            "optimizer":self.optimizer.state_dict()
        }, os.path.join(path, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        self.logger.info("Saved checkpoint to %s", path)

    def load(self, path):
        del self.parallel_model
        # load the model
        m_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        # load using cpu
        self.model = torch.load(m_path, map_location=torch.device("cpu"))
        # to device
        self.model.to(self.device)
        # maybe parallelize
        self._parallelize()
        log_strs = ["model from {}".format(m_path)]

        # init/load the optimizer
        o_path = os.path.join(path, "optimizer.pt") if os.path.isdir(path) else None
        if o_path and os.path.exists(o_path):
            # init according to the type of the saved optimizer, and then load
            checkpoint = torch.load(o_path, map_location=torch.device("cpu"))
            optimizer_state = checkpoint["optimizer"]
            if "t0" in optimizer_state["param_groups"][0]:
                # ASGD
                self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.learning_rate,
                                                  t0=0, lambd=0., weight_decay=self.weight_decay)
            else:
                # SGD
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                                 weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(optimizer_state)
            self.last_epoch = self.epoch = checkpoint["epoch"]
            log_strs.append("optimizer from {}".format(o_path))
        else:
            # just init a SGD optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                             weight_decay=self.weight_decay)

        # init/load the scheduler
        self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)
        if self.optimizer_scheduler_cfg is not None:
            s_path = os.path.join(path, "scheduler.pt") if os.path.isdir(path) else None
            if s_path and os.path.exists(s_path):
                self.scheduler.load_state_dict(torch.load(s_path, map_location=torch.device("cpu")))
                log_strs.append("scheduler from {}".format(s_path))

        self.logger.info("param size = %f M",
                         utils.count_parameters(self.model)/1.e6)
        self.logger.info("Loaded checkpoint from %s: %s", path, ", ".join(log_strs))
        self.logger.info("Last epoch: %d", self.last_epoch)

    def train(self):
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.epoch = epoch
            self.on_epoch_start(epoch)
            self._scheduler_step()
            self.logger.info("epoch %d lr %e", epoch, self.optimizer.param_groups[0]["lr"])

            train_obj, train_loss = self.train_epoch(*self.train_data, bptt_steps=self.bptt_steps)
            self.logger.info("train: perp %.3f; bpc %.3f ; loss %.3f ; loss with reg %.3f",
                             np.exp(train_obj), train_obj / np.log(2), train_obj, train_loss)

            self._eval_maybe_save()

            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch))
                self.save(path)
            self.on_epoch_end(epoch)

        self.save(os.path.join(self.train_dir, "final"))

    def _eval_maybe_save(self):
        if "t0" in self.optimizer.param_groups[0]:
            # Averaged SGD
            backup = {}
            for prm in self.model.parameters():
                backup[prm] = prm.data.clone()
                prm.data = self.optimizer.state[prm]["ax"].clone()

            valid_obj = self.evaluate_epoch(*self.valid_data, bptt_steps=self.bptt_steps)
            self.logger.info("valid(averaged): perp %.3f ; bpc %.3f ; loss %.3f",
                             np.exp(valid_obj), valid_obj/np.log(2), valid_obj)

            if self.best_valid_obj is None or valid_obj < self.best_valid_obj:
                path = os.path.join(self.train_dir, "best")
                self.save(path)
                self.logger.info("Epoch %3d: Saving Averaged model to %s", self.epoch, path)
                self.best_valid_obj = valid_obj

            for prm in self.model.parameters():
                prm.data = backup[prm].clone()
        else:
            # SGD
            valid_obj = self.evaluate_epoch(*self.valid_data, bptt_steps=self.bptt_steps)
            self.logger.info("valid: perp %.3f ; bpc %.3f ; loss %.3f",
                             np.exp(valid_obj), valid_obj/np.log(2), valid_obj)

            if self.best_valid_obj is None or valid_obj < self.best_valid_obj:
                path = os.path.join(self.train_dir, "best")
                self.save(path)
                self.logger.info("Epoch %3d: Saving model to %s", self.epoch, path)
                self.best_valid_obj = valid_obj

            window_objs = self.valid_objs[:-self.valid_decay_window]
            if len(window_objs) >= self.valid_decay_window and valid_obj > min(window_objs):
                # check whether the valid performance does not decrease
                # for `self.valid_decay_window` epochs
                cur_lr = self.optimizer.param_groups[0]["lr"]
                # initialize the ASGD optimizer
                self.logger.info("Begin using ASGD optimizer! "
                                 "All the models saved later will be weights-averaged models")
                self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=cur_lr,
                                                  t0=0, lambd=0., weight_decay=self.weight_decay)
                self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)

            self.valid_objs.append(valid_obj)

    def evaluate_split(self, split):
        assert split in {"train", "valid", "test"}
        data = getattr(self, split+"_data")
        obj = self.evaluate_epoch(data[0], data[1], self.bptt_steps)
        self.logger.info("eval on split %s: perp %.3f ; bpc %.3f ; loss %.3f",
                         split, np.exp(obj), obj/np.log(2), obj)
        return obj

    def evaluate_epoch(self, data, targets, bptt_steps):
        expect(self._is_setup, "trainer.setup should be called first")
        batch_size = data.shape[1]
        self.model.eval()
        objs = utils.AverageMeter()
        hiddens = self.model.init_hidden(batch_size)
        for i in range(0, data.size(0), bptt_steps):
            seq_len = min(bptt_steps, len(data)-i)
            inp, targ = data[i:i+seq_len], targets[i:i+seq_len]
            logits, _, _, hiddens = self.parallel_model(inp, hiddens)
            objs.update(self._criterion(logits.view(-1, logits.size(-1)),
                                        targ.view(-1)).item(),
                        seq_len)
        return objs.avg

    def train_epoch(self, data, targets, bptt_steps):
        expect(self._is_setup, "trainer.setup should be called first")
        batch_size = data.shape[1]
        num_total_steps = data.shape[0]
        self.model.train()
        objs = utils.AverageMeter()
        losses = utils.AverageMeter()

        hiddens = self.model.init_hidden(batch_size)

        if self.random_bptt:
            # random sequece lengths
            seq_lens = []
            i = 0
            while i < data.size(0):
                mean_ = bptt_steps if np.random.random() < 0.95 else bptt_steps / 2
                seq_len = min(max(5, int(np.random.normal(mean_, 5))), bptt_steps + 20)
                seq_lens.append(seq_len)
                i += seq_len
            seq_lens[-1] -= i - data.size(0)
            num_total_batches = len(seq_lens)
        else:
            # fixed sequence length == bptt_steps
            num_total_batches = int(np.ceil(data.size(0) / bptt_steps))
            seq_lens = [bptt_steps] * num_total_batches
            seq_lens[-1] = num_total_steps - bptt_steps * (num_total_batches-1)

        lr_bak = self.optimizer.param_groups[0]["lr"]
        i = 0
        for batch in range(1, num_total_batches+1):
            seq_len = seq_lens[batch-1]
            inp, targ = data[i:i+seq_len], targets[i:i+seq_len]

            # linear adjusting learning rate
            self.optimizer.param_groups[0]["lr"] = lr_bak * seq_len / bptt_steps
            self.optimizer.zero_grad()

            logits, raw_outs, outs, hiddens = self.parallel_model(inp, hiddens)

            raw_loss = self._criterion(logits.view(-1, logits.size(-1)), targ.view(-1))

            loss = raw_loss
            # Activiation Regularization
            if self.rnn_act_reg > 0:
                loss = loss + self.rnn_act_reg * outs.pow(2).mean()
            # Temporal Activation Regularization (slowness)
            if self.rnn_slowness_reg > 0:
                loss = loss + self.rnn_slowness_reg * (raw_outs[1:] - raw_outs[:-1]).pow(2).mean()

            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            self.model.step_current_gradients(self.optimizer)

            objs.update(raw_loss.item(), seq_len)
            losses.update(loss.item(), seq_len)

            # del logits, raw_outs, outs, raw_loss, loss

            i += seq_len
            if batch % self.report_every == 0:
                self.logger.info("train %3d/%3d: perp %.3f; loss %.3f; loss(with reg) %.3f",
                                 batch, num_total_batches, np.exp(objs.avg), objs.avg, losses.avg)

        self.optimizer.param_groups[0]["lr"] = lr_bak
        return objs.avg, losses.avg

    def _parallelize(self):
        if self.model:
            if len(self.gpus) >= 2:
                # dim=1 for batchsize, as dim=0 is time-step
                self.parallel_model = DataParallel(self.model, self.gpus, dim=1).cuda()
            else:
                self.parallel_model = self.model

    @classmethod
    def supported_data_types(cls):
        return ["sequence"]

    @staticmethod
    def _init_scheduler(optimizer, cfg):
        if cfg and optimizer is not None:
            cfg = {k:v for k, v in cfg.items()}
            sch_cls = utils.get_scheduler_cls(cfg.pop("type"))
            return sch_cls(optimizer, **cfg)
        return None

    def _scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def on_epoch_start(self, epoch):
        super(RNNFinalTrainer, self).on_epoch_start(epoch)
        self.model.on_epoch_start(epoch)

    def on_epoch_end(self, epoch):
        super(RNNFinalTrainer, self).on_epoch_end(epoch)
        self.model.on_epoch_end(epoch)
