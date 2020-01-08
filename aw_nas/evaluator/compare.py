"""
Compare evaluator
"""

from collections import OrderedDict
from aw_nas.evaluator.base import BaseEvaluator
from aw_nas.evaluator.arch_network import ArchNetwork

class ArchNetworkEvaluator(BaseEvaluator):
    NAME = "comparator"

    def __init__(
            self, dataset, weights_manager, objective, rollout_type="compare",
            guide_evaluator_type="mepa", guide_evaluator_cfg=None,
            arch_network_type="pointwise_comparator", arch_network_cfg=None,
            guide_batch_size=16,
            schedule_cfg=None
    ):
        super(ArchNetworkEvaluator, self).__init__(
            dataset, weights_manager, objective, rollout_type, schedule_cfg)

        # construct the evaluator that will be used to guide the learning of the predictor
        ge_cls = BaseEvaluator.get_class_(guide_evaluator_type)
        self.guide_evaluator = ge_cls(dataset, weights_manager, objective,
                                      rollout_type=rollout_type,
                                      **(guide_evaluator_cfg or {}))

        # construct the architecture network
        an_cls = ArchNetwork.get_class_(arch_network_type)
        self.arch_network = an_cls(search_space=self.weights_manager.search_space,
                                   **(arch_network_cfg or {}))

        # configurations
        self.guide_batch_size = guide_batch_size

    # ---- APIs ----
    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["compare", "discrete"]

    def suggested_controller_steps_per_epoch(self):
        return self.guide_evaluator.suggested_controller_steps_per_epoch()

    def suggested_evaluator_steps_per_epoch(self):
        return self.guide_evaluator.suggested_evaluator_steps_per_epoch()

    def evaluate_rollouts(self, rollouts, is_training, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        for rollout in rollouts:
            if self.rollout_type == "compare":
                better = self.arch_network.compare(rollout.rollout1.arch, rollout.rollout2.arch)
                rollout.set_perfs(OrderedDict(
                    [
                        ("compare_result", better),
                        # ("confidence", 1),
                    ]))
            elif self.rollout_type == "discrete":
                acc = self.arch_network.predict(rollout.arch)
                rollout.set_perfs(OrderedDict(
                    [
                        ("reward", acc),
                    ]))
        return rollouts

    def update_evaluator(self, controller):
        # sample CompareRollout
        rollouts = controller.sample(n=self.guide_batch_size)

        # evaluate using the guide evaluator
        # FIXME: should `is_training` be true or false?
        rollouts = self.guide_evaluator.evaluate_rollouts(rollouts, is_training=True)

        if self.rollout_type == "compare":
            self.arch_network.update_compare_rollouts(
                rollouts, [r.perf["compare_result"] for r in rollouts])
        elif self.rollout_type == "discrete":
            self.arch_network.update_predict_rollouts(
                rollouts, [r.perf["reward"] for r in rollouts])

        # TODO: return stats
        return OrderedDict()

    def update_rollouts(self, rollouts):
        """
        Do not need.
        """
        pass

    def save(self, path):
        self.arch_network.save("{}-arch_network".format(path))

    def load(self, path):
        self.arch_network.load("{}-arch_network".format(path))

    def on_epoch_start(self, epoch):
        super(ArchNetworkEvaluator, self).on_epoch_start(epoch)
        self.arch_network.on_epoch_start(epoch)
        self.guide_evaluator.on_epoch_start(epoch)


class BatchUpdateArchNetworkEvaluator(BaseEvaluator):
    NAME = "batch_update_comparator"

    def __init__(
            self, search_space, rollout_type="compare",
            arch_network_type="pointwise_comparator", arch_network_cfg=None,
            schedule_cfg=None
    ):
        super(BatchUpdateArchNetworkEvaluator, self).__init__(
            dataset=None, weights_manager=None, objective=None,
            rollout_type=rollout_type, schedule_cfg=schedule_cfg)

        # construct the architecture network
        an_cls = ArchNetwork.get_class_(arch_network_type)
        self.arch_network = an_cls(search_space=self.weights_manager.search_space,
                                   **(arch_network_cfg or {}))

    # ---- APIs ----
    @classmethod
    def supported_data_types(cls):
        return ["image"]

    @classmethod
    def supported_rollout_types(cls):
        return ["compare", "discrete"]

    def suggested_controller_steps_per_epoch(self):
        return None

    def suggested_evaluator_steps_per_epoch(self):
        return None

    def evaluate_rollouts(self, rollouts, is_training, portion=None, eval_batches=None,
                          return_candidate_net=False, callback=None):
        archs = [r.arch for r in rollouts]
        scores = self.arch_network.predict(archs)
        for rollout, score in zip(rollouts, scores):
            rollout.set_perfs(OrderedDict(
                [
                    ("reward", acc),
                ]))
        # for rollout in rollouts:
        #     if self.rollout_type == "compare":
        #         better = self.arch_network.compare(rollout.rollout1.arch, rollout.rollout2.arch)
        #         rollout.set_perfs(OrderedDict(
        #             [
        #                 ("compare_result", better),
        #                 # ("confidence", 1),
        #             ]))
        #     elif self.rollout_type == "discrete":
        #         acc = self.arch_network.predict(rollout.arch)
        #         rollout.set_perfs(OrderedDict(
        #             [
        #                 ("reward", acc),
        #             ]))
        # return rollouts
        return rollouts

    def update_evaluator(self, controller):
        raise NotImplementedError("`update_evaluator` of batch_update_comparator "
                                  "should not be called; maybe you want to call `update_rollouts``")

    def update_rollouts(self, rollouts):
        # TODO: batch training of predictor
        # init dataset
        pass

    def save(self, path):
        self.arch_network.save("{}-arch_network".format(path))

    def load(self, path):
        self.arch_network.load("{}-arch_network".format(path))

    def on_epoch_start(self, epoch):
        super(BatchUpdateArchNetworkEvaluator, self).on_epoch_start(epoch)
        self.arch_network.on_epoch_start(epoch)
