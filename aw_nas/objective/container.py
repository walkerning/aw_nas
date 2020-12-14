from aw_nas.objective.base import BaseObjective


class ContainerObjective(BaseObjective):
    NAME = "container"

    def __init__(self, search_space, sub_objectives,
                 losses_coef=None, rewards_coef=None,
                 schedule_cfg=None):
        super().__init__(search_space, schedule_cfg=schedule_cfg)
        self.objectives = [
            BaseObjective.get_class_(obj["objective_type"])(
                search_space, **obj["objective_cfg"]) for obj in sub_objectives
        ]
        self.losses_coef = losses_coef
        self.rewards_coef = rewards_coef
        if self.losses_coef is None:
            self.losses_coef = [1.] * len(self.objectives)
        if self.rewards_coef is None:
            self.rewards_coef = [1.] * len(self.objectives)
        assert len(self.rewards_coef) == len(self.losses_coef) == len(self.objectives), \
            ("expect rewards_coef and losses_coef have the exactly"
             "same length with objectives, got {}, {} and {} instead.").format(
                 len(rewards_coef), len(self.losses_coef), len(self.objectives))

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def aggregate_fn(self, perf_name, is_training=True):
        for obj in self.objectives:
            if perf_name in obj.perf_names():
                return obj.aggregate_fn(perf_name, is_training)
        else:
            return super().aggregate_fn(perf_name, is_training)

    def perf_names(self):
        return sum([obj.perf_names() for obj in self.objectives], [])

    def get_perfs(self, inputs, outputs, targets, cand_net):
        perfs = []
        for obj in self.objectives:
            perfs.extend(obj.get_perfs(inputs, outputs, targets, cand_net))
        assert len(perfs) == len(self.perf_names()), \
            ("expect performances have the exactly "
             "same length with perf_names, got {} and {} instead.").format(
                 len(perfs), len(self.perf_names()))
        return perfs

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        losses = [
            obj.get_loss(
                inputs, outputs, targets, cand_net,
                add_controller_regularization, add_evaluator_regularization
            ) for obj in self.objectives
        ]
        weighted_loss = [l * c for l, c in zip(losses, self.losses_coef)]
        return sum(weighted_loss)

    def get_reward(self, inputs, outputs, targets, cand_net):
        rewards = [
            obj.get_reward(
                inputs, outputs, targets, cand_net
            ) for obj in self.objectives
        ]
        weighted_rewards = [l * c for l, c in zip(rewards, self.rewards_coef)]
        return sum(weighted_rewards)
