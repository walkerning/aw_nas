# Development of New Components

To implement a component, you should inherit from the corresponding base class. Some examples are as follows.
* Search space: `aw_nas.common.SearchSpace`
* Rollout: `aw_nas.rollout.base.BaseRollout`
* Dataset: `aw_nas.dataset.base.BaseDataset`
* Controller: `aw_nas.controller.base.BaseController`
* Weights manager: `aw_nas.weights_manager.base.BaseWeightsManager`
* Evaluator: `aw_nas.evaluator.base.BaseEvaluator`
* Objective: `aw_nas.evaluator.base.BaseObjective`
* Trainer: `aw_nas.trainer.base.BaseTrainer`
* Architecture network: `aw_nas.evaluator.arch_network.ArchNetwork`
* Final trainer: `aw_nas.final.base.FinalTrainer`
* Final model: `aw_nas.final.base.FinalModel`

One can use the help utility `awnas registry` to print out the current registry information. For some examples,
* `awnas registry` print all the classes in all the registries.
* `awnas registry -t search_space -t rollout -n 'nasbench.*' -r -v` means to print the classes in the `search_space and `rollout` registries whose name starts with "nasbench". `-r` option means using regexp matching instead of exact matching. `-v` option means also printing the class documentation.

Each component type has a few abstract interfaces/methods declared, and one should implement all these abstract methods to get the new component class successfully registered. If some classes are not successfully registered, one will get warnings during awnas initialization.

## Define a New Rollout
Rollouts are the interface objects that are defined by the search space, sampled from controller, evaluated by the weights manager and evaluator. To define a new rollout, you should first define the corresponding search space. A **search space** contains some searchable decisions, and a **rollout** instance contains a specific set of these decisions. In `aw_nas`, a **genotype** is a more human-readable representation of the , which can be a namedtuple (ENAS, DenseNet, OFA) or a string (NAS-Bench-201), or of other types (ModelSpec for NAS-Bench-101).

You can check existing search space and rollout definitions for more examples. Run `awnas registry -t search_space` to see which search space definitions exist. And a search space class corresponds to one or more rollout classes. For example, the basic cell-based search space `cnn` (class `awnas.common.CNNSearchSpace`) corresponds to two rollout types: `discrete` discrete rollouts that are used in RL-based, EVO-based controllers, etc. (class `awnas.rollout.base.Rollout`); `differentiable` differentiable rollouts that are used in gradient-based NAS (class `awnas.rollout.base.DifferentiableRollout`).

**Define component's supported rollout types**: Certain components support handling specific types of rollouts, and the component class and declare which rollout types it supports, and checks will be carried out according to this information during component initialization. For example, the parameter-sharing evaluator `aw_nas.evaluator.mepa.MepaEvalautor` declares it supported rollouts as
```
@classmethod
def supported_rollout_types(cls):
    return ["discrete", "differentiable", "compare", "ofa"]
```

Apart from this method, during, a rollout class can also declare which components it can use with a class attribute named `supported_components`. For example, when we want to reuse the parameter-sharing `mepa` evaluator and the reinforcment-learning `rl` controller for `nasbench-201` rollouts, instead of modifying these components' `supported_rollout_types` classmethod def., we add following codes in the `NasBench201Rollout` class def..
```
class NasBench201Rollout(BaseRollout):
    NAME = "nasbench-201"
    supported_components = [("controller", "rl"), ("evaluator", "mepa")]

    def __init__(self, matrix, search_space):
       ...
```

This mechanism enables a newly defined rollout type to reuse components without intrusively modify the `supported_rollout_types` definition in the components' codes. However, one might need understand the component's mechanism and check its code to ensure this component can be indeed reused for the newly-defined rollout type.

## Pass in Configurations

The mechanism that `aw_nas` pass the configurations defined in the configuration files to the components is simple: Pass all configurations in the file as keyword arguments to initialize the corresponding component instance `ComponentClass(**config_dict)`.

It's recommended to give all configuration terms a default value, since the `awnas gen-sample-config` and `awnas gen-final-sample-config` utilities parse the signature of the `__init__` method to generate the default configuration files. For required configurations with no reasonable default values, one can using `aw_nas.utils.exception.expect` or `assert` inside the `__init__` to check the configuration sanity.

## Sub-components

There can be sub component types in some component class, for which one can reuse the registry mechanism. For example, RL-based controller includes two further choices: 1) `rl_agent`: what RL agent (e.g., PG, PPO, etc.) is used to optimize the controller, 2) `controller_network`: how to construct the controller network (e.g., embedding-based RNN, attention-based RNN, etc.). Thus, two sub component registries, `rl_agent` and `controller_network`, are defined. And in the `__init__` method of `aw_nas.controller.rl.RLController`, the sub-components are constructed by (see `aw_nas/controller/rl.py` for the complete codes):
```
        cn_cls = BaseRLControllerNet.get_class_(controller_network_type)
        ...
        self.controllers = [cn_cls(self.search_space, self.device,
                                   cell_index=i if self.independent_cell_group else None,
                                   **controller_network_cfg) for i in range(num_cnet)]
        self.agents = [BaseRLAgent.get_class_(rl_agent_type)(cnet, **rl_agent_cfg)\
                       for cnet in self.controllers]
```

Another example is that a parametrized architecture network is usually constructed by an architecture embedder followed by an MLP to output a score. Different architecture embedders can be used, thus we define a component type `arch_embedder`, and the base class is `aw_nas.evaluator.arch_network.ArchEmbedder`. To implement a new architecture embedder, one can see `aw_nas/btcs/nasbench_101.py`, `aw_nas/btcs/nasbench_201.py`, `aw_nas/evaluator/arch_network.py` and for examples in different search spaces (ENAS, NAS-Bench-101, NAS-Bench-201).

## Utilities

Since all components inherit from the base class `aw_nas.base.Component`, some basic and common utilities are implemented in this base class.

* Logging: `self.logger` will be created on demand, and is a `logging.Logger` instance. Just call `self.logger.warn`, `self.logger.info`, etc. at where you need.
* Scheduling: You can define some instance attributes as `SCHEDULABLE_ATTRS`, the corresponding attributes will be scheduled according to the `schedulable_cfg` attribute at each epoch. A simple example is as follows, in which the weighted coefficient of two rewards are schedulable. And according to the `schedule_cfg` of `coeff` in the configuration file, the instance attribute `coeff` will decay by 0.9 every 5 epoch, starting from epoch 50, and stop decaying when it reaches 0.01.

```
class FakeObjective(BaseObjective):
    NAME = "an_fake_example_objective"
    SCHEDULABLE_ATTRS = ["coeff"]

    def __init__(self, search_space, init_coeff=1., schedule_cfg=None):
        super(FakeObjective, self).__init__(search_space, schedule_cfg)
	self.coeff = init_coeff

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.reward1(inputs, outputs, targets, cand_net) + self.coeff * self.reward2(inputs, outputs, targets, cand_net)

    def reward1(self, inputs, outputs, targets, cand_net):
        # calculate reward 1
        ...

    def reward2(self, inputs, outputs, targets, cand_net):
        # calculate reward 2
        ...
```

```
objective_type: an_fake_example_objective
objective_cfg:
  init_coeff: 1.
  schedule_cfg:
    coeff:
      type: mul
      min: 0.01
      every: 5
      start_epoch: 50
      step: 0.9
```