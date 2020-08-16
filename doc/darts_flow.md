## Walk-through of the DARTS implementation

The DARTS cfg file is [here](../examples/basic/darts.md), also check [enas cfg breakup](enas_cfg_notes.md) for some explainations of the cfg sections.

All `rollout_type` in DARTS configuration file is **differentiable**.

### Components

#### Search space

* class: `CNNSearchSpace` (in `aw_nas.common`)
    * 8 cells
    * [0,0,1,0,0,1,0,0] (1 denotes where the reduction cell is)

#### Controller

* class: `DifferentiableController` (in `aw_nas.controller.differentiable`)
* Call: `aw_nas.trainer.simple.SimpleTrainer._controller_update`
    * `rollout = controller.sample()`  - sample architecture from search space
    * `evaluator.evaluate_rollout(rollout)` - call `evaluator.evaluate_rollout` to evaluate the architecture
    * `controller.step()` - update the controller with evaluated rollouts, in DARTS, the controller parameters is the architecture parameters a, and they are updated with validation set gradients

#### Weights manager
* class: `DiffSuperNet` (in `aw_nas.weights_manager.diff_super_net`)
* Call: the `cand_net.eval_data`, `cand_net.eval_queue`, `cand_net.gradients` calls in `aw_nas.tranier.simple.SimpleTrainer.evaluate_rollout` `aw_nas.tranier.simple.SimpleTrainer.update_evaluator`
* Interfaces:
  * `DiffSubCandidateNet` instance will have a reference to the architecture parameter `arch`, and its `forward` call will call `self.super_net.forward(inputs, self.arch)` (See `aw_nas.weights_manager.diff_super_net.DiffSubCandidateNet.forward`)
  * `diff_supernet.forward(inputs, arch)` - forward a sub-net/candidate net of the supernet using the specified `arch` parameter
  * `diff_supernet.assemble_candidate(rollout)` - assemble an architecture rollout into a candidate network `DiffSubCandidateNet`

#### Evaluator
* class: `MepaEvaluator` (in `aw_nas.evaluator.mepa`) Compatible with shared-weights evaluator, please refer to the [config breakup](./enas_cfg_notes.md#component-evaluator) for more details
* Call: `aw_nas.trainer.simple.SimpleTrainer._evaluator_update()`
* Interfaces:
    * `evaluator.update_rollout(rollout)` - do nothing
    * `evaluator.update_evaluator(controller)` - optimize the shared weights on the training data queue (`mepa_queue` in the code)
    * `evaluator.evaluate_rollout(rollout)` - evaluate the rollout on the validation data queue (`controller_queue` in the code)
