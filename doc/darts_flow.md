# Example of the darts

> For details, refer to [cfg file](../examples/basic/darts.md)

all rollout_type for darts is **differentiable**

## Components

### search space

* type: CNNSearchSpace (in ```$AW_NAS_HOME/common.py```)
    * 8 cells
    * [0,0,1,0,1,0,1] (1 denotes where the reduction cell is)
    * 2 input_node
    * 4 inner_node(num_steps)

### Controller

* type: DifferentiableController (in ```$AW_NAS_HOME/controller/differentiable.py```)
* forward: ```controller._controller_update()```
    * ```rollout = controller.sample()```  - sample architecture from search space (determine the ɑ)
    * ```evaluator.evaluate&update(rollout)``` - call evaluate to evaluate the architecture (acquire best Ω under current ɑ)
    * ```controller_optimizer.step()``` - update the ɑ with evaluator rewards  

### Weight-manager
* type: DiffSuperNet (in ```$AW_NAS_HOME/weights_manager/differentiable.py```)
* forward: ````diff_supernet.forward(inputs, arch)``` - fill the sampled arch with shared-weights and acquire output
* ```diff_supernet.subcandidate_net``` - the sampled arch from supernet
    * ```eval_data(input)```
* ```diff_supernet.assemble_candidate()``` - pack the arch into rollouts

### Evaluator
* type: MepaEvaluator(**when the steps are 0, just plain evaluator**)
* forward: ```evaluator._evaluator_update()```
    * ```update_rollout(rollout)``` - pass 
    * ```update_evaluator(controller)``` - optimize Ω on the train set, acquire performance of arch in valid set as the reward for updating the controller
* mepa optimizer/scheduler: specifies optimization hyper-params for updating Ω on trainset
* data_potion - [0,0.5,0.5] - half of the dataset as train-set(update Ω), the other half as valid-set(update ɑ)
