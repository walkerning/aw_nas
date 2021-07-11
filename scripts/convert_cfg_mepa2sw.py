#pylint: disable-all
import sys
import copy
import yaml

old_cfg_path = sys.argv[1]
new_cfg_path = sys.argv[2]

with open(old_cfg_path, "r") as f:
    old_cfg = yaml.load(f)

new_cfg = copy.deepcopy(old_cfg)
eval_cfg = new_cfg["evaluator_cfg"]


assert old_cfg["evaluator_type"] == "mepa"

if "diff" in new_cfg["evaluator_cfg"]["rollout_type"]:
    evaluator_type = "differentiable_shared_weights"
else:
    evaluator_type = "discrete_shared_weights"

new_cfg["evaluator_type"] = evaluator_type

pop_list = [
    "controller_surrogate_steps",
    "mepa_surrogate_steps",
    "derive_surrogate_steps",
    "surrogate_optimizer",
    "surrogate_scheduler",
    "schedule_every_batch",
    "use_maml_plus",
    "high_order",
    "learn_per_weight_step_lr",
    "use_multi_step_loss",
    "multi_step_loss_epochs",
    "multi_step_loss_start",
    "surrogate_lr_optimizer",
    "surrogate_lr_scheduler",
    "report_inner_diagnostics",
    "report_cont_data_diagnostics",
    "update_mepa_surrogate_steps",
    "use_same_surrogate_data",
    "mepa_as_surrogate",
]

for k in pop_list:
    eval_cfg.pop(k, None)

if "data_portion" in eval_cfg:
    eval_cfg["data_portion"] = eval_cfg["data_portion"][1:]

need_key_mapping = {
    "mepa_scheduler": "eval_scheduler",
    "mepa_optimizer": "eval_optimizer",
    "mepa_samples": "eval_samples",
    "mepa_step_current": "eval_step_current"
}

for o_key, n_key in need_key_mapping.items():
    if o_key in eval_cfg:
        eval_cfg[n_key] = eval_cfg.pop(o_key)

print("Convert old configuration in {} to new configuration in {}, evaluator type {}".format(
    old_cfg_path, new_cfg_path, evaluator_type))
with open(new_cfg_path, "w") as f:
    yaml.dump(new_cfg, f)
