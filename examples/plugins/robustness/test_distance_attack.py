# pylint: disable-all
"""
Test a final model with distance attacks
"""
import os
import copy
import argparse
import subprocess

import yaml

if __name__ == "__main__":
    TEST_TYPES = {
        "BIM_L2": {
            "adversary_type": "L2BasicIterativeAttack",
            "distance_type": "MeanSquaredDistance",
        },
        "BIM_LINF": {
            "adversary_type": "LinfinityBasicIterativeAttack",
            "distance_type": "Linfinity",
        },
        "CW_L2": {
            "adversary_type": "CarliniWagnerL2Attack",
            "distance_type": "MeanSquaredDistance",
        },
        "DEEPFOOL_L2": {
            "adversary_type": "DeepFoolL2Attack",
            "distance_type": "MeanSquaredDistance",
        },
        "DEEPFOOL_LINF": {
            "adversary_type": "DeepFoolLinfinityAttack",
            "distance_type": "Linfinity",
        },
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--load", default=None, help="load a checkpoint")
    parser.add_argument(
        "--load_state_dict", default=None, help="load checkpoint's state dict"
    )
    parser.add_argument(
        "--type",
        "-t",
        required=True,
        action="append",
        default=[],
        help="distance attack type",
        choices=list(TEST_TYPES.keys()) + [t.lower() for t in TEST_TYPES.keys()],
    )
    parser.add_argument("--gpu", default=0, type=int)
    args = parser.parse_args()

    assert (
        args.load is not None or args.load_state_dict is not None
    ), "Checkpoint Required."

    with open(args.cfg_file, "r") as rf:
        base_cfg = yaml.load(rf)

    test_cfg_files = []
    log_files = []
    save_path = (
        args.load if args.load is not None else os.path.dirname(args.load_state_dict)
    )

    for test_type in args.type:
        cfg = copy.deepcopy(base_cfg)
        cfg["objective_type"] = "adversarial_distance_objective"
        cfg["objective_cfg"] = {}
        cfg["objective_cfg"]["mean"] = base_cfg["objective_cfg"]["mean"]
        cfg["objective_cfg"]["std"] = base_cfg["objective_cfg"]["std"]
        cfg["objective_cfg"]["num_classes"] = base_cfg["objective_cfg"].get(
            "num_classes", base_cfg["final_model_cfg"].get("num_classes", 10)
        )
        cfg["objective_cfg"].update(TEST_TYPES[test_type.upper()])
        test_cfg_files.append(
            "{}-test-{}.yaml".format(
                os.path.splitext(args.cfg_file)[0], test_type.upper()
            )
        )
        log_files.append(
            os.path.join(save_path, "test-{}.log".format(test_type.upper()))
        )
        with open(test_cfg_files[-1], "w") as wf:
            yaml.dump(cfg, wf)

    for test_type, test_cfg_file, log_file in zip(args.type, test_cfg_files, log_files):
        print(
            "****Test {}. Test cfg: {}. Log saved to {}.****".format(
                test_type, test_cfg_file, log_file
            )
        )
        if args.load_state_dict is not None:
            subprocess.check_call(
                "awnas test {} --load-state-dict {} --gpus {} -s test 2>&1 | tee {}".format(
                    test_cfg_file, args.load_state_dict, args.gpu, log_file
                ),
                shell=True,
            )
        elif args.load is not None:
            subprocess.check_call(
                "awnas test {} --load {} --gpus {} -s test 2>&1 | tee {}".format(
                    test_cfg_file, args.load, args.gpu, log_file
                ),
                shell=True,
            )
