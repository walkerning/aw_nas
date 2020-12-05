# pylint: disable-all
"""
Test a final model with PGD/FGSM attacks
"""
import os
import copy
import argparse
import subprocess

import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--load", default=None, help="load a checkpoint")
    parser.add_argument(
        "--load_state_dict", default=None, help="load checkpoint's state dict"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["FGSM", "PGD"],
        required=True,
        help="adversary type",
    )
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--n_step", type=int, default=7, help="step number of PGD attack"
    )
    parser.add_argument(
        "--rand_init",
        action="store_true",
        default=True,
        help="random init before running PGD attack",
    )
    args = parser.parse_args()

    assert (
        args.load is not None or args.load_state_dict is not None
    ), "Checkpoint Required."

    with open(args.cfg_file, "r") as rf:
        base_cfg = yaml.load(rf)

    # Construct new cfg file
    cfg = copy.deepcopy(base_cfg)
    cfg["objective_type"] = "adversarial_robustness_objective"
    cfg["objective_cfg"] = base_cfg["objective_cfg"]
    cfg["objective_cfg"]["n_step"] = args.n_step
    cfg["objective_cfg"]["adversary_type"] = args.type
    cfg["objective_cfg"]["rand_init"] = "true" if args.rand_init else "false"

    # Path for new cfg and log
    save_path = (
        os.path.dirname(args.load)
        if args.load is not None
        else os.path.dirname(args.load_state_dict)
    )
    name = os.path.join(
        save_path,
        "test_FGSM" if args.type == "FGSM" else "test_PGD{}".format(args.n_step),
    )
    test_cfg_file = name + ".yaml"
    log_file = name + ".log"

    with open(test_cfg_file, "w") as wf:
        yaml.dump(cfg, wf)

    # Start testing
    print(
        "****Test {}. Test cfg: {}. Log saved to {}.****".format(
            args.type, test_cfg_file, log_file
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
