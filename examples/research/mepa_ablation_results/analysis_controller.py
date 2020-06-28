#pylint: disable-all
import os
import re
import sys
import inspect
import argparse
import numpy as np
import six
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_evaluator import spearmanr, kendalltau, mean_acc

def _print_table(caption, control_var, display_names, corr_names, corr_results, num_eva, num_corrs,
                 _surrogate_names, _surrogate_accs_list, _mean_surrogate_acc_list, max_direction="row"):
    assert max_direction in {"row", "col"}, \
        "label out the max correlation among each *row* or each *column*"
    assert len(display_names) == num_eva
    assert len(corr_names) == num_corrs

    if max_direction == "row":
        # field index to highlight
        max_inds = np.argmax(np.reshape(corr_results, (-1, num_eva, num_corrs)), axis=1)
        max_inds = max_inds * num_corrs + np.arange(num_corrs)
    else: # col
        max_inds = np.argmax(corr_results, axis=0)

    mean_acc_max_ind = np.argmax(_mean_surrogate_acc_list)

    # print header
    multicol_str = "\multicolumn{" + str(num_corrs) + "}{c}"
    print(r"\begin{tabular}{c|c" + "c" * (num_eva * num_corrs) + "}")
    print(r"\toprule")
    print(r"\multirow{2}[3]{*}{" + control_var + "}" r"  & E_t[acc] & " + multicol_str + multicol_str.join([r"{" + dis_name + (r"} & " if i < num_eva - 1 else r"} \\") for i, dis_name in enumerate(display_names)]))

    # cmid rules
    if num_corrs > 1:
        start_ind = 3
        cmid_rules = []
        for _ in range(num_eva):
            end_ind = start_ind + num_corrs - 1
            cmid_range = "{}-{}".format(start_ind, end_ind)
            start_ind = end_ind + 1
            cmid_rules.append(cmid_range)
        print(r"\cmidrule(lr){" + r"}\cmidrule(lr){".join(cmid_rules) + "}")

    print("& " + "".join([" & {} ".format(n) for n in corr_names]) * num_eva + r" \\")
    print(r"\midrule")

    for i_exp, s_name in enumerate(_surrogate_names):
        # print surrogate/controller name, and the mean acc of the archs sampled by this controller
        if i_exp == mean_acc_max_ind:
            print("{:18} ({:3d}) & ".format(s_name.replace("_", "-"),
                                            len(_surrogate_accs_list[i_exp]))
                  + r"{\bf " +
                  "{:5.2f}".format(_mean_surrogate_acc_list[i_exp]) + "}", end="")
        else:
            print("{:18} ({:3d}) &    {:5.2f}   ".format(
                s_name.replace("_", "-"),
                len(_surrogate_accs_list[i_exp]),
                _mean_surrogate_acc_list[i_exp]), end="")
        for i_number, number in enumerate(corr_results[i_exp]):
            if (max_direction == "row" and i_number in max_inds[i_exp]) or \
               (max_direction == "col" and i_exp == max_inds[i_number]):
                print(" & " + r"{\bf " + "{:.3f}".format(number) + "}", end="")
            else:
                print(" &    " + "{:.3f}   ".format(number), end="")
        print(r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + caption + "}")

def prob(corr_names="spearmanr,kendalltau", lr="1e-2"):
    caption = "Prob-acc correlation on controller sampled archs (lr={})".format(lr)
    control_var = "controller"

    lr_str = "" if lr == "1e-2" else ("lr" + lr + "-")
    corr_funcs = [globals()[corr_name] for corr_name in corr_names.split(",")]
    simple_corr_names = [corr_name[0] for corr_name in corr_names.split(",")]
    num_corrs = len(corr_funcs)

    corr_results = []
    for s_name, sur_accs in zip(surrogate_names, surrogate_accs_list):
        logprobs = accs_dct[s_name + "-derive-logprob"]
        eva_mepa_step = int(re.search("mepa([0-9]+).*", s_name).group(1))
        derive_accs = accs_dct["{}-derive-{}-{}test{}".format(
            s_name, s_name, lr_str, eva_mepa_step)]
        derive_corrs = [corr_func(logprobs, derive_accs) for corr_func in corr_funcs]
        final_corrs = [corr_func(logprobs, sur_accs) for corr_func in corr_funcs]
        corr_results.append(derive_corrs + final_corrs)

    _print_table(caption, control_var, ["derive", "final"], simple_corr_names, corr_results, 2, num_corrs,
                 surrogate_names, surrogate_accs_list, mean_surrogate_acc_list, max_direction="col")

def correlation(evaluator, corr_names="spearmanr,kendalltau", display_names="", lr="1e-2"):
    caption = "Evaluator correlation on controller sampled archs (lr={})".format(lr)
    control_var = "controller"

    lr_str = "" if lr == "1e-2" else ("lr" + lr + "-")
    corr_funcs = [globals()[corr_name] for corr_name in corr_names.split(",")]
    simple_corr_names = [corr_name[0] for corr_name in corr_names.split(",")]
    evaluator_names = evaluator.split(",")
    if display_names:
        display_names = display_names.split(",")
    else:
        display_names = evaluator_names
    num_corrs = len(corr_funcs)
    num_eva = len(evaluator_names)

    # prepare correlation results
    corr_results = []
    _surrogate_names = surrogate_names + ["all"]
    _surrogate_accs_list = surrogate_accs_list + [sum(surrogate_accs_list, [])]
    _mean_surrogate_acc_list = mean_surrogate_acc_list + [np.mean(mean_surrogate_acc_list)]
    for s_name, s_accs in zip(_surrogate_names, _surrogate_accs_list):
        corr_results.append([])
        for eva_name in evaluator_names:
            eva_mepa_step = int(re.search("mepa([0-9]+).*", eva_name).group(1))
            if s_name == "all":
                derive_accs = sum([accs_dct["{}-derive-{}-{}test{}".format(
                    s_name, eva_name, lr_str, eva_mepa_step)]
                                   for s_name in surrogate_names], [])
            else:
                derive_accs = accs_dct["{}-derive-{}-{}test{}".format(
                                       s_name, eva_name, lr_str, eva_mepa_step)]
            for corr_func in corr_funcs:
                corr_results[-1].append(corr_func(derive_accs, s_accs))

    _print_table(caption, control_var, display_names, simple_corr_names,
                 corr_results, num_eva, num_corrs, _surrogate_names,
                 _surrogate_accs_list, _mean_surrogate_acc_list)


avail_ablation = {
    "corr": correlation,
    "prob-corr": prob
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--surrogate-name", default="surrogate")
    parser.add_argument("-f", "--result-file", default="mepa_controller_results.yaml")
    subparsers = parser.add_subparsers(help="Type of ablation results")
    for name, func in avail_ablation.items():
        sub_parser = subparsers.add_parser(
            name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        sub_parser.set_defaults(func=func)
        sig = inspect.signature(func)
        sub_parser.set_defaults(param_names=list(sig.parameters.keys()))

        for n, param in sig.parameters.items():
            if not param.default is param.empty:
                # default value as option
                kwargs = {
                    "default": param.default
                }
                if param.default is not None:
                    kwargs["type"] = type(param.default)
            else:
                kwargs = {"required": True}
            sub_parser.add_argument("--{}".format(n.replace("_", "-")),
                                **kwargs)

    args = parser.parse_args()

    # ---- handling results file ----
    accs_dct = yaml.safe_load(open(args.result_file))
    surrogate_names = args.surrogate_name.split(",")
    surrogate_accs_list = [accs_dct.pop(sur_name + "-surrogate") for sur_name in surrogate_names]
    mean_surrogate_acc_list = [np.mean(surrogate_accs) for surrogate_accs in surrogate_accs_list]

    for n, accs in six.iteritems(accs_dct):
        if isinstance(accs, str):
            accs_dct[n] = accs_dct[accs]
    # ---- end handle ----

    args.func(*[getattr(args, name) for name in args.param_names])
