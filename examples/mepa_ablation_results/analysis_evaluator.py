#pylint: disable-all
# Python3
import sys
import inspect
import argparse
import collections

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import six
import yaml
from scipy.stats import stats

## ---- correlation funtions ----
def spearmanr(surrogate_accs, accs):
    return stats.spearmanr(surrogate_accs, accs).correlation

def kendalltau(surrogate_accs, accs):
    return stats.kendalltau(surrogate_accs, accs).correlation

def mean_acc(surrogate_accs, accs):
    return np.mean(accs)

## ---- helpers ----
def _print_latex_table(control_var, search_names, display_names, caption):
    numbers = []
    for search_name in search_names:
        numbers.append([])
        for test_step in test_steps:
            numbers[-1] += corrs_dct[search_name + "_test{}".format(test_step)][0]
    best_indexes = np.argmax(np.array(numbers), axis=0)

    # print header
    multicol_str = "\multicolumn{" + str(num_corrs) + "}{c}"
    print(r"\begin{tabular}{c|" + "c" * (num_test_steps * num_corrs) + "}")
    print(r"\toprule")
    print(r"\multirow{2}[3]{*}{" + control_var + "}" r"  & " + multicol_str + multicol_str.join([r"{test step " + str(test_step) + (r"} & " if i < num_test_steps - 1 else r"} \\") for i, test_step in enumerate(test_steps)]))

    # cmid rules
    if num_corrs > 1:
        start_ind = 2
        cmid_rules = []
        for _ in range(num_test_steps):
            end_ind = start_ind + num_corrs - 1
            cmid_range = "{}-{}".format(start_ind, end_ind)
            start_ind = end_ind + 1
            cmid_rules.append(cmid_range)
        print(r"\cmidrule(lr){" + r"}\cmidrule(lr){".join(cmid_rules) + "}")

    print("".join([" & {} ".format(n) for n in simple_corr_names]) * num_test_steps + r" \\")
    print(r"\midrule")

    for i_exp, display_name in enumerate(display_names):
        print("{:15}".format(display_name), end="")
        for i_number, number in enumerate(numbers[i_exp]):
            if i_exp == best_indexes[i_number]:
                print(" & " + r"{\bf " + "{:.3f}".format(number) + "}", end="")
            else:
                print(" &    " + "{:.3f}   ".format(number), end="")
        print(r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + caption + "}")

def _plot_results(xs, corrs_list, search_names, caption, xlabel, save_name):
    fig = plt.figure(figsize=(2*(num_corrs+1), 2))
    gs = gridspec.GridSpec(nrows=1, ncols=num_corrs+1, width_ratios=[3]*num_corrs + [2])
    for i, corrs in enumerate(corrs_list):
        ax = fig.add_subplot(gs[0, i])
        handles = []
        for s_corrs in corrs:
            handles.append(ax.plot(xs, s_corrs)[0])
        ax.set_xlabel(xlabel)
        ax.set_title(corr_names[i])
        plt.xticks(xs)

    ax = fig.add_subplot(gs[:, num_corrs], frameon=False) # no frame (remove the four spines)
    plt.gca().axes.get_xaxis().set_visible(False) # no ticks
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.legend(tuple(handles), search_names, loc="center") # center the legends info in the subplot
    plt.suptitle(caption)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(save_name)
    print("Epoch corr plot saved to", save_name)

## ---- ablation functions ----
def table_names(search_names, display_names="", lr="1e-2"):
    search_names = search_names.split(",")
    caption = "correlations (lr={})".format(lr)
    control_var = ""
    if not display_names:
        display_names = [n.replace("_", "-") for n in search_names]
    else:
        display_names = display_names.split(",")
    search_names = [n + ("_lr{}".format(lr) if lr else "") for n in search_names]
    _print_latex_table(control_var, search_names, display_names, caption)

def table_epochs():
    """
    Table: Correlation - search epochs

    fixed: sample=4, test lr=1e-2, [enas, mepa step=3]
    """
    caption = "correlation - epochs (samples=4, lr=1e-2)"
    control_var = "epochs"
    display_names = ["enas 10epoch", "mepa3 10epoch",
                     "enas 20epoch", "mepa3 20epoch",
                     "enas 40epoch", "mepa3 40epoch",
                     "enas 80epoch", "mepa3 80epoch",
                     "enas 120epoch", "mepa3 120epoch",
                     "enas 160epoch", "mepa3 160epoch",
                     "enas 200epoch", "mepa3 200epoch",
                     "enas 320epoch", "mepa3 320epoch"]
    search_names = [
        "enas_sample4_10epoch_lr1e-2",
        "mepa3_sample4_10epoch_lr1e-2",
        "enas_sample4_lr1e-2",
        "mepa3_sample4_lr1e-2",
        "enas_sample4_40epoch_lr1e-2",
        "mepa3_sample4_40epoch_lr1e-2",
        "enas_sample4_80epoch_lr1e-2",
        "mepa3_sample4_80epoch_lr1e-2",
        "enas_sample4_120epoch_lr1e-2",
        "mepa3_sample4_120epoch_lr1e-2",
        "enas_sample4_160epoch_lr1e-2",
        "mepa3_sample4_160epoch_lr1e-2",
        "enas_sample4_200epoch_lr1e-2",
        "mepa3_sample4_200epoch_lr1e-2",
        "enas_sample4_320epoch_lr1e-2",
        "mepa3_sample4_320epoch_lr1e-2"
    ]
    _print_latex_table(control_var, search_names, display_names, caption)

def table_meta_steps(epoch=20, steps="0,1,3,5"):
    """
    Table: Correlation - mepa steps

    fixed: sample=4, test lr=1e-2
    """
    control_var = "mepa step"
    caption = "{} - steps (epoch={})".format("correlations" if num_corrs > 1 else corr_names[0].replace("_", " "),
                                             epoch)
    if epoch == 20:
        epoch_str = ""
    else:
        epoch_str = "_{}epoch".format(epoch)
    steps = [int(s) for s in steps.split(",")]
    search_names = []
    display_names = []
    for s in steps:
        if s == 0:
            prefix = "enas"
        else:
            prefix = "mepa{}".format(s)
        search_names.append("{}_sample4{}_lr1e-2".format(prefix, epoch_str))
        display_names.append(str(s))

    _print_latex_table(control_var, search_names, display_names, caption)

def table_samples(samples="1,2,4,6,8,12", mepa_steps="0,3", epoch=20):
    control_var = "arch samples"
    samples = [int(s) for s in samples.split(",")]
    mepa_steps = [int(step) for step in mepa_steps.split(",")]
    test_lr = "1e-2"
    caption = "correlations - samples (epoch={} lr={})".format(epoch, test_lr)
    if epoch == 20:
        epoch_str = ""
    else:
        epoch_str = "_{}epoch".format(epoch)
    search_names = []
    display_names = []
    for sample in samples:
        for step in mepa_steps:
            if step == 0:
                prefix = "enas"
            else:
                prefix = "mepa{}".format(step)
            label = "{prefix}_sample{sample}{epoch_str}_lr{test_lr}".format(
                prefix=prefix, sample=sample, epoch_str=epoch_str, test_lr=test_lr)
            search_names.append(label)
            display_names.append("{}; {}sample".format(prefix, sample))
    _print_latex_table(control_var, search_names, display_names, caption)

def plot_corr_epochs():
    """
    Plot: correlations - search epoch

    fixed: sample=4, test step=mepa steps, test lr 1e-2, [enas, mepa step=3]
    """
    caption = "{} - epochs (lr=1e-2)".format("correlations" if num_corrs > 1 else corr_names[0].replace("_", " "))
    search_names = ["enas_sample4", "mepa3_sample4"]
    corrs_list = [[] for _ in range(num_corrs)]
    epochs = [10, 20, 40, 80, 120, 160, 200, 320]
    for search_name in search_names:
        [corrs.append([]) for corrs in corrs_list]
        for epoch in epochs:
            if epoch == 20:
                suffix = ""
            else:
                suffix = "_{}epoch".format(epoch)
            if "enas" in search_name:
                label = search_name + suffix + "_lr1e-2_test0"
            else:
                label = search_name + suffix + "_lr1e-2_test" + search_name[4]
            for corrs, value in zip(corrs_list, corrs_dct[label][0]):
                corrs[-1].append(value)

    _plot_results(epochs, corrs_list, search_names, caption, "epoch", "plot_epochs.pdf")

def plot_corr_teststeps(mepa_steps="0,1,3,5", samples=4, epoch=40, test_lr="1e-2"):
    mepa_steps = [int(step) for step in mepa_steps.split(",")]
    print("plot mepa steps: ", mepa_steps)
    caption = "corr - test step (samples={} epoch={} test_lr={})".format(samples, epoch, test_lr)
    corrs_list = [[] for _ in range(num_corrs)] # num_plots x num_curves x num_xticks
    display_names = []
    if epoch == 20:
        epoch_str = ""
    else:
        epoch_str = "_{}epoch".format(epoch)
    if test_lr == "1e-3":
        lr_str = ""
    else:
        lr_str = "_lr{}".format(test_lr)

    samples_str = "sample{}".format(samples)
    for step in mepa_steps:
        [corrs.append([]) for corrs in corrs_list]
        if step == 0:
            prefix = "enas"
        else:
            prefix = "mepa{}".format(step)
        for test_step in test_steps:
            label = "{prefix}_{samples_str}{epoch_str}{lr_str}_test{test_step}".format(
                prefix=prefix, samples_str=samples_str, epoch_str=epoch_str,
                lr_str=lr_str, test_step=test_step)
            for corrs, value in zip(corrs_list, corrs_dct[label][0]):
                corrs[-1].append(value)
        display_names.append(prefix)

    _plot_results(test_steps, corrs_list, display_names, caption, "test step",
                  "plot_corr_teststep_{}epoch_{}sample_{}.pdf".format(epoch, samples, test_lr))

def plot_scatter(mepa_steps="0,1,3,5", epoch=40, samples=4, test_lr="1e-2"):
    _s_accs = np.array(surrogate_accs)/100
    mepa_steps = [int(step) for step in mepa_steps.split(",")]
    num_search = len(mepa_steps)
    if epoch == 20:
        epoch_str = ""
    else:
        epoch_str = "_{}epoch".format(epoch)
    if test_lr == "1e-3":
        lr_str = ""
    else:
        lr_str = "_lr{}".format(test_lr)
    samples_str = "sample{}".format(samples)
    fig = plt.figure(figsize=(3*num_search, 3))
    gs = gridspec.GridSpec(nrows=1, ncols=num_search)

    for i, step in enumerate(mepa_steps):
        if step == 0:
            prefix = "enas"
        else:
            prefix = "mepa{}".format(step)
        # use the same test step as mepa step
        label = "{prefix}_{samples_str}{epoch_str}{lr_str}_test{test_step}".format(
            prefix=prefix, samples_str=samples_str, epoch_str=epoch_str,
            lr_str=lr_str, test_step=step)
        ax = fig.add_subplot(gs[0, i])
        plt.scatter(_s_accs, accs_dct[label])
        ax.set_xlim([0.6, 0.9])
        ax.set_ylim([0.4, 0.9])
        ax.set_title(prefix)
    plt.suptitle("one-shot acc - acc (samples={} epoch={} test_lr={})".format(
        samples, epoch, test_lr))
    plt.tight_layout()
    save_name = "plot_scatter_{}epoch_{}sample_{}.pdf".format(epoch, samples, test_lr)
    print("Save to ", save_name)
    plt.savefig(save_name)

def plot_layeraug(search_names, layers, display_names=""):
    layers = [int(x) for x in layers.split(",")]
    search_names = search_names.split(",")
    caption = "correlations - surrogate layers"
    control_var = "surrogate layers"
    corrs_list = [corrs_dct[name] for name in search_names] # num_curves x num_xticks x num_plots
    corrs_list = np.transpose(corrs_list, [2, 0, 1])# num_plots x num_curves x num_xticks
    if not display_names:
        display_names = search_names
    else:
        display_names = display_names.split(",")
    assert len(display_names) == len(search_names)
    _plot_results(layers, corrs_list, display_names, caption,
                  "surrogate layers", "plot_layeraug.pdf")

avail_ablation = {
    "table_names": table_names,
    "table_steps": table_meta_steps,
    "table_epochs": table_epochs,
    "table_samples": table_samples,
    "plot_corr_epochs": plot_corr_epochs,
    "plot_corr_teststeps": plot_corr_teststeps,
    "plot_scatter": plot_scatter,
    "plot_layeraug": plot_layeraug
}
## ---- main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--surrogate-name", default="surrogate")
    parser.add_argument("-f", "--result-file", default="mepa_100arch_results.yaml")
    parser.add_argument("-n", default=None, type=int, help="number of data point used")
    subparsers = parser.add_subparsers(help="Type of ablation results")
    parser.add_argument("-t", "--test-steps", default="0,1,3,5",
                        help="Test steps. only work for some ablation")
    parser.add_argument("-c", "--corr-funcs", default="s,k,a",
                        help="Correlation funcs.")

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

    # test surrogate steps
    test_steps = [int(step) for step in args.test_steps.split(",")]
    num_test_steps = len(test_steps)

    # correlation tests
    avail_corr_dct = {
        "s": ("spearmanr", "s"),
        "k": ("kendalltau", "k"),
        "a": ("mean_acc", "$E_t[acc]$")
    }
    arg_corrs = args.corr_funcs.split(",")
    corr_names, simple_corr_names = zip(*[avail_corr_dct[n] for n in arg_corrs])
    corr_names = list(corr_names)
    simple_corr_names = list(simple_corr_names)
    num_corrs = len(corr_names)

    # ---- handling results file ----
    accs_dct = yaml.safe_load(open(args.result_file))
    args.surrogate_name = args.surrogate_name.split(",")
    surrogate_accs_list = [accs_dct.pop(s_name) for s_name in args.surrogate_name]
    if args.n:
        surrogate_accs_list = [s_accs[:args.n] for s_accs in surrogate_accs_list]
    # print({n: len(accs) for n, accs in six.iteritems(accs_dct) if isinstance(accs, (list, tuple))})

    corrs_dct = {}
    ignored_dueto_n = 0
    for n, accs in six.iteritems(accs_dct):
        if "surrogate" in n:
            continue
        if args.n:
            accs = accs[:args.n]
            if len(accs) < args.n:
                ignored_dueto_n += 1
                continue
        if not (isinstance(accs, (list, tuple)) and accs):
            continue
        try:
            _corrs = [[globals()[corr_name](s_accs, accs) for corr_name in corr_names]
                      for s_accs in surrogate_accs_list]
        except (ValueError, TypeError):
            print(n)
            raise
        corrs_dct[n] = _corrs

    print("INFO: Total {} test results loaded".format(len(corrs_dct)))
    if ignored_dueto_n > 0:
        print("INFO: Ignored {} records because there are not {} derived accs in the value".format(ignored_dueto_n, args.n))

    for n, accs in six.iteritems(accs_dct):
        if isinstance(accs, str) and accs in corrs_dct:
            corrs_dct[n] = corrs_dct[accs]
            accs_dct[n] = accs_dct[accs]
    # ---- end handle ----

    args.func(*[getattr(args, name) for name in args.param_names])

    # for name in sys.argv[1:]:
    #     print("---- Results {} ----".format(name))
    #     avail_ablation[name]()
    #     print("---- End results {} ----".format(name))

