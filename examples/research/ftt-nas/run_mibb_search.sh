#!/bin/bash
set -e

here=$(dirname "$0")
fixed=${FIXED:-1}
gpu=${GPU:-0}
cfg_file=${1}
default_exp_name=$(basename ${cfg_file})
default_exp_name=${default_exp_name%.yaml}
exp_name=${2:-${default_exp_name}}
result_dir=$here/results/ftt_nas_mibb_search/$exp_name
addi_args=${ADDI_ARGS:-""}

echo "use plugin dir: $result_dir/awnas/plugins"
if [[ -d "$result_dir/awnas/plugins" ]]; then
    rm -r $result_dir/awnas/plugins
fi
mkdir -p $result_dir/awnas/plugins
if [[ ! -e $result_dir/awnas/data ]]; then
    ln -s $HOME/awnas/data $result_dir/awnas/data
fi

if [[ $fixed -gt 0 ]]; then
    echo "copy fixed patch to plugin dir $result_dir/awnas/plugins/"
    cp $here/fixed_point_plugins/fixed_point_patch_new.py $result_dir/awnas/plugins/
fi
# For profiling only
# AWNAS_HOME=$result_dir/awnas/ python -m cProfile awnas search --gpu $gpu --train-dir $result_dir/train --vis-dir results/tensorboard/ftt_search_tcad/$exp_name/ --save-every 10 ${2} --develop ${addi_args}

AWNAS_HOME=$result_dir/awnas/ awnas search --gpu $gpu --train-dir $result_dir/train --vis-dir $result_dir/tensorboard --save-every 10 ${cfg_file} --develop ${addi_args}
