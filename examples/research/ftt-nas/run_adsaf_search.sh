#!/bin/bash
set -e

here=$(dirname "$0")
weight_fault=1
gpu=${GPU:-0}
cfg_file=${1}
default_exp_name=$(basename ${cfg_file})
default_exp_name=${default_exp_name%.yaml}
exp_name=${2:-${default_exp_name}}
result_dir=$here/results/ftt_nas_adsaf_search/$exp_name

if [[ $weight_fault -gt 0 ]]; then
    echo "$result_dir/awnas/plugins"
    if [[ -d "$result_dir/awnas/plugins" ]]; then
	rm -r $result_dir/awnas/plugins
    fi
    mkdir -p $result_dir/awnas/plugins
    if [[ ! -e $result_dir/awnas/data ]]; then
	ln -s ~/awnas/data $result_dir/awnas/data
    fi
    cp $here/fixed_point_plugins/fixed_point_rram_patch_bit.py $result_dir/awnas/plugins/
fi

AWNAS_HOME=$result_dir/awnas/ awnas search --gpu $gpu --train-dir $result_dir/train --vis-dir results/tensorboard_new/weights/$exp_name/ --save-every 10 $cfg_file --develop
