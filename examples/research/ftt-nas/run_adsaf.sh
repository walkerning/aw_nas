#!/bin/bash
here=$(dirname "$0")
gpu=${GPU:-0}
weight_fault=1
exp_name=${1}
result_dir=$here/results/ftt_nas_adsaf/$exp_name
seed=${seed:-123}

if [[ $weight_fault -gt 0 ]]; then
    if [[ -d "$result_dir/awnas/plugins" ]]; then
        rm -r $result_dir/awnas/plugins
    fi
    mkdir -p $result_dir/awnas/plugins
    if [[ ! -e $result_dir/awnas/data ]]; then
        ln -s ~/awnas/data $result_dir/awnas/data
    fi
    # cp $here/fixed_point_plugins/fixed_point_rram_patch_bit.py $result_dir/awnas/plugins/
    cp $here/fixed_point_plugins/fixed_point_rram_patch_all.py $result_dir/awnas/plugins/
fi
config=${2}
shift 2
AWNAS_HOME=$result_dir/awnas/ awnas train --gpus $gpu --train-dir $result_dir/train --seed $seed --save-every 50 $config $@
