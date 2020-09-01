#!/bin/bash
here=$(dirname "$0")
fixed=${FIXED:-1}
gpu=${GPU:-0}
exp_name=${1}
result_dir=$here/results/ftt_nas_mibb/$exp_name
seed=${seed:-123}

echo "use plugin dir: $result_dir/awnas/plugins"
if [[ -d "$result_dir/awnas/plugins" ]]; then
    rm -r $result_dir/awnas/plugins
fi
mkdir -p $result_dir/awnas/plugins
if [[ ! -e $result_dir/awnas/data ]]; then
    ln -s ~/awnas/data $result_dir/awnas/data
fi

if [[ $fixed -gt 0 ]]; then
    echo "copy fixed patch to plugin dir $result_dir/awnas/plugins/"
    cp $here/fixed_point_plugins/fixed_point_patch_new.py $result_dir/awnas/plugins/
fi
config=${2}
shift 2
AWNAS_HOME=$result_dir/awnas/ awnas train --gpus $gpu --train-dir $result_dir/train --seed $seed --save-every 50 $config $@
