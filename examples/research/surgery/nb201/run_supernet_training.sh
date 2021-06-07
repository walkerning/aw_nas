set -e

gpu=${GPU:-0}
cfg_file=${1}
default_exp_name=$(basename ${cfg_file})
default_exp_name=${default_exp_name%.yaml}
seed=${2}
exp_name=${3:-${default_exp_name}_seed${seed}}
ADDI_ARGS=${ADDI_ARGS:-}
save_every=${SAVE_EVERY:-20}
result_base_dir=${4:-./results/nb201/results_supernet_training/}

awnas search $cfg_file --gpu $gpu --seed $seed --save-every ${save_every} --train-dir $result_base_dir/$exp_name ${ADDI_ARGS}
