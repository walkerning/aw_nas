set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
gpu=${GPU:-0}
cfg_file=${1}
default_exp_name=$(basename ${cfg_file})
default_exp_name=${default_exp_name%.yaml}
exp_name=${2:-${default_exp_name}}
ADDI_ARGS=${ADDI_ARGS:-}
result_dir=results_final/$exp_name

#AWNAS_HOME=$DIR/awnas/
awnas train $cfg_file --gpus $gpu --seed 123 --save-every 10 --train-dir $result_dir ${ADDI_ARGS}
