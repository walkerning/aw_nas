set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
gpu=${GPU:-0}
cfg_file=${1}
default_exp_name=$(basename ${cfg_file})
default_exp_name=${default_exp_name%.yaml}
exp_name=${2:-${default_exp_name}}
ADDI_ARGS=${ADDI_ARGS:-}
result_base_dir=${3:-results_search}

# AWNAS_HOME=${DIR}/awnas/
awnas search $cfg_file --gpu $gpu --seed 123 --save-every 1 --train-dir $result_base_dir/$exp_name ${ADDI_ARGS}
