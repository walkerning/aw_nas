# Usage:
# cd $AWNAS_HOME
# export PYTHONPATH=$PYTHONPATH:`pwd`
# ./scripts/dist_train.sh 4 --train-dir $TRAIN_DIR --save-every 1 --seed 20

NUM_GPUS=$1
PORT=${PORT:-29500}
SEED=${SEED:-20}
shift 1
export PYTHONPATH=`pwd`:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port=$PORT aw_nas/main.py mptrain "$@" --seed $SEED
