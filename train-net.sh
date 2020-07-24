gpu=${1:-0}

echo $gpu

awnas train examples/baselines/bireal_resnet18.yaml --train-dir ./logs/bireal_0/ --gpus ${gpu:-0}
# awnas train examples/baselines/resnet18.yaml --train-dir ./logs/test_0/ --gpus 0
