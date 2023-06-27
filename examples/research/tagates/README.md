## TA-GATES: An Encoding Scheme for Neural Network Architectures, in NeurIPS 2022
--------

If you find this work/repo helpful, please cite:
```
@article{ning2022ta,
    title={TA-GATES: An Encoding Scheme for Neural Network Architectures},
    author={Ning, Xuefei and Zhou, Zixuan and Zhao, Junbo and Zhao, Tianchen and Deng, Yiping and Tang, Changcheng and Liang, Shuang and Yang, Huazhong and Wang, Yu},
    journal={Advances in Neural Information Processing Systems},
    year={2022}
}
```

There are several python files and cfgs` dir under this directory. The python files are:

* `common_utils.py`: Dataset defination and p@topk calculation.
* `train_nasbench*01.py`: Conduct TA-GATES predictor training on NAS-Bench-101 / NAS-Bench-201 / NAS-Bench-301.
* `train_enas.py`: Conduct TA-GATES predictor training on ENAS search space.
* `train_nasbench*01_anytime.py`: Conduct anytime TA-GATES predictor training on NAS-Bench-101 / NAS-Bench-201 / NAS-Bench-301.

### Environmental Setup
First, please install `awnas`.

And then, please install the extra packages below: 
* [nasbench](https://github.com/google-research/nasbench): Package of NAS-Bench-101.
* [nas-bench-201](https://github.com/D-X-Y/NAS-Bench-201): Package of NAS-Bench-201.
* [nasbench301](https://github.com/automl/nasbench301): Package of NAS-Bench-301.

### Data Preparation
Download the data from [here](https://drive.google.com/drive/folders/1vvhXz5_vE2Lh3zFzRsuVUaKsb0fAZ7Qz?usp=sharing).

### Run Predictor Training
To train the proposed TA-GATES or anytime TA-GATES predictor, use the provided scriptS. Specifically, run `python <SCRIPT> <CFG_FILE> --train-ratio <TRAIN_RATIO> --train-pkl <TRAIN_PKL> --valid-pkl <VALID_PKL> --seed <SEED> --gpu <GPU_ID> --train-dir <TRAIN_DIR>`, where:

* `SCRIPT`: `train_nasbench*01.py` or `train_enas.py` or `train_nasbench*01_anytime.py`
* `CFG_FILE`: Path of the configuration file
* `TRAIN_RATIO`: Proportion of training samples used in the training
* `TRAIN_PKL`: Path of the training data
* `VALID_PKL`: Path of the validation data
* `SEED`: Seed (optional)
* `GPU_ID`: ID of the used GPU. Currently, we only support single-GPU training. Default: 0
* `TRAIN_DIR`: Path to save the logs and results

We provide example predictor training configuration files under `./cfgs`, including:

* `nb101_cfgs`:
  * `tagates.yaml`: Train the 4-step TA-GATES on NAS-Bench-101 with ranking loss.
  * `tagates_anytime.yaml`: Train the 2-step anytime TA-GATES on NAS-Bench-101 with regression loss.
* `nb201_cfgs`:
  * `tagates.yaml`: Train the 4-step TA-GATES on NAS-Bench-201 with ranking loss.
  * `tagates_anytime.yaml`: Train the 2-step anytime TA-GATES on NAS-Bench-201 with regression loss.
* `nb301_cfgs`:
  * `tagates.yaml`: Train the 4-step TA-GATES on NAS-Bench-301 with ranking loss.
  * `tagates_anytime.yaml`: Train the 2-step anytime TA-GATES on NAS-Bench-301 with regression loss.
* `enas_cfgs`:
  * `tagates.yaml`: Train the 2-step TA-GATES on ENAS with ranking loss.


For example, run TA-GATES predictor training on NAS-Bench-201 with 100% training samples by

`python train_nasbench201.py cfgs/nb201_cfgs/tagates.yaml --gpu <GPU_ID> --seed <SEED> --train-dir <TRAIN_DIR> --train-pkl ./data/NAS-Bench-201/nasbench201_zsall_train.pkl --valid-pkl ./data/NAS-Bench-201/nasbench201_zsall_valid.pkl --train-ratio 1.`.

Run anytime TA-GATES predictor training on NAS-Bench-301 with 10% training samples by

`python train_nasbench301_anytime.py cfgs/nb301_cfgs/tagates_anytime.yaml --gpu <GPU_ID> --seed <SEED> --train-dir <TRAIN_DIR> --train-pkl ./data/NAS-Bench-301/nasbench301_zsall_anytime_train.pkl --valid-pkl ./data/NAS-Bench-301/nasbench301_zsall_anytime_valid.pkl --train-ratio 0.1`.

