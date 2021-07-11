# Transferable NAS for Few-shot Learning

## Prepare Data

### *mini*ImageNet [Vinyals et. al., NIPS 2016]

Please follow [this repository](https://github.com/yaoyao-liu/mini-imagenet-tools) to prepare the dataset as follows:

```
AWNAS_HOME/data/miniimagenet
 |-- images
 |   `-- n????????????????.jpg
 |-- train.csv
 |-- val.csv
 `-- test.csv
```

## Reproduction of Some Works

### T-NAS [Lian et. al., ICLR 2019]

Config files:

- **5-way, 1-shot T-NAS on *mini*ImageNet** tnas_5way_1shot.yaml
- **5-way, 5-shot T-NAS on *mini*ImageNet** tnas_5way_5shot.yaml

## References

- ***mini*ImageNet** Vinyals, Oriol, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. "Matching networks for one shot learning." Advances in neural information processing systems 29 (2016): 3630-3638.
- **T-NAS** Lian, Dongze, Yin Zheng, Yintao Xu, Yanxiong Lu, Leyu Lin, Peilin Zhao, Junzhou Huang, and Shenghua Gao. "Towards fast adaptation of neural architectures with meta learning." In International Conference on Learning Representations. 2019.
