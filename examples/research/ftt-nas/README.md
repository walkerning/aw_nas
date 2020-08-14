## FTT-NAS: Discovering Fault-Tolerant Neural Architecture

If you find this work/repo helpful, please cite:
```
@article{ning2020ftt,
   title={FTT-NAS: Discovering Fault-Tolerant Neural Architecture},
   author={Ning, Xuefei and Ge, Guangjun and Li, Wenshuo and Zhu, Zhenhua and Zheng, Yin and Chen, Xiaoming and Gao, Zhen and Wang, Yu and Yang, Huazhong},
   journal={arXiv preprint arXiv:2003.10375},
   year={2020}
}
```

#### feature fault model (MiBB)

`bash ./examples/research/ftt-nas/run_mibb.sh [exp name] ./examples/research/ftt-nas/mibb.yaml --load_state_dict {state_dict}`
`(option) GPU=x seed=x fixed=x(0/1)`

#### weight fault model (adSAF)

`bash ./examples/research/ftt-nas/run_adsaf.sh [exp name] ./examples/research/ftt-nas/adsaf.yaml --load_state_dict {state_dict}`
`(option) GPU=x seed=x`
