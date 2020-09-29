Black Box Search Space Profiling for Accelerator-Aware Neural Architecture Search, in ASP-DAC 2020
--------

If you find this work/code helpful, please cite:
```
@article{zeng2020blackbs,
  title={Black Box Search Space Profiling for Accelerator-Aware Neural Architecture Search},
  author={Shulin Zeng and Hanbo Sun and Y. Xing and Xuefei Ning and Y. Shan and X. Chen and Yu Wang and Huazhong Yang},
  journal={2020 25th Asia and South Pacific Design Automation Conference (ASP-DAC)},
  year={2020},
  pages={518-523}
}
```

> See [here](https://github.com/patrick22414/ssprofile) for the codes of the search space profiling phase.

### Black box search space profiling

This repo doesn't include the scripts and codes of black box search space profiling. Instead, we provide several search space designs after profiling and the policy-aware latency LUT for Xilinx DPU.

### Search on the optimized search space

First, we need to set up the plugins for defining different neural network blocks.

`cp -r ssp_plugin/ ~/awnas/plugins/`

#### Differentiable method

`awnas search --gpu 0 --seed 123 --save-every 50 --train-dir diff_result/cifar10_allblock diff_config/diff_cifar10_allblock.yaml`

There is only one yaml file for differentiable method in the `diff_config` repo. You can find all the yaml files of different optimized search space and experimental settings in the `diff_config` from [this url](https://cloud.tsinghua.edu.cn/f/9f2dbf9f9c514981aaa3/?dl=1). 

#### Reinforcement learning method

`awnas search --gpu 0 --seed 123 --save-every 50 --train-dir rl_result/cifar10_allblock rl_config/rl_cifar10_allblock.yaml`

There is only one yaml file for reinforcement learning method in the `rl_config` repo. You can find all the yaml files of different optimized search space and experimental settings in the `rl_config` from [this url](https://cloud.tsinghua.edu.cn/f/9f2dbf9f9c514981aaa3/?dl=1).