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

All experiments are conducted with 8-bit quantization. We use a patch-based quantization library [nics_fix_pytorch](https://github.com/walkerning/nics_fix_pytorch), you can install the compatible version of `nics_fix_pytorch` by:
```
pip install git+git://github.com/walkerning/nics_fix_pytorch.git@9b97b9402521577cf40910ba4f18c790abe5319f
```

Note that since the quantization is simulated, it makes the search and final process much slower.

### feature fault model (MiBB)
The quantization patch of MiBB model is in `examples/research/ftt-nas/fixed_point_plugins/fixed_point_patch_new.py`. And the following scripts will copy this patch into the plugin directory of `aw_nas`. This patch will quantize all the weights before each call of `forward` or `forward_one_step_callback`. And the every feature map is quantized in `aw_nas.objective.fault_injection:FaultInjectionObjective.inject`, which will be called by `forward_one_step_callback`.

MiBB fault injection is conducted in `aw_nas.objective.fault_injection.FaultInjectionObjective.inject`.

*Feel free to ignore this note* An unimportant note is that, there is another patch named `fixed_point_patch.py` that patches `nn.Conv2d` and `nn.Linear` modules directly. During our experiments, we find the previous patch method in `fixed_point_patch_new.py` is faster (see the comments in the patch), thus we use the `fixed_point_patch_new.py` patch.

#### Search
```
FIXED=1 bash ./examples/research/ftt-nas/run_mibb_search.sh ./examples/research/ftt-nas/mibb.yaml
```
Use `GPU=1 ...` to run on different GPUs.

#### Final training
`bash ./examples/research/ftt-nas/run_mibb.sh [exp name] [final config] --load_state_dict {state_dict}`
`(option) GPU=x seed=x fixed=x(0/1)`

### weight fault model (adSAF)

Different from the MiBB model, the quantization and fault injection under adSAF fault model are all conducted in the `fixed_point_plugins/fixed_point_rram_patch*.py` patches. These two patches are slightly different.
* The `fixed_point_plugins/fixed_point_rram_patch_all.py` patch adds differently-shifted biases onto the weights. This is only an approximation of bit-stucks in RRAM cells.
* The `fixed_point_plugins/fixed_point_rram_patch_bit.py` patch employs bitwise operations, which corresponds better to the hardware faults.

The experiments in the paper are conducted with the `_all.py` patch.

#### Search
```
bash ./examples/research/ftt-nas/run_adsaf_search.sh ./examples/research/ftt-nas/adsaf.yaml
```

#### Final training
`bash ./examples/research/ftt-nas/run_adsaf.sh adsaf_final ./examples/research/ftt-nas/adsaf_final.yaml`
You can add optional environment variables such as `GPU=x seed=x`. Optionally, can add other arguments such as `--load_state_dict {state_dict}`

Because the FTT-NAS experiments are conducted using the commit `27d1aeb4121c320ed11361b705`, I have adapted the `adsaf_final.yaml` configuration to the current master `d9d0ba26870b009778f2209f22fde876c0e55aa2`. But I'm not sure whether there are other sutble changes that would make the results differ. If you find that you cannot reproduce the results with the latest code, you can contact us by email or issue.