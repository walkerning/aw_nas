"""
Generate ResNet baseline cfg yaml files.
"""
import os
import sys
import yaml

here = os.path.dirname(__file__)

def generate_res(primitive_name, num_layers, expansion=1,
                 stage_channels=[64, 128, 256, 512], stage_strides=[1, 2, 2, 2],
                 init_channels=64, wide=False, **kwargs):
    with open(os.path.join(here, "resnet50.yaml"), "r") as r_f:
        template_cfg = yaml.load(r_f)

    # fill in primtive name
    if not wide:
        template_cfg["final_model_cfg"]["genotypes"] = \
                            "normal_0=[('{}', 0, 1)], reduce_1=[('{}', 0, 1)]".format(
                                primitive_name, primitive_name)
    else:
        template_cfg["final_model_cfg"]["genotypes"] = \
            "normal_0=[('{}', 0, 1)], reduce_1=[('{}', 0, 1)], normal_2=[('bn_relu', 0, 1)]".format(
                primitive_name, primitive_name)

    # calculate layer_channels
    layer_channels = [init_channels]
    stage_channels = [n * expansion for n in stage_channels]
    assert len(stage_channels) == len(stage_strides)
    cell_layout = []
    for i_stage, stage_num_layer in enumerate(num_layers):
        layer_channels += [stage_channels[i_stage]] * stage_num_layer
        num_stride_layer = int(stage_strides[i_stage] > 1)
        cell_layout += [1] * num_stride_layer + [0] * (stage_num_layer - num_stride_layer)
    if wide:
        # add a BNReLU layer after wide resnet
        cell_layout.append(2)
        layer_channels.append(layer_channels[-1])
        template_cfg["search_space_cfg"]["num_cell_groups"] = 3 # cell group for bn_relu
    template_cfg["search_space_cfg"]["cell_layout"] = cell_layout
    template_cfg["search_space_cfg"]["num_layers"] = len(cell_layout)
    template_cfg["final_model_cfg"]["layer_channels"] = layer_channels
    template_cfg["final_model_cfg"]["init_channels"] = init_channels

    # fill in other fields (e.g., use_stem for wideresnet)
    template_cfg["final_model_cfg"].update(kwargs)
    return template_cfg

# ---- adapted resnets from imagenet ----
def generate_res18():
    return generate_res("resnet_block", [2, 2, 2, 2])

def generate_res34():
    return generate_res("resnet_block", [3, 4, 6, 3])

def generate_res50():
    return generate_res("resnet_bottleneck_block", [3, 4, 6, 3], expansion=4)

def generate_res101():
    return generate_res("resnet_bottleneck_block", [3, 4, 23, 3], expansion=4)

def generate_res152():
    return generate_res("resnet_bottleneck_block", [3, 8, 36, 3], expansion=4)

# ---- downsample resnets on cifar-10----
def generate_res20():
    return generate_res("resnet_block_pool_downsample",
                        [3, 3, 3], stage_channels=[16, 32, 64], stage_strides=[1, 2, 2],
                        init_channels=16)
def generate_res32():
    return generate_res("resnet_block_pool_downsample",
                        [5, 5, 5], stage_channels=[16, 32, 64], stage_strides=[1, 2, 2],
                        init_channels=16)
def generate_res44():
    return generate_res("resnet_block_pool_downsample",
                        [7, 7, 7], stage_channels=[16, 32, 64], stage_strides=[1, 2, 2],
                        init_channels=16)
def generate_res56():
    return generate_res("resnet_block_pool_downsample",
                        [9, 9, 9], stage_channels=[16, 32, 64], stage_strides=[1, 2, 2],
                        init_channels=16)
def generate_res110():
    return generate_res("resnet_block_pool_downsample",
                        [18, 18, 18], stage_channels=[16, 32, 64], stage_strides=[1, 2, 2],
                        init_channels=16)

# ---- wide resnets ----
def generate_wideres_28_10():
    return generate_res(
        "wideresnet_block_3x3", [4, 4, 4],
        stage_channels=[160, 320, 640], stage_strides=[1, 2, 2],
        use_stem="conv_3x3", init_channels=16, wide=True)


if __name__ == "__main__":
    cfg = globals()["generate_{}".format(sys.argv[1])]()
    with open(sys.argv[2], "w") as w_f:
        yaml.dump(cfg, w_f)
