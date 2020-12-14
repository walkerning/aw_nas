# -*- coding: utf-8 -*-

import logging
import os
import pickle
import re
import sys
import shutil
import subprocess
from collections import namedtuple

import yaml
import torch
from torch.autograd import Variable

from aw_nas import utils
from aw_nas.main import _init_component
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.hardware.base import BaseHardwareCompiler
from aw_nas.utils.log import LEVEL as _LEVEL
from aw_nas.rollout.general import GeneralSearchSpace
from aw_nas.hardware.utils import Prim

try:
    from aw_nas.utils.pytorch2caffe import pytorch_to_caffe
except:
    pytorch_to_caffe = None

CAFFE_DATA_LAYER_STR = """
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    crop_size: INPUT_SIZE
    mean_value: 103.53
    mean_value: 116.28
    mean_value: 123.675
    use_standard_std: true
  }
  image_data_param {
    source: "/datasets/imgNet/imagenet1k_valid_source.txt"
    root_folder: "/datasets/imgNet/imagenet1k_valid_dataset/"
    batch_size: 50
    new_height: 256
    new_width: 256
  }
}
"""


class DPUCompiler(BaseHardwareCompiler):
    """
    A hardware interface class for Xilinx DPU (CNN accelerator).
    """

    NAME = "dpu"

    def __init__(self, dcf=None, mode="debug", calib_iter=0, gpu=0, input_size=None):
        super(DPUCompiler, self).__init__()

        expect(input_size is not None, "must specificy `input_size`", ConfigException)
        self.dcf = dcf
        self.mode = mode
        self.calib_iter = calib_iter
        self._debug_output = _LEVEL <= logging.DEBUG  # debug output
        self.gpu = gpu
        self.input_size = input_size

    def _run_pytorch_to_caffe(self, model, name, output_dir, input_size, debug):
        self.logger.info("-------- Run pytorch to caffe --------")
        inputs = Variable(torch.ones([1, 3, input_size, input_size]))

        if not debug:
            backup_stdout = sys.stdout
            sys.stdout = open("/dev/null", "w")
        pytorch_to_caffe.trans_net(model, inputs, name)
        if not debug:
            sys.stdout = backup_stdout

        utils.makedir(output_dir)
        out_proto = "{}/{}.prototxt".format(output_dir, name)
        out_caffemodel = "{}/{}.caffemodel".format(output_dir, name)
        out_torch_to_caffe = "{}/{}_torch2caffe.pkl".format(output_dir, name)
        pytorch_to_caffe.save_prototxt(out_proto)
        pytorch_to_caffe.save_caffemodel(out_caffemodel)
        with open(out_proto, "r") as fr:
            a = fr.read()
        a = a.replace(
            'input: "blob1"\ninput_dim: 1\ninput_dim: 3\ninput_dim: %d\ninput_dim: %d'
            % (input_size, input_size),
            'layer {\n\tname: "blob1"\n\ttype: "Input"\n\ttop: "blob1"\n\tinput_param {\n\t\tshape {\n\t\t\tdim: 1\n\t\t\tdim: 3\n\t\t\tdim: %d\n\t\t\tdim: %d\n\t\t}\n\t}\n}'
            % (input_size, input_size),
        )
        with open(out_proto, "w") as fw:
            fw.write(a)
        with open(out_torch_to_caffe, "wb") as fw:
            pickle.dump(
                pytorch_to_caffe.torch_to_caffe_names, fw, pickle.HIGHEST_PROTOCOL
            )
        self.logger.info(
            "Finish convert pytorch model to caffe, check {}, {} and {}.".format(
                out_proto, out_caffemodel, out_torch_to_caffe
            )
        )
        return out_proto, out_caffemodel, pytorch_to_caffe.torch_to_caffe_names

    def _caffe_fix(
        self, prototxt, caffemodel, output_dir, gpu, calib_iter, input_size, debug
    ):
        self.logger.info("-------- Run caffe deephi_fix --------")
        ## Modify the data layer in the input prototxt
        # As anyway dnnc's inner caffe verrsion do not support `ceil_mode`,
        # we just remove this config here.

        # And just use the caffe version installed using conda (same with PytorchToCaffe)
        # Might cause some archs to end up with wrong output shape.
        # e.g. Resnet50 converted from Pytorch
        input_prototxt = prototxt + ".tofix.prototxt"
        subprocess.check_call(
            (
                "cat {} | sed '/ceil_mode/d' | sed '/input_dim/d' |"
                " sed '/input:/d' | sed 's/\"blob1\"/\"data\"/' > {}"
            ).format(prototxt, input_prototxt),
            shell=True,
        )
        with open(input_prototxt, "r") as r_f:
            content = (
                CAFFE_DATA_LAYER_STR.replace("INPUT_SIZE", str(input_size)) + r_f.read()
            )
        with open(input_prototxt, "w") as w_f:
            w_f.write(content)
        self.logger.info(
            "Fixed-point input prototxt saved to {}.".format(input_prototxt)
        )

        ## fixpoint
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "run_fix.log")
        self.logger.info(
            "Running deephi_fix, log will be saved to {}.".format(log_file)
        )
        with open(log_file, "w") as logf:
            subprocess.check_call(
                (
                    "/home/foxfi/projects/caffe_dev/build/tools/deephi_fix fix -calib_iter {} "
                    "-gpu {} -model {} -weights {} -output_dir {}"
                ).format(calib_iter, gpu, input_prototxt, caffemodel, output_dir),
                shell=True,
                stdout=logf,
                stderr=logf,
            )
        self.logger.info(
            "Finish running deephi_fix, check output dir {}.".format(output_dir)
        )

        ## modify the generated deploy.prototxt to be compatible with dnnc
        output_prototxt = os.path.join(output_dir, "deploy.prototxt")
        mod_output_prototxt = os.path.join(output_dir, "deploy_dnnc.prototxt")
        output_caffemodel = os.path.join(output_dir, "deploy.caffemodel")
        shutil.copy(output_prototxt, mod_output_prototxt)

        subprocess.check_call(
            "python modify_for_dnnc.py {}".format(mod_output_prototxt), shell=True
        )

        self.logger.info(
            "Finish generating dnnc-compatible prototxt: {}, weights: {}.".format(
                mod_output_prototxt, output_caffemodel
            )
        )
        return mod_output_prototxt, output_caffemodel

    def _run_dnnc(self, name, prototxt, caffemodel, output_dir, dcf, mode, debug=False):
        self.logger.info("-------- Run dnnc --------")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        expect(dcf is not None, "must specificy dcf file", ConfigException)
        subprocess.check_call(
            (
                "dnnc --mode {mode} --cpu_arch arm64 --save_kernel --prototxt {prototxt}"
                " --caffemodel {caffemodel}  --output_dir {output_dir} --dcf {dcf} "
                "--net_name {name}{debug_cmd}"
            ).format(
                name=name,
                prototxt=prototxt,
                caffemodel=caffemodel,
                output_dir=output_dir,
                dcf=dcf,
                mode=mode,
                debug_cmd=" --dump=all" if debug else "",
            ),
            shell=True,
        )
        output_elf = os.path.join(output_dir, "dpu_{}.elf".format(name))
        self.logger.info(
            "Finish running dnnc for {} (mode: {}), elf file: {}.".format(
                name, mode, output_elf
            )
        )
        return output_elf

    def compile(self, compile_name, net_cfg, result_dir):
        # construct aw_nas final model

        if pytorch_to_caffe is None:
            self.logger.warn("the submodule pytorch_to_caffe does not exists.")
            return

        search_space = _init_component(net_cfg, "search_space")
        assert isinstance(search_space, GeneralSearchSpace)
        model = _init_component(
            net_cfg,
            "final_model",
            search_space=search_space,
            device="cuda:{}".format(self.gpu),
        )

        rollout = search_space.rollout_from_genotype(
            net_cfg["final_model_cfg"]["genotypes"]
        )

        # pytorch to caffe
        input_size = self.input_size
        ptc_out_dir = utils.makedir(os.path.join(result_dir, "pytorch_to_caffe"))
        proto, caffemodel, torch_to_caffe = self._run_pytorch_to_caffe(
            model,
            compile_name,
            ptc_out_dir,
            input_size=input_size,
            debug=self._debug_output,
        )

        # map prims to torch layers, and combining with torch layer to caffe layer name.
        prims = rollout.genotype_list()
        prims_to_torch_layers = {}
        for idx, prim in enumerate(prims):
            torch_layer_names = list(model.layer_idx_to_named_modules(idx))
            prims_to_torch_layers[Prim(**prim)] = torch_layer_names

        prims_to_caffe_name = {}
        for prim, torch_layers in prims_to_torch_layers.items():
            prims_to_caffe_name[prim] = [
                torch_to_caffe[t] for t in torch_layers if t in torch_to_caffe
            ]
        with open("{}/{}_prim2names.pkl".format(ptc_out_dir, compile_name), "wb") as fw:
            pickle.dump(prims_to_caffe_name, fw, pickle.HIGHEST_PROTOCOL)

        try:
            # caffe fix
            fix_out_dir = os.path.join(result_dir, "fix")
            proto, caffemodel = self._caffe_fix(
                proto,
                caffemodel,
                fix_out_dir,
                self.gpu,
                self.calib_iter,
                input_size,
                debug=self._debug_output,
            )

            # dnnc
            dnnc_out_dir = os.path.join(result_dir, "dnnc_{}".format(self.mode))
            self._run_dnnc(
                compile_name,
                proto,
                caffemodel,
                dnnc_out_dir,
                self.dcf,
                self.mode,
                debug=self._debug_output,
            )
        except Exception as e:
            self.logger.error(str(e))

        return proto, caffemodel, prims_to_caffe_name

    def parse_file(
        self,
        prof_result_dir,
        prof_prim_file,
        prim_to_ops_file,
        result_dir,
        perf_fn=None,
        perf_names=("latency",),
    ):
        # prof_result consists of all basic operators ,like conv_bn_relu, pooling and concat
        """
        prof_result_dir: the measurement result file
        prof_prim_file: all primitives that need to be profiled.
        prim_to_ops_file: primitive to the names of caffe layers file
        """

        # load prims to op dict
        prim_to_ops = dict()
        for _dir in os.listdir(prim_to_ops_file):
            cur_dir = os.path.join(prim_to_ops_file, _dir, 'pytorch_to_caffe')
            for _file in os.listdir(cur_dir):
                if not _file.endswith('prim2names.pkl'): 
                    continue
                pkl = os.path.join(cur_dir, _file)
                with open(pkl, "rb") as r_f:
                    _dict = pickle.load(r_f)
                    prim_to_ops.update(_dict)
                    self.logger.info("Unpickled file: {pkl}".format(pkl=pkl))

        # meta info: prim_to_ops is saved when generate profiling final-yaml folder for each net
        for _dir in os.listdir(prof_result_dir):
            cur_dir = os.path.join(prof_result_dir, _dir)
            if os.path.isdir(cur_dir):
                parsed_dir = os.path.join(result_dir, _dir)
                os.makedirs(parsed_dir, exist_ok=True)
                for i, _file in enumerate(os.listdir(cur_dir)):
                    perf_yaml = self.parse_one_network(
                        os.path.join(cur_dir, _file), prof_prim_file, prim_to_ops)
                    if perf_yaml:
                        with open(os.path.join(parsed_dir, i + ".yaml"),
                                "w") as fw:
                            yaml.safe_dump(perf_yaml, fw)
    
    def parse_one_network(
        self,
        prof_result_file,
        prof_prim_file,
        prim_to_ops,
        perf_fn=None,
        perf_names=("latency",)
    ):
        # parse result file
        profiled_yaml = {"overall_{}".format(k): 0. for k in perf_names}
        if perf_fn is None:
            perf_fn = lambda split_line: (
                split_line[0],
                {"latency": float(split_line[3])},
            )
        name_to_perf_dict = {}
        Perf = namedtuple("Perf", perf_names)
        with open(prof_result_file, "r") as fl:
            fl_lines = fl.readlines()
            for line in fl_lines[3:-3]:
                if "Total Time" in line:
                    profiled_yaml["overall_latency"] = int(re.findall(r"(\d+)", a[-5])[0])
                    continue
                split_line = re.split(r"\s+", line)
                if len(split_line) > 4:
                    name, performance = perf_fn(split_line)
                    if not name.startswith("NodeName"):
                        if name not in name_to_perf_dict:
                            name_to_perf_dict[name] = Perf([], [])
                        for perf in perf_names:
                            getattr(name_to_perf_dict[name], perf).append(
                                performance[perf]
                                    )
        name_to_perf_dict = {
            k: Perf(
                *[sum(getattr(v, perf)) / len(getattr(v, perf)) for perf in perf_names]
            )
            for k, v in name_to_perf_dict.items()
        }
        # mapping name of op to performances
        # {conv2: Perf(latency=12., memory: 2048), ...}

        with open(prof_prim_file, "r") as fr:
            prof_prim = yaml.load(fr)

        nets_prim_to_perf = []
        for prim in prof_prim:
            # prim consists of prim_type, input_ch, output_ch, kernel, stride
            # Now, use prim_to_ops mapping prim into basic ops' names
            # Using function instead of dict to handle exceptions
            names = prim_to_ops(Prim(**prim))
            if len(set.intersection(names, name_to_perf_dict)) == 0:
                self.logger.debug("prims {} is not measured.".format(prim))
                continue
            for perf in perf_names:
                prim[perf] = sum(
                    [
                        getattr(name_to_perf_dict[name], perf)
                        for name in names
                        if name in name_to_perf_dict
                    ]
                )
            nets_prim_to_perf.append(prim)
        profiled_yaml["primitives"] = nets_prim_to_perf
        return profiled_yaml
