# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import os
from functools import partial

import tensorrt as trt
import torch
import tqdm

from cosmos_transfer2._src.imaginaire.utils import log
from scripts.byoc_utils.model import (
    FIXED_CONTROL_INPUTS,
    FIXED_INPUTS,
    VARIANTS,
    ModelDimensions,
    ModelMeta,
    make_dummy_tensors,
)
from scripts.byoc_utils.pipeline import QUANTIZATION_MODES, setup_pipeline_from_defaults
from scripts.byoc_utils.trt import (
    TRT_LOGGER,
    create_execution_context_from_pool,
    trt_engine_from_onnx_block,
    trt_set_tensor_check
)

BLOCK_FILE = "cosmos_transfer2.5_{block_type}_block{block_index}.{ext}"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", choices=VARIANTS, required=True, type=str,
                        help="Model variant to use for control-video-to-world generation")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Working directory of where to retrieve ONNX files from and export TRT engines to.")
    parser.add_argument("--mode", type=str, choices=list(QUANTIZATION_MODES.keys()), default="FP8",
                        help="Quantization mode (FP8 or NVFP4)")
    parser.add_argument("-O", dest="optimization_level", type=int, default=3, help="TRT optimization level")
    parser.add_argument("--skip_testrun", action="store_true", help="Skip testrun")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    return parser


class CosmosTRTEngineBuilder:
    def __init__(self, root_dir: str, model_meta: ModelMeta, dims: ModelDimensions, resolution: str, quant_mode: str):
        self.pyt_stream = torch.cuda.current_stream()
        self.trt_stream = torch.cuda.Stream()
        self.trt_builder = trt.Builder(TRT_LOGGER)
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        cc_major, cc_minor = torch.cuda.get_device_capability()
        self.onnx_dir = os.path.join(root_dir, f"onnx_{model_meta.safe_name}_2B_{quant_mode}")
        self.trt_dir = os.path.join(root_dir, f"trt_{model_meta.safe_name}_2B_{quant_mode}", f"sm{cc_major}{cc_minor}")
        self.onnx_path = os.path.join(self.onnx_dir, BLOCK_FILE)
        self.engine_path = os.path.join(self.trt_dir, BLOCK_FILE)

        self.model_dims = dims
        self.low_res = resolution == "480"

    def build(self, optimization_level: int, test_engines: bool):
        assert os.path.exists(self.onnx_dir), f"Missing ONNX source folder: {self.onnx_dir}"
        os.makedirs(self.trt_dir, exist_ok=True)

        for block_index in tqdm.trange(self.model_dims.BK, disable=False, desc="Processing base block"):
            self._process_block(block_index, optimization_level, test_engine=test_engines, is_control=False)
        for block_index in tqdm.trange(self.model_dims.BC, disable=False, desc="Processing control block"):
            self._process_block(block_index, optimization_level, test_engine=test_engines, is_control=True)

    def _process_block(self, block_index: int, optimization_level: int, test_engine: bool, is_control: bool):
        block_type = "controlnet" if is_control else "net"
        onnx_file = self.onnx_path.format(block_type=block_type, block_index=block_index, ext='onnx')
        engine_file = self.engine_path.format(block_type=block_type, block_index=block_index, ext='trt')

        # Build
        engine_serialized = trt_engine_from_onnx_block(
            self.trt_builder,
            onnx_file,
            block_index,
            is_control,
            dims=self.model_dims,
            optimization_level=optimization_level,
        )

        # Test
        if test_engine:
            log.info("Testing TRT engine")
            self._test_block_engine(engine_serialized, block_index, is_control)
        else:
            log.info("Testing TRT engine: skipped")

        # Save
        log.info("Saving TRT engine")
        with open(engine_file, "wb") as f:
            f.write(engine_serialized)
        log.info(f"Engine saved to {engine_file}")

    def _test_block_engine(self, engine, block_index: int, is_control: bool = True):
        engine = self.trt_runtime.deserialize_cuda_engine(engine)
        context = create_execution_context_from_pool(engine)

        register_input = partial(trt_set_tensor_check, context, check_shape=True)
        register_output = partial(trt_set_tensor_check, context, check_shape=False)

        dummy_tensors = make_dummy_tensors(self.model_dims, with_outputs=True)

        if is_control:
            c = dummy_tensors["control_B_T_H_W_D"] if block_index == 0 else dummy_tensors["output_hints"][:block_index+1]
            register_input("c", c)
            for key in FIXED_CONTROL_INPUTS:
                register_input(key, dummy_tensors[key])
            register_output('output', dummy_tensors["output_hints"][:block_index+2])
        else:
            for key in FIXED_INPUTS:
                register_input(key, dummy_tensors[key])
            register_output('output', dummy_tensors["output_hints"])

        self.trt_stream.wait_stream(self.pyt_stream)
        context.execute_async_v3(self.trt_stream.cuda_stream)
        self.pyt_stream.wait_stream(self.trt_stream)

        del context
        del engine
        del dummy_tensors
        gc.collect()
        torch.cuda.empty_cache()


def main(cmdargs):
    pipe, args, dims = setup_pipeline_from_defaults({
        "model_variant": cmdargs.model_variant,
        "output_dir": cmdargs.output_dir,
        "resolution": cmdargs.resolution,
        "disable_guardrail": True,
    })
    del pipe

    builder = CosmosTRTEngineBuilder(args.output_dir, args.model, dims, cmdargs.resolution, cmdargs.mode)
    builder.build(optimization_level=cmdargs.optimization_level, test_engines=not cmdargs.skip_testrun)


if __name__ == "__main__":
    main(make_parser().parse_args())
