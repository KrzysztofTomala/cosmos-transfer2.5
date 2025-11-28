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
from scripts.byoc_utils.block import BlockMeta
from scripts.byoc_utils.model import VARIANTS, ModelDimensions, ModelMeta, make_dummy_tensors
from scripts.byoc_utils.pipeline import QUANTIZATION_MODES, setup_pipeline_from_defaults
from scripts.byoc_utils.trt import (
    TRT_LOGGER,
    create_execution_context_from_pool,
    trt_engine_from_onnx_block,
    trt_set_tensor_check
)
# Import PluginV3
import packages._trt_plugins as _
from packages._trt_plugins.context_registry import set_loc_cp_ranks


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", nargs='+', choices=VARIANTS, required=True, type=str,
                        help="Model variant(s) to use for control-video-to-world generation. "
                             "Provide single variant (e.g., edge) for single control, "
                             "or multiple variants (e.g., edge vis depth seg) for multicontrol mode.")
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
    def __init__(
        self,
        root_dir: str,
        model_meta: ModelMeta,
        dims: ModelDimensions,
        control_receiving_layers: list[int],
        quant_mode: str
    ):
        self.pyt_stream = torch.cuda.current_stream()
        self.trt_stream = torch.cuda.Stream()
        self.trt_builder = trt.Builder(TRT_LOGGER)
        self.trt_runtime = trt.Runtime(TRT_LOGGER)

        cc_major, cc_minor = torch.cuda.get_device_capability()
        self.onnx_dir = os.path.join(root_dir, f"onnx_{model_meta.safe_name}_2B_{quant_mode}")
        self.trt_dir = os.path.join(root_dir, f"trt_{model_meta.safe_name}_2B_{quant_mode}", f"sm{cc_major}{cc_minor}")
        self.onnx_path = os.path.join(self.onnx_dir, "{block_label}.onnx")
        self.engine_path = os.path.join(self.trt_dir, "{block_label}.trt")

        self.model_meta = model_meta
        self.model_dims = dims
        self.control_receiving_layers = control_receiving_layers

    def build(self, optimization_level: int, test_engines: bool):
        assert os.path.exists(self.onnx_dir), f"Missing ONNX source folder: {self.onnx_dir}"
        os.makedirs(self.trt_dir, exist_ok=True)

        for block_index in tqdm.trange(self.model_dims.BK, disable=False, desc="Processing base block"):
            receives_control = block_index in self.control_receiving_layers
            block_meta = BlockMeta(block_index, is_control=False, receives_control=receives_control)
            self._process_block(block_meta, optimization_level, test_engine=test_engines)
        if self.model_meta.is_multicontrol:
            for control_branch in range(len(self.model_meta.variants)):
                for block_index in tqdm.trange(
                        self.model_dims.BC, disable=False, desc=f"Processing control block ({control_branch=})"):
                    block_meta = BlockMeta(
                        block_index, is_control=True, receives_control=False, control_branch=control_branch)
                    self._process_block(block_meta, optimization_level, test_engine=test_engines)
        else:
            for block_index in tqdm.trange(self.model_dims.BC, disable=False, desc="Processing control block"):
                block_meta = BlockMeta(block_index, is_control=True, receives_control=False)
                self._process_block(block_meta, optimization_level, test_engine=test_engines)

    def _process_block(self, meta: BlockMeta, optimization_level: int, test_engine: bool):
        onnx_file = self.onnx_path.format(block_label=meta.block_label)
        engine_file = self.engine_path.format(block_label=meta.block_label)

        # Build
        engine_serialized = trt_engine_from_onnx_block(
            self.trt_builder,
            onnx_file,
            block_meta=meta,
            dims=self.model_dims,
            optimization_level=optimization_level,
        )

        # Test
        if test_engine:
            log.info("Testing TRT engine")
            self._test_block_engine(engine_serialized, meta)
        else:
            log.info("Testing TRT engine: skipped")

        # Save
        log.info("Saving TRT engine")
        with open(engine_file, "wb") as f:
            f.write(engine_serialized)
        log.info(f"Engine saved to {engine_file}")

    def _test_block_engine(self, engine, meta: BlockMeta):
        engine = self.trt_runtime.deserialize_cuda_engine(engine)
        context = create_execution_context_from_pool(engine)

        register_input = partial(trt_set_tensor_check, context, check_shape=True)
        register_output = partial(trt_set_tensor_check, context, check_shape=False)

        dummy_tensors = make_dummy_tensors(self.model_dims, with_outputs=True)
        set_loc_cp_ranks([0])

        for key in meta.fixed_inputs:
            register_input(key, dummy_tensors[key])
        if meta.is_control:
            if meta.block_index == 0:
                c = dummy_tensors["control_B_T_H_W_D"]
            else:
                c = dummy_tensors["output_hints"][:meta.block_index+1]
            register_input("c", c)
            register_output('output', dummy_tensors["output_hints"][:meta.block_index+2])
        else:
            register_output('output', dummy_tensors["output_B_T_H_W_D"])

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
    control_receiving_layers = pipe.model.net.control_layers
    del pipe

    builder = CosmosTRTEngineBuilder(
        args.output_dir, args.model, dims, control_receiving_layers, cmdargs.mode)
    builder.build(optimization_level=cmdargs.optimization_level, test_engines=not cmdargs.skip_testrun)


if __name__ == "__main__":
    main(make_parser().parse_args())
