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
import os

import modelopt.torch.opt as mto
import torch
import torch.onnx
import tqdm

from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
    ControlAwareDiTBlock,
    ControlEncoderDiTBlock,
)
from scripts.byoc_utils.block import BlockMeta
from scripts.byoc_utils.model import ModelDimensions, make_dummy_tensors
from scripts.byoc_utils.pipeline import setup_pipeline_from_defaults
from scripts.quantize_model import QUANTIZATION_MODES, VARIANTS, ModelMeta


def make_parser():
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", choices=VARIANTS, required=True, type=str,
                        help="Model variant to use for control-video-to-world generation")
    parser.add_argument("--modelopt_checkpoint", type=str, required=True, help="Path to ModelOPT-quantized checkpoint.")
    parser.add_argument("--output_dir", type=str, default="output", help="Folder to export ONNX files to.")
    parser.add_argument("--mode", type=str, choices=list(QUANTIZATION_MODES.keys()), default="FP8",
                        help="Quantization mode (FP8 or NVFP4)")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    return parser


# TODO(rafonsorodri): add support for NVFP4
def export_dit_onnx(model: ModelMeta, dims: ModelDimensions, dit_controlnet, cmdargs) -> str:
    # Fuse QKV projection
    for block in dit_controlnet.blocks + dit_controlnet.control_blocks:
        block.self_attn.fuse_qkv_proj()  # self-attention only, cross-attention has Sq != Sk

    # ModelOPT quantization schema
    assert os.path.exists(cmdargs.modelopt_checkpoint), "ModelOPT-quantized checkpoint not found"
    mto.restore(dit_controlnet, cmdargs.modelopt_checkpoint)

    dummy_tensors = make_dummy_tensors(dims)

    onnx_dir = os.path.join(cmdargs.output_dir, f"onnx_{model.safe_name}_2B_{cmdargs.mode}")
    os.makedirs(onnx_dir, exist_ok=True)

    # Export regular DiT blocks
    for bidx in tqdm.trange(dims.BK, disable=False, desc="Exporting base block to ONNX"):
        block_meta = BlockMeta(bidx, is_control=False, receives_control=bidx in dit_controlnet.control_layers)
        inputs = {key: dummy_tensors[key] for key in block_meta.fixed_inputs}
        export_block_as_onnx(onnx_dir, inputs, block_meta, dit_controlnet.blocks[bidx])

    # Export control DiT blocks
    c = dummy_tensors["control_B_T_H_W_D"]
    for bidx in tqdm.trange(dims.BC, disable=False, desc="Exporting control block to ONNX"):
        block_meta = BlockMeta(bidx, is_control=True, receives_control=False)
        inputs = {"c": c, **{key: dummy_tensors[key] for key in block_meta.fixed_inputs}}
        c = export_block_as_onnx(onnx_dir, inputs, block_meta, dit_controlnet.control_blocks[bidx])

    return onnx_dir


def export_block_as_onnx(
    onnx_dir: str,
    input_dict: dict,
    meta: BlockMeta,
    block: ControlAwareDiTBlock | ControlEncoderDiTBlock,
) -> torch.Tensor:

    # Prepare block
    block = meta.wrapper_class(block)
    block.cuda()
    block.eval()

    # Switch TE implementations with native PyTorch
    block.block.self_attn.prepare_for_export()
    block.block.cross_attn.prepare_for_export()

    # Call forward on random inputs
    inputs = tuple(input_dict.values())
    names = list(input_dict)
    output = block(*inputs)

    # Export to ONNX
    onnx_file = os.path.join(onnx_dir,  f"cosmos_transfer2.5_{meta.block_type}_block{meta.block_index}.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            block,
            inputs,
            onnx_file,
            opset_version=20,
            autograd_inlining=False,
            input_names=names,
            output_names=['output'],
            dynamic_axes=meta.dynamic_axes,
            dynamo=False,
        )

    return output


def main(cmdargs) -> str:
    pipe, args, dims = setup_pipeline_from_defaults({
        "model_variant": cmdargs.model_variant,
        "output_dir": cmdargs.output_dir,
        "resolution": cmdargs.resolution,
        "disable_guardrail": True,
    })
    dit_controlnet = pipe.model.net
    del pipe

    onnx_dir = export_dit_onnx(args.model, dims, dit_controlnet, cmdargs)
    return onnx_dir


if __name__ == "__main__":
    main(make_parser().parse_args())
