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
import json

import modelopt.torch.opt as mto
import torch
import torch.onnx
import tqdm

from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
    ControlAwareDiTBlock,
    ControlEncoderDiTBlock,
)
from scripts.byoc_utils.block import BlockMeta
from scripts.byoc_utils.model import ModelDimensions, make_dummy_tensors, fuse_qkv_projections
from scripts.byoc_utils.pipeline import PipelineArgs, setup_pipeline
from scripts.byoc_utils.model import ASSETS_ROOT, SCRIPTS_ROOT, VARIANTS, fuse_qkv_projections, get_model_dimensions

from scripts.quantize_model import QUANTIZATION_MODES, VARIANTS, ModelMeta


def make_parser():
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", nargs='+', choices=VARIANTS, required=True, type=str,
                        help="Model variant(s) to use for control-video-to-world generation. "
                             "Provide single variant (e.g., edge) for single control, "
                             "or multiple variants (e.g., edge vis depth seg) for multicontrol mode.")
    parser.add_argument("--modelopt_checkpoint", type=str, default=None, help="Path to ModelOPT-quantized checkpoint.")
    parser.add_argument("--controls_only", action="store_true", help="Export control layers only (skip base layers).")
    parser.add_argument("--output_dir", type=str, default="output", help="Folder to export ONNX files to.")
    parser.add_argument("--mode", type=str, choices=list(QUANTIZATION_MODES.keys()) + ["BF16"], default="FP8",
                        help="Quantization mode (FP8 or NVFP4 or BF16)")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    parser.add_argument("--edge_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for edge control. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--depth_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for depth control. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--seg_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for seg control. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--vis_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for vis control. Defaults to the"
                             " registered post-trained checkpoint.")

    return parser


# TODO(rafonsorodri): add support for NVFP4
def export_dit_onnx(model: ModelMeta, dims: ModelDimensions, dit_controlnet, cmdargs) -> str:
    # Fuse QKV projection
    fuse_qkv_projections(dit_controlnet, model.is_multicontrol)

    # ModelOPT quantization schema
    if cmdargs.mode != "BF16":
        assert os.path.exists(cmdargs.modelopt_checkpoint), "ModelOPT-quantized checkpoint not found"
        mto.restore(dit_controlnet, cmdargs.modelopt_checkpoint)

    dummy_tensors = make_dummy_tensors(dims)

    onnx_dir = os.path.join(cmdargs.output_dir, f"onnx_{model.safe_name}_2B_{cmdargs.mode}")
    os.makedirs(onnx_dir, exist_ok=True)

    # Export control DiT blocks
    if model.is_multicontrol:
        for control_branch in range(len(model.variants)):
            c = dummy_tensors["control_B_T_H_W_D"]
            control_blocks = getattr(dit_controlnet, f"control_blocks_{control_branch}")
            for bidx in tqdm.trange(
                    dims.BC, disable=False, desc=f"Exporting control block to ONNX ({control_branch=})"):
                block_meta = BlockMeta(bidx, is_control=True, receives_control=False, control_branch=control_branch)
                inputs = {"c": c, **{key: dummy_tensors[key] for key in block_meta.fixed_inputs}}
                c = export_block_as_onnx(onnx_dir, inputs, block_meta, control_blocks[bidx])
    else:
        c = dummy_tensors["control_B_T_H_W_D"]
        for bidx in tqdm.trange(dims.BC, disable=False, desc="Exporting control block to ONNX"):
            block_meta = BlockMeta(bidx, is_control=True, receives_control=False)
            inputs = {"c": c, **{key: dummy_tensors[key] for key in block_meta.fixed_inputs}}
            c = export_block_as_onnx(onnx_dir, inputs, block_meta, dit_controlnet.control_blocks[bidx])

    if cmdargs.controls_only:
        return onnx_dir

    # Export regular DiT blocks
    for bidx in tqdm.trange(dims.BK, disable=False, desc="Exporting base block to ONNX"):
        block_meta = BlockMeta(bidx, is_control=False, receives_control=bidx in dit_controlnet.control_layers)
        inputs = {key: dummy_tensors[key] for key in block_meta.fixed_inputs}
        export_block_as_onnx(onnx_dir, inputs, block_meta, dit_controlnet.blocks[bidx])

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
    onnx_file = os.path.join(onnx_dir,  f"{meta.block_label}.onnx")
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
    model_variant = cmdargs.model_variant if isinstance(cmdargs.model_variant, list) else [cmdargs.model_variant]
    model_meta = ModelMeta.from_text(model_variant)

    input_file = "optim_dit_args_multicontrol.json" if len(cmdargs.model_variant) > 1 else "optim_dit_args.json"

    with open(os.path.join(SCRIPTS_ROOT, "byoc_utils", input_file), "rt") as f:
        args: dict = json.load(f)
        checkpoint_base_path = '/opt/nim/workspace/checkpoints/diffusion/torch/'
        args["checkpoint_paths"] = {
            "edge": cmdargs.edge_checkpoint_name or checkpoint_base_path + "general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
            "depth": cmdargs.depth_checkpoint_name or checkpoint_base_path + "general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt",
            "seg": cmdargs.seg_checkpoint_name or checkpoint_base_path + "general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt",
            "vis": cmdargs.vis_checkpoint_name or checkpoint_base_path + "general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt",
        }
        args.update({
            "model": model_meta,
            "output_dir": cmdargs.output_dir,
            "disable_guardrail": True,
        })

    pipeline_args = PipelineArgs(**args)
    pipe = setup_pipeline(pipeline_args)
    dit_controlnet = pipe.model.net

    if isinstance(args["resolution"], list):
        dims = {}
        for resolution in args["resolution"]:
            dims[resolution] = get_model_dimensions(pipe.config, resolution)
    else:
        dims = get_model_dimensions(pipe.config, args["resolution"])

    del pipe
    

    onnx_dir = export_dit_onnx(args['model'], dims, dit_controlnet, cmdargs)
    return onnx_dir


if __name__ == "__main__":
    main(make_parser().parse_args())
