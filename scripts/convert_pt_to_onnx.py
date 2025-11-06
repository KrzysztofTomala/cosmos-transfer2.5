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
import json
import os
from typing import Any

import modelopt.torch.opt as mto
import torch
import torch.onnx
import tqdm

from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
    ControlAwareDiTBlock,
    ControlEncoderDiTBlock,
)
from scripts.byoc_utils.pipeline import SCRIPTS_ROOT, PipelineArgs, ModelDimensions, get_model_dimensions
from scripts.quantize_model import QUANTIZATION_MODES, VARIANTS, ModelMeta, setup_pipeline


def make_parser():
    # Command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", choices=VARIANTS, required=True, type=str,
                        help="Model variant to use for control-video-to-world generation")
    parser.add_argument("--output_dir", type=str, help="Folder to export ONNX files to.")
    parser.add_argument("--modelopt_model", type=str, help="Path to ModelOPT-quantized checkpoint.")
    parser.add_argument("--mode", type=str, choices=list(QUANTIZATION_MODES.keys()), default="FP8",
                        help="Quantization mode (FP8 or NVFP4)")
    # parser.add_argument("--nunchaku", action="store_true",
    #                     help="Export SVDquant layers as NunchakuGemmPlugin. Only useful for NVFP4 quantization.")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    return parser


COMMON_DYNAMIC_AXES = {
    'x_B_T_H_W_D': {1: 'T', 2: 'H', 3: 'W'},
    'emb_B_T_D': {1: 'T'},
    'crossattn_emb': {1: 'N'},
    'rope_emb_T_H_W_1_1_D': {0: 'T', 1: 'H', 2: 'W'},
    'adaln_lora_B_T_3D': {1: 'T'},
}
REGULAR_DYNAMIC_AXES = {
    "hints": {1: 'T', 2: 'H', 3: 'W'},
    "output": {1: 'T', 2: 'H', 3: 'W'},
    **COMMON_DYNAMIC_AXES
}
CONTROL_DYNAMIC_AXES_0 = {
    "c": {1: 'T', 2: 'H', 3: 'W'},
    "output": {2: 'T', 3: 'H', 4: 'W'},  # Control blocks stack outputs
    **COMMON_DYNAMIC_AXES
}
CONTROL_DYNAMIC_AXES_N = {
    "c": {2: 'T', 3: 'H', 4: 'W'},  # shape becomes [bidx+1, B, T, H, W, D]
    "output": {2: 'T', 3: 'H', 4: 'W'},  # Control blocks stack outputs
    **COMMON_DYNAMIC_AXES
}


class RegularTracedDitBlock(torch.nn.Module):
    def __init__(self, block: ControlAwareDiTBlock):
        """Tracing wrapper"""
        super().__init__()
        self.block = block

    def forward(self, x_B_T_H_W_D, hints, control_context_scale, emb_B_T_D, crossattn_emb, rope_emb_T_H_W_1_1_D, adaln_lora_B_T_3D):
        rope_emb_L_1_1_D = rope_emb_T_H_W_1_1_D.flatten(0, 2)
        return self.block(
            x_B_T_H_W_D,
            hints,
            control_context_scale,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )


class ControlTracedDitBlock(torch.nn.Module):
    def __init__(self, block: ControlEncoderDiTBlock):
        """Tracing wrapper"""
        super().__init__()
        self.block = block

    def forward(self, c, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_T_H_W_1_1_D, adaln_lora_B_T_3D):
        rope_emb_L_1_1_D = rope_emb_T_H_W_1_1_D.flatten(0, 2)
        return self.block(
            c,
            x_B_T_H_W_D,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )


# TODO(rafonsorodri): add support for NVFP4
def export_dit_onnx(dims: ModelDimensions, dit_controlnet, cmdargs):
    # Fuse QKV projection
    for block in dit_controlnet.blocks + dit_controlnet.control_blocks:
        block.self_attn.fuse_qkv_proj()  # self-attention only, cross-attention has Sq != Sk

    # ModelOPT quantization schema
    assert os.path.exists(cmdargs.modelopt_model), "ModelOPT-quantized checkpoint not found"
    mto.restore(dit_controlnet, cmdargs.modelopt_model)

    def _make(shape):
        return torch.randn(shape, requires_grad=False, device="cuda", dtype=torch.bfloat16)

    x_B_T_H_W_D = _make((dims.B, dims.T, dims.H, dims.W, dims.HS*dims.D))
    control_B_T_H_W_D = _make((dims.B, dims.T, dims.H, dims.W, dims.HS*dims.D))
    hint_B_T_H_W_D = _make((dims.B, dims.T, dims.H, dims.W, dims.HS*dims.D))
    emb_B_T_D = _make((dims.B, dims.T, dims.HS*dims.D))
    crossattn_emb = _make((dims.B, dims.N, dims.HX*dims.D))
    rope_emb_T_H_W_1_1_D = _make((dims.T, dims.H, dims.W, 1, 1, dims.D)).float()
    adaln_lora_B_T_3D = _make((dims.B, dims.T, 3*dims.HS*dims.D))
    hints = (hint_B_T_H_W_D,) * dims.BC
    control_context_scale = 1.0

    onnx_dir = os.path.join(cmdargs.output_dir, f"onnx_{cmdargs.model.safe_name}_2B_{cmdargs.mode}")
    os.makedirs(onnx_dir, exist_ok=True)

    # Export regular DiT blocks
    inputs = {
        "x_B_T_H_W_D": x_B_T_H_W_D,
        "hints": hints,
        "control_context_scale": control_context_scale,
        "emb_B_T_D": emb_B_T_D,
        "crossattn_emb": crossattn_emb,
        "rope_emb_T_H_W_1_1_D": rope_emb_T_H_W_1_1_D,
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
    }
    for bidx in tqdm.trange(dims.BK, disable=False, desc="Exporting base block to ONNX"):
        export_block_as_onnx(onnx_dir, inputs, REGULAR_DYNAMIC_AXES, bidx, dit_controlnet.blocks[bidx], False)

    # Export control DiT blocks
    c = control_B_T_H_W_D
    for bidx in tqdm.trange(dims.BC, disable=False, desc="Exporting control block to ONNX"):
        inputs = {
            "c": c,
            "x_B_T_H_W_D": x_B_T_H_W_D,
            "emb_B_T_D": emb_B_T_D,
            "crossattn_emb": crossattn_emb,
            "rope_emb_T_H_W_1_1_D": rope_emb_T_H_W_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        }
        dynamic_axes = CONTROL_DYNAMIC_AXES_0 if bidx == 0 else CONTROL_DYNAMIC_AXES_N
        c = export_block_as_onnx(onnx_dir, inputs, dynamic_axes, bidx, dit_controlnet.control_blocks[bidx], True)


def export_block_as_onnx(
    onnx_dir: str,
    input_dict: dict,
    dynamic_axes_dict: dict,
    bidx: int,
    block: torch.nn.Module,
    is_control: bool
) -> Any:

    block_type = "controlnet" if is_control else "net"
    onnx_file = os.path.join(onnx_dir,  f"cosmos_transfer2.5_{block_type}_block{bidx}.onnx")

    # Prepare block
    wrapper_class = ControlTracedDitBlock if is_control else RegularTracedDitBlock
    block = wrapper_class(block)
    block.cuda()
    block.eval()
    block.block.self_attn.export_attn()
    block.block.self_attn.export_norm()
    block.block.self_attn.export_rope()
    block.block.cross_attn.export_attn()
    block.block.cross_attn.export_norm()

    # Call forward on random inputs
    inputs = tuple(input_dict.values())
    names = list(input_dict)
    outputs = block(*inputs)

    # Export to ONNX
    with torch.inference_mode():
        torch.onnx.export(
            block,
            inputs,
            onnx_file,
            opset_version=20,
            autograd_inlining=False,
            input_names=names,
            output_names=['output'],
            dynamic_axes={n: dynamic_axes_dict[n] for n in names},
        )

    return outputs


def main(cmdargs):
    model_meta = ModelMeta.from_text(cmdargs.model_variant)

    with open(os.path.join(SCRIPTS_ROOT, "byoc_utils", "optim_dit_args.json"), "rt") as f:
        args: dict = json.load(f)
    args.update({
        "model": model_meta,
        "output_dir": cmdargs.output_dir,
        "model_size": cmdargs.model_size,
        "resolution": cmdargs.resolution,
        "disable_guardrail": True,
    })

    pipe = setup_pipeline(PipelineArgs(**args))
    dims = get_model_dimensions(pipe.config, cmdargs.resolution)
    dit_controlnet = pipe.dit
    del pipe

    export_dit_onnx(dims, dit_controlnet, cmdargs)


if __name__ == "__main__":
    main(make_parser().parse_args())
