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

from dataclasses import dataclass

import torch

from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import (
    ControlAwareDiTBlock,
    ControlEncoderDiTBlock,
)
from scripts.byoc_utils.model import FIXED_INPUTS, FIXED_INPUTS_BASE_CONTROLLED


class RegularDiTBlock(torch.nn.Module):
    def __init__(self, block: ControlAwareDiTBlock):
        super().__init__()
        self.block = block

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_T_H_W_1_1_D, adaln_lora_B_T_3D):
        rope_emb_L_1_1_D = rope_emb_T_H_W_1_1_D.flatten(0, 2)
        return self.block(
            x_B_T_H_W_D,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )


class ControlReceivingDiTBlock(torch.nn.Module):
    def __init__(self, block: ControlAwareDiTBlock):
        super().__init__()
        self.block = block

    def forward(self, x_B_T_H_W_D, hints, control_context_scale, emb_B_T_D, crossattn_emb, rope_emb_T_H_W_1_1_D, adaln_lora_B_T_3D):
        rope_emb_L_1_1_D = rope_emb_T_H_W_1_1_D.flatten(0, 2)
        return self.block(
            x_B_T_H_W_D,
            torch.unbind(hints),
            control_context_scale,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )


class ControlProducingDiTBlock0(torch.nn.Module):
    def __init__(self, block: ControlEncoderDiTBlock):
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


class ControlProducingDiTBlockN(torch.nn.Module):
    def __init__(self, block: ControlEncoderDiTBlock):
        super().__init__()
        self.block = block

    def forward(self, c, emb_B_T_D, crossattn_emb, rope_emb_T_H_W_1_1_D, adaln_lora_B_T_3D):
        rope_emb_L_1_1_D = rope_emb_T_H_W_1_1_D.flatten(0, 2)
        return self.block(
            c,
            x_B_T_H_W_D=None,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )


@dataclass
class BlockMeta:
    block_index: int
    is_control: bool
    receives_control: bool

    control_branch: int = -1

    @property
    def block_type(self):
        return "controlnet" if self.is_control else "net"

    @property
    def block_label(self):
        if self.control_branch > -1:
            return f"cosmos_transfer2.5_{self.block_type}_branch{self.control_branch}_block{self.block_index}"
        return f"cosmos_transfer2.5_{self.block_type}_block{self.block_index}"

    @property
    def wrapper_class(self):
        if self.is_control:
            if self.block_index == 0:
                return ControlProducingDiTBlock0
            return ControlProducingDiTBlockN
        if self.receives_control:
            return ControlReceivingDiTBlock
        return RegularDiTBlock

    @property
    def fixed_inputs(self):
        if self.receives_control:
            return FIXED_INPUTS_BASE_CONTROLLED.copy()
        if self.is_control and self.block_index > 0:
            fixed_inputs = FIXED_INPUTS.copy()
            fixed_inputs.remove("x_B_T_H_W_D")
            return fixed_inputs
        return FIXED_INPUTS.copy()

    @property
    def dynamic_axes(self):
        dynamic_axes = {
            'x_B_T_H_W_D': {2: 'H', 3: 'W'},
            'rope_emb_T_H_W_1_1_D': {1: 'H', 2: 'W'},
        }
        if self.is_control:
            dynamic_axes["output"] = {3: 'H', 4: 'W'}  # Control blocks stack outputs of shape [bidx+2, B, T, H, W, D]
            if self.block_index == 0:
                dynamic_axes["c"] = {2: 'H', 3: 'W'}
            else:
                dynamic_axes.pop("x_B_T_H_W_D")
                dynamic_axes["c"] = {3: 'H', 4: 'W'}  # For subsequent blocks shape becomes [bidx+1, B, T, H, W, D]
        else:
            dynamic_axes["output"] = {2: 'H', 3: 'W'}
            if self.receives_control:
                dynamic_axes["hints"] = {3: 'H', 4: 'W'}  # Assume a stacked tensor rather than list of control outputs
        return dynamic_axes
