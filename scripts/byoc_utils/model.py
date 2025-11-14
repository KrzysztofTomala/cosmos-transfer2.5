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

import os
from dataclasses import dataclass

import torch

from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import NUM_EMBEDDING_PADDING_TOKENS
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import Config
from cosmos_transfer2.config import BASE_MODEL_VARIANTS, ModelVariant

SCRIPTS_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_ROOT = os.path.realpath(os.path.join(SCRIPTS_ROOT, "..", "assets"))
CONTROL2WORLD_ASSETS = os.path.join(ASSETS_ROOT, "{modality}.jsonl")

# TODO(rafonsorodri): add support for robot-domain variants upon their release
VARIANTS = [v.value for v in BASE_MODEL_VARIANTS] + [ModelVariant.AUTO_MULTIVIEW.value]
ASPECT_RATIO = "9,16"

FIXED_INPUTS = [
    "x_B_T_H_W_D",
    "hints",
    "control_context_scale",
    "emb_B_T_D",
    "crossattn_emb",
    "rope_emb_T_H_W_1_1_D",
    "adaln_lora_B_T_3D"
]
FIXED_CONTROL_INPUTS = FIXED_INPUTS.copy()
FIXED_CONTROL_INPUTS.remove("hints")
FIXED_CONTROL_INPUTS.remove("control_context_scale")

SHAPE_SPECS = {
    "x_B_T_H_W_D": ["B", "T", "H", "W", "HS*DS"],
    "control_B_T_H_W_D": ["B", "T", "H", "W", "HS*DS"],
    "hints": ["BC", "B", "T", "H", "W", "HS*DS"],
    "emb_B_T_D": ["B", "T", "HS*DS"],
    "crossattn_emb": ["B", "T", "HX*DX"],
    "rope_emb_T_H_W_1_1_D": ["T", "H", "W", "1", "1", "DS"],
    "adaln_lora_B_T_3D": ["B", "T", "3*HS*DS"],
    "control_context_scale": ["1"],
}


@dataclass
class ModelDimensions:
    B: int = 1  # batch size
    T: int = 1  # frames
    N: int = 1  # sequence length
    H: int = 1  # latent height
    W: int = 1  # latent width
    DS: int = 1  # head dimension in SelfAttn
    DX: int = 1  # head dimension in CrossAttn
    HS: int = 1  # heads in SelfAttn
    HX: int = 1  # heads in CrossAttn
    BK: int = 1  # transformer base blocks
    BC: int = 1  # transformer control blocks


@dataclass
class OperationalBounds:
    # Frames bounds
    T_MIN: int
    T_MAX: int
    # Height bounds
    H_MIN: int
    H_MAX: int
    # Width bounds
    W_MIN: int
    W_MAX: int

    @property
    def __DIMENSIONS(self):
        return ["T", "H", "W"]

    def __contains__(self, item):
        return item in self.__DIMENSIONS


class ModelMeta:
    def __init__(self, model_variant: ModelVariant):
        self._variant = model_variant

    @property
    def variant(self):
        return self._variant

    @property
    def domain(self):
        tokens = self._variant.value.split('/')
        if len(tokens) == 1:
            return "general"
        return tokens[0]

    @property
    def hint_key(self):
        tokens = self._variant.value.split('/')
        if len(tokens) == 1:
            return tokens[0]
        return tokens[1]

    @property
    def name(self):
        return self._variant.value

    @property
    def safe_name(self):
        return self.name.replace("/", "-")

    @property
    def calibration_dataset(self):
        return CONTROL2WORLD_ASSETS.format(modality=self.safe_name)

    @classmethod
    def from_text(cls, value: str):
        try:
            model_variant = ModelVariant(value)
        except ValueError as e:
            raise ValueError(f"Choose either {VARIANTS}.") from e
        return cls(model_variant)


def get_model_dimensions(model_config: Config, resolution) -> ModelDimensions:
    config = model_config.model.config
    try:
        resolution_hw = VIDEO_RES_SIZE_INFO[resolution][ASPECT_RATIO]
    except KeyError as e:
        raise ValueError(f"Unsupported resolution. Choose either '480' or '720'.") from e

    patch_size = config.text_encoder_config.model_config.model_config.vision_encoder_config.patch_size
    return ModelDimensions(
        B=1,
        T=config.state_t,  # frame count
        N=NUM_EMBEDDING_PADDING_TOKENS,  # CrossAttn seq len (the effective sequence length for text embeddings)
        HS=config.net.num_heads,  # SelfAttn head count
        HX=config.net.num_heads,  # CrossAttn head count
        DS=config.net.model_channels // config.net.num_heads,  # SelfAttn head dimension
        DX=config.net.crossattn_emb_channels // config.net.num_heads,  # CrossAttn head dimension
        H=resolution_hw[0] // patch_size,
        W=resolution_hw[1] // patch_size,
        BK=config.net.num_blocks,
        BC=config.net.num_blocks // config.net.vace_block_every_n
    )


def shapes_from_spec(spec: list, dims: ModelDimensions, bounds: OperationalBounds | None = None) -> dict[str, tuple]:
    opt_idx = 0
    min_idx = 1
    max_idx = 2

    shapes = torch.ones((3, len(spec)))
    for i, dim in enumerate(spec):
        for subdim in dim.split('*'):
            base = getattr(dims, subdim, None) or int(subdim)
            shapes[:, i] *= base
            if bounds and subdim in bounds:
                shapes[min_idx, i] *= getattr(bounds, f"{subdim}_MIN") / base
                shapes[max_idx, i] *= getattr(bounds, f"{subdim}_MAX") / base

    shapes = shapes.long().tolist()
    return {
        "opt": tuple(shapes[opt_idx]),
        "min": tuple(shapes[min_idx]),
        "max": tuple(shapes[max_idx]),
    }


def make_dummy_tensors(dims: ModelDimensions, with_outputs: bool = False) -> dict[str, torch.Tensor]:
    def _make(name, dtype=torch.bfloat16):
        return torch.randn(
            shapes_from_spec(SHAPE_SPECS[name], dims)['opt'],
            requires_grad=False, device="cuda", dtype=dtype
        )

    tensors = {
        "x_B_T_H_W_D": _make("x_B_T_H_W_D"),
        "control_B_T_H_W_D": _make("control_B_T_H_W_D"),
        "hints": _make("hints"),
        "emb_B_T_D": _make("emb_B_T_D", dtype=torch.float),
        "crossattn_emb": _make("crossattn_emb"),
        "rope_emb_T_H_W_1_1_D": _make("rope_emb_T_H_W_1_1_D", dtype=torch.float),
        "adaln_lora_B_T_3D": _make("adaln_lora_B_T_3D", dtype=torch.float),
        "control_context_scale": torch.ones_like(_make("control_context_scale")),
    }

    if with_outputs:
        tensors["output_B_T_H_W_D"] = torch.empty_like(_make("x_B_T_H_W_D"))
        tensor = torch.empty_like(_make("control_B_T_H_W_D"))
        tensors["output_hints"] = tensor.unsqueeze(0).repeat_interleave(dims.BC + 1, dim=0)

    return tensors
