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

import modelopt.torch.quantization as mtq
import pydantic
import torch
from megatron.core import parallel_state


from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import NUM_EMBEDDING_PADDING_TOKENS
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import Config
from cosmos_transfer2.config import BASE_MODEL_VARIANTS, MODEL_CHECKPOINTS, ModelKey, ModelVariant, SetupArguments
from cosmos_transfer2.inference import Control2WorldInference

SCRIPTS_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_ROOT = os.path.realpath(os.path.join(SCRIPTS_ROOT, "..", "assets"))
CONTROL2WORLD_ASSETS = os.path.join(ASSETS_ROOT, "{modality}.jsonl")
DIT_PATH = "checkpoints/nvidia/Cosmos-Transfer2.5-2B/{domain}/{modality}/{checkpoint_name}"

QUANTIZATION_MODES = {
    "FP8": mtq.FP8_DEFAULT_CFG,
    "NVFP4": mtq.NVFP4_SVDQUANT_DEFAULT_CFG,  # Quantize with SVDquant for NVFP4
}
# TODO(rafonsorodri): add support for robot-domain variants upon their release
VARIANTS = [v.value for v in BASE_MODEL_VARIANTS] + [ModelVariant.AUTO_MULTIVIEW.value]


@dataclass
class ModelDimensions:
    B: int = 1  # batch size
    T: int = 1  # frames
    N: int = 1  # sequence length
    H: int = 1  # latent height
    W: int = 1  # latent width
    D: int = 1  # head dimension
    HS: int = 1  # heads in SelfAttn
    HX: int = 1  # heads in CrossAttn
    BK: int = 1  # transformer base blocks
    BC: int = 1  # transformer control blocks


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


class PipelineArgs(pydantic.BaseModel):
    """Common arguments for pipeline setup."""

    model_config = pydantic.ConfigDict(extra="ignore", frozen=True, arbitrary_types_allowed=True)

    # Required parameters
    output_dir: str
    """Output directory."""
    model: ModelMeta
    """Model metadata."""

    # Optional parameters
    # pyrefly: ignore  # invalid-annotation
    checkpoint_name: str | None = None
    """Filename of checkpoint."""
    num_gpus: int = 1
    """Number of available GPUs."""
    disable_guardrail: bool = False
    """Whether to disable model guardrails."""
    offload_guardrail: bool = True
    """Offload guardrail models to CPU to save GPU memory."""
    benchmark: bool = False
    """Enable benchmarking mode. Runs the single video processing 4 times and reports average of last 3 runs."""
    seed: int = 0
    """Base seed for PRN generation"""


def setup_pipeline(args: PipelineArgs):
    log.info(f"Using model variant: {args.model.name}")
    model_key = ModelKey(variant=args.model.variant)

    if args.checkpoint_name:
        checkpoint_path = DIT_PATH.format(
            domain=args.model.domain, modality=args.model.hint_key, checkpoint_name=args.checkpoint_name)
    else:
        try:
            model_checkpoint = MODEL_CHECKPOINTS[model_key]
            checkpoint_path = model_checkpoint.path
        except KeyError as e:
            raise NotImplementedError(f"Model configuration not supported") from e
        except ValueError as e:
            # raised upon calling `.path` property if the checkpoint does not specify a HuggingFace location
            raise NotImplementedError(f"No HF repository defined") from e
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    setup_args = SetupArguments.model_validate({
        # Required parameters
        "output_dir": args.output_dir,
        # Optional parameters
        "model": model_key.name,
        "checkpoint_path": checkpoint_path,
        "context_parallel_size": args.num_gpus,
        "disable_guardrails": args.disable_guardrail,
        "offload_guardrail_models": args.offload_guardrail,
        "benchmark": args.benchmark,
    })

    if setup_args.benchmark:
        log.warning(
            "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation "
            "time will be shown."
        )

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")

        # Check if distributed environment is already initialized
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
            log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
        else:
            log.info("Distributed environment already initialized, skipping initialization")
            # Check if we need to reinitialize with different context parallel size
            current_cp_size = parallel_state.get_context_parallel_world_size()
            if current_cp_size != args.num_gpus:
                log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                log.warning("Using existing context parallel configuration")
            else:
                log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

    # Load models
    log.info(f"Initializing ControlVideo2WorldInference for model: {args.model.variant.name}")
    inference = Control2WorldInference(setup_args, batch_hint_keys=[args.model.hint_key])
    return inference.inference_pipeline


def get_model_dimensions(model_config: Config, resolution) -> ModelDimensions:
    config = model_config.model.config
    try:
        resolution_hw = VIDEO_RES_SIZE_INFO[resolution]["9,16"]
    except KeyError as e:
        raise ValueError(f"Unsupported resolution. Choose either '480' or '720'.") from e

    patch_size = config.text_encoder_config.model_config.model_config.vision_encoder_config.patch_size
    return ModelDimensions(
        B=1,
        T=config.state_t,  # frame count
        N=NUM_EMBEDDING_PADDING_TOKENS,  # CrossAttn seq len (the effective sequence length for text embeddings)
        HX=config.net.num_heads,  # CrossAttn head count
        HS=config.net.num_heads,  # SelfAttn head count
        D=config.net.model_channels // config.net.num_heads,  # SelfAttn head dimension
        H=resolution_hw[0] // patch_size,
        W=resolution_hw[1] // patch_size,
        BK=config.net.num_blocks,
        BC=config.net.num_blocks // config.net.vace_block_every_n
    )
