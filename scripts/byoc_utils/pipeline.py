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
import json
import os

import modelopt.torch.quantization as mtq
import pydantic
import torch
from megatron.core import parallel_state

from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2.config import MODEL_CHECKPOINTS, ModelKey, SetupArguments
from cosmos_transfer2.inference import Control2WorldInference
from scripts.byoc_utils.model import ModelMeta, SCRIPTS_ROOT, ModelDimensions, get_model_dimensions

DIT_PATH = "checkpoints/nvidia/Cosmos-Transfer2.5-2B/{domain}/{modality}/{checkpoint_name}"

QUANTIZATION_MODES = {
    "FP8": mtq.FP8_DEFAULT_CFG,
    "NVFP4": mtq.NVFP4_SVDQUANT_DEFAULT_CFG,  # Quantize with SVDquant for NVFP4
}


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
    log.info(f"Initializing ControlVideo2WorldInference for model: {args.model.name}")
    log.info(f"Using batch_hint_keys: {args.model.hint_keys}")
    inference = Control2WorldInference(setup_args, batch_hint_keys=args.model.hint_keys)
    return inference.inference_pipeline


def setup_pipeline_from_defaults(overrides: dict | None = None) -> tuple[ControlVideo2WorldInference, PipelineArgs, ModelDimensions]:
    variants = overrides.pop("model_variant")
    if isinstance(variants, list):
        input_file = "optim_dit_args_multicontrol.json"
    else:
        variants = [variants]
        input_file = "optim_dit_args.json"
    overrides["model_variant"] = variants

    # Base args
    with open(os.path.join(SCRIPTS_ROOT, "byoc_utils", input_file), "rt") as f:
        config: dict = json.load(f)
    if overrides:
        config.update(overrides)
    model_variant = config["model_variant"]
    if not isinstance(model_variant, list):
        model_variant = [model_variant]
    config.setdefault("model", ModelMeta.from_text(model_variant))
    args = PipelineArgs(**config)
    pipe = setup_pipeline(args)
    dims = get_model_dimensions(pipe.config, config["resolution"])
    return pipe, args, dims
