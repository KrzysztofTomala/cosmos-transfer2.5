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
import random
import sys
import json
import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from megatron.core import parallel_state

# ModelOPT
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import tqdm

from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.predict2.inference.video2world import _VIDEO_EXTENSIONS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2.config import BASE_MODEL_VARIANTS, ModelVariant
from cosmos_transfer2.inference import Control2WorldInference

SCRIPTS_ROOT = os.path.dirname(__file__)
VIDEO2WORLD_ASSETS = os.path.join(SCRIPTS_ROOT, "../assets/video2world")  # TODO(rafonsorodri): gather controlvideo2world assets for each modality and multivew variants
DIT_PATH = "checkpoints/nvidia/Cosmos-Transfer2.5-2B/{domain}/{modality}/{checkpoint_name}"

VARIANTS = [v.name for v in BASE_MODEL_VARIANTS] + [ModelVariant.AUTO_MULTIVIEW.name]  # TODO(rafonsorodri): add support for robot-domain variants upon their release


@dataclass
class InferenceSample:
    prompt: str
    video_path: str


def make_parser():
    # Command line args
    parser = argparse.ArgumentParser(
        description="Just-in-time post-training quantization of a single control checkpoint. Note, for multi control, "
                    "quantize each base modality separately.")
    parser.add_argument("--variant", choices=VARIANTS, required=True, type=str,
                        help="Model variant to use for control-video-to-world generation")
    parser.add_argument("--checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for calibration. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--config", type=str, default="fp8", help="Quantization mode (fp8 or svdquant)")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    # parser.add_argument("--natten", action="store_true",
    #                     help="Optimize checkpoints with NeighbourhoodAttention")
    parser.add_argument("--calibration_mode", choices=["FULL", "MINIMAL"], default="FULL",
                        help='Calibration mode controls the number of samples to use during model optimization. "FULL"'
                             'processes the entire dataset, leading to higher accuracy retention in exchange for slower'
                             'calibration. "MINIMAL" adapts the number of samples according to the model context for a'
                             'balanced optimisation performance.')
    parser.add_argument("--calibration_dataset", type=str, default=VIDEO2WORLD_ASSETS,
                        help='Optional path to a custom calibration dataset. Defaults to Video2World assets.')
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use. Activates CP if > 1.")
    return parser


def extract_domain_and_modality(variant: ModelVariant) -> tuple[str, str]:
    tokens = variant.name.split('/')
    if len(tokens) == 1:
        return "general", tokens[0]
    return tokens[0], tokens[1]


def calibration_samples_required(args: argparse.Namespace, quant_config: dict) -> int:
    return 1  # TODO(rafonsorodri): Adapt samples according to model size, resolution, FPS, NATTEN, quantized precision


def setup_pipeline(args: argparse.Namespace):
    from cosmos_transfer2.config import MODEL_CHECKPOINTS, ModelKey, SetupArguments

    log.info(f"Using model variant: {args.model_variant}")
    try:
        model_variant = ModelVariant(args.model_variant)
    except ValueError as e:
        raise ValueError(f"Choose either {VARIANTS}.") from e
    model_key = ModelKey(variant=model_variant)

    if args.checkpoint_name:
        domain, modality = extract_domain_and_modality(model_variant)
        checkpoint_path = DIT_PATH.format(domain=domain, modality=modality, checkpoint_name=args.checkpoint_name)
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
        "output_dir": "/tmp/cosmos_predict2",
        # Optional parameters
        "model": model_key.name,
        "checkpoint_path": checkpoint_path,
        "context_parallel_size": args.num_gpus,
        "disable_guardrails": args.disable_guardrail,
        "offload_guardrail_models": args.offload_guardrail,
    })

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
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
    log.info(f"Initializing ControlVideo2WorldInference with model size: {args.model_variant}")
    inference = Control2WorldInference(setup_args, batch_hint_keys=[model_variant])
    return inference.inference_pipeline


def validate_input_file(input_path: str, num_conditional_frames: int) -> bool:
    if not os.path.exists(input_path):
        log.warning(f"Input file does not exist, skipping: {input_path}")
        return False

    ext = os.path.splitext(input_path)[1].lower()
    # Control only accept videos
    if ext not in _VIDEO_EXTENSIONS:
        log.warning(
            f"Skipping file for control (requires video): {input_path} (expected: {_VIDEO_EXTENSIONS}, got: {ext})"
        )
        return False

    if num_conditional_frames not in [1, 5]:
        log.error(f"Invalid num_conditional_frames: {num_conditional_frames} (must be 1 or 5)")
        return False

    return True


def process_single_generation(
    pipe: ControlVideo2WorldInference,
    input_path: str,
    prompt: str,
    negative_prompt: str,
    resolution: str,
    num_conditional_frames: int,
    num_video_frames: int,
    guidance: float,
    seed: int,
) -> bool:
    del num_video_frames  # Implicit from control video
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False
    log.info(f"Running ControlVideo2WorldInference\ninput: {input_path}\nprompt: {prompt}")
    start_time = time.time()
    video = pipe.generate_img2world(
        prompt=prompt,
        video_path=input_path,
        guidance=guidance,
        seed=seed,
        resolution=resolution,
        num_conditional_frames=num_conditional_frames,
        negative_prompt=negative_prompt,
    )
    torch.cuda.synchronize()
    distributed.barrier()
    elapsed = time.time() - start_time
    log.info(f" Generation time: {elapsed:.1f} seconds.")
    return video is not None


def generate_video(pipe: ControlVideo2WorldInference, inference_args: argparse.Namespace, samples: list[InferenceSample]) -> None:
    if inference_args.benchmark:
        log.warning(
            "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
        )

    log.info(f"Running {inference_args.inference_type} generation")
    for idx in tqdm.trange(len(samples), disable=False, desc="Processing batch item"):
        sample = samples[idx]

        if not sample.video_path or not sample.prompt:
            log.warning(f"Skipping item {idx}: Missing input_path or prompt")
            continue

        process_single_generation(
            pipe=pipe,
            input_path=sample.video_path,
            prompt=sample.prompt,
            negative_prompt=inference_args.negative_prompt,
            resolution=inference_args.resolution_hw,
            num_video_frames=inference_args.num_output_frames,
            num_conditional_frames=inference_args.num_conditional_frames,
            guidance=inference_args.guidance,
            seed=inference_args.seed,
        )


def calibrate_dit_denoiser(pipe, args: argparse.Namespace, quant_config):
    # Collect calibration data
    dataset_path = args['calibration_dataset']
    assert os.path.exists(dataset_path), f"Calibration dataset does not exist: {dataset_path}"
    prompts = []
    for batch_file in Path(dataset_path).rglob("*batch*.json"):
        with open(batch_file, 'rt') as fd:
            prompts.extend(json.load(fd))
    assert prompts, f"Did not find calibration samples in {dataset_path}."
    inference_samples = [
        InferenceSample(prompt=sample.get('prompt'), video_path=sample.get('input_video'))
        for sample in prompts
    ]

    num_samples = calibration_samples_required(args, quant_config)
    if args['calibration_mode'] == "FULL":
        if len(inference_samples) <= num_samples:
            warnings.warn(UserWarning("Inadequate number of samples detected. Consider using a larger calibration dataset for an improved optimization outcome."))
    else:  # MINIMAL
        if len(inference_samples) < num_samples:
            warnings.warn(UserWarning(f"Inadequate number of samples detected. MINIMAL calibration for the given model context requires {num_samples} samples."))
        inference_samples = random.sample(inference_samples, k=num_samples)

    inference_args = argparse.Namespace(**{
        "name": "calibration",
        "negative_prompt": args.negative_prompt,
        "resolution_hw": args.resolution,
        "num_conditional_frames": args.num_conditional_frames,
        "control_weight_dict": {args.model_variant: "1.0"},
        "guidance": args.guidance,
        "seed": args.seed,
    })

    dit_net = pipe.model.net  # TODO(rafonsorodri): + control blocks

    # Fuse QKV projection
    for block in dit_net.blocks:
        block.self_attn.fuse_qkv_proj()

    def forward_loop(dit_net):
        pipe.model.net = dit_net  # TODO(rafonsorodri): + control blocks
        generate_video(pipe, inference_args, inference_samples)
        return pipe.model.net

    return mtq.quantize(dit_net, quant_config, forward_loop)


def main(cmdargs):
    # Base args
    with open(os.path.join(SCRIPTS_ROOT, "export_dit_onnx_args.json"), "rt") as f:
        args: dict = json.load(f)
    args.update({
        "checkpoint_name": cmdargs.checkpoint_name,
        "model_variant": cmdargs.model_variant,
        "resolution": cmdargs.resolution,
        "calibration_mode": cmdargs.calibration_mode,
        "calibration_dataset": cmdargs.calibration_dataset,
    })
    if cmdargs.num_gpus > 1:
        args["num_gpus"] = cmdargs.num_gpus
    args: argparse.Namespace = argparse.Namespace(**args)

    pipe = setup_pipeline(args)

    quant_config = {
        # Quantize into FP8
        "fp8": mtq.FP8_DEFAULT_CFG,
        # Quantize with SVDquant (FP4)
        "svdquant": mtq.NVFP4_SVDQUANT_DEFAULT_CFG,
    }[cmdargs.config.lower()]

    dit_net = calibrate_dit_denoiser(pipe, args, quant_config)

    # Save checkpoint and return quantized pipeline
    filename_noext = f"output/dit_net_{cmdargs.model_variant}_{cmdargs.resolution}_{cmdargs.config}"
    mto.save(dit_net, f"{filename_noext}.pt")
    real_stdout = sys.stdout
    with open(f"{filename_noext}.mtq-rep.txt", 'w') as sys.stdout:
        mtq.print_quant_summary(dit_net)
    sys.stdout = real_stdout

    pipe.dit = dit_net

    return pipe


if __name__ == "__main__":
    # Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = make_parser()
    cmdargs = parser.parse_args()
    main(cmdargs)
