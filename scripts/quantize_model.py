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
import random
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import torch
import tqdm
from nimlib import nimutils

model_manifest = (nimutils.get_model_manifest())  # sets nimutils._MODEL_MANIFEST if not yet set
nimutils.download_models(model_manifest)

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2.config import DEFAULT_NEGATIVE_PROMPT
from scripts.byoc_utils.model import ASSETS_ROOT, SCRIPTS_ROOT, VARIANTS, fuse_qkv_projections
from scripts.byoc_utils.pipeline import QUANTIZATION_MODES, ModelMeta, PipelineArgs, setup_pipeline


@dataclass
class CalibrationSample:
    prompt: str
    video_path: str
    control_keys: list[str]
    control_paths: dict[str, str]
    control_weights: dict[str, str]

    @property
    def control_weights_str(self):
        return ",".join([str(self.control_weights[k]) for k in self.control_keys])

    @classmethod
    def from_json(cls, data: dict, modality: list[str]):
        # Parse prompt
        if data.get("prompt"):
            prompt = data["prompt"]
        elif prompt_path := data.get("prompt_path"):
            with open(os.path.join(ASSETS_ROOT, prompt_path), 'rt') as fd:
                prompt = fd.read()
            if not prompt:
                raise ValueError(f"Prompt file is empty: {prompt_path}")
        else:
            raise ValueError("Sample must specify a text prompt inplace (`prompt`) or as .txt (`prompt_path`).")

        # Parse input video
        if not (video_path := data.get("video_path")):
            raise ValueError("Sample must specify an input video (`video_path`).")
        video_path = os.path.join(ASSETS_ROOT, video_path)
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        # Parse control configuration for each modality
        control_keys = []
        control_paths = {}
        control_weights = {}

        for mod in modality:
            if not ((control_config := data.get(mod)) and isinstance(control_config, dict)):
                raise ValueError(f'Control "{mod}" configuration not found in sample, must specify'
                                 f' `{mod}.control_weight` and `{mod}.control_path` properties.')

            control_path = control_config.get("control_path")
            if not control_path:
                if mod in ["edge", "vis"]:
                    log.warning(f"To compute {mod} control for sample online.")
                else:
                    raise ValueError(f"Sample must specify a control video (`{mod}.control_path`). Use "
                                     f"`examples/inference.py` for online control computation.")
            else:
                control_path = os.path.join(ASSETS_ROOT, control_path)
                if not os.path.exists(control_path):
                    raise ValueError(f"Control file does not exist: {control_path}")

            control_weight = control_config.get("control_weight", 1.0)
            if not isinstance(control_weight, int | float) or control_weight < 0.0:
                raise ValueError(f"Control weight must be non-negative: {control_weight}")

            control_keys.append(mod)
            control_paths[mod] = control_path
            control_weights[mod] = control_weight

        return CalibrationSample(
            prompt=prompt,
            video_path=video_path,
            control_keys=control_keys,
            control_paths=control_paths,
            control_weights=control_weights,
        )


def make_parser():
    # Command line args
    parser = argparse.ArgumentParser(
        description="Just-in-time post-training quantization for control checkpoint(s). "
                    "Supports both single and multi-control. Provide multiple variants for multicontrol mode.")
    parser.add_argument("--model_variant", nargs='+', choices=VARIANTS, required=True, type=str,
                        help="Model variant(s) to use for control-video-to-world generation. "
                             "Provide single variant (e.g., edge) for single control, "
                             "or multiple variants (e.g., edge vis depth seg) for multicontrol mode.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Folder to save quantized checkpoint.")
    parser.add_argument("--edge_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for calibration. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--depth_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for calibration. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--seg_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for calibration. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--vis_checkpoint_name", type=str, default="",
                        help="Optionally override checkpoint filename (`*.pt`) to load for calibration. Defaults to the"
                             " registered post-trained checkpoint.")
    parser.add_argument("--mode", type=str, choices=list(QUANTIZATION_MODES.keys()), default="FP8",
                        help="Quantization mode (FP8 or NVFP4)")
    parser.add_argument("--resolution", choices=["480", "720"], default="720", type=str,
                        help="Resolution of the model to use for video-to-world generation")
    parser.add_argument("--calibration_mode", choices=["FULL", "MINIMAL"], default="FULL",
                        help='Calibration mode controls the number of samples to use during model optimization. "FULL"'
                             'processes the entire dataset, leading to higher accuracy retention in exchange for slower'
                             'calibration. "MINIMAL" adapts the number of samples according to the model context for a'
                             'balanced optimisation performance.')
    parser.add_argument("--calibration_dataset", type=str, default="",
                        help="Optional path to a custom calibration dataset JSONL file. Defaults to modality-specific "
                             "datasets in top-level `assets` folder.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use. Activates CP if > 1.")
    return parser


def calibration_samples_required(args: argparse.Namespace, quant_config: dict) -> int:
    return 1  # TODO(rafonsorodri): Adapt samples according to model size, resolution, FPS, NATTEN, quantized precision


def process_single_generation(
    pipe: ControlVideo2WorldInference,
    input_path: str,
    control_keys: list[str],
    control_paths: dict[str, str],
    control_weights: str,
    prompt: str,
    negative_prompt: str,
    resolution: str,
    num_conditional_frames: int,
    guidance: float,
    seed: int,
) -> bool:
    log.info(f"Running ControlVideo2WorldInference"
             f"\n\tinput: {input_path}\n\tcontrol(s): {control_paths}\n\tprompt: {prompt}")
    start_time = time.time()
    video = pipe.generate_img2world(
        prompt=prompt,
        video_path=input_path,
        hint_key=control_keys,
        input_control_video_paths=control_paths,
        control_weight=control_weights,
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


def generate_video(pipe: ControlVideo2WorldInference, inference_args: argparse.Namespace, samples: list[CalibrationSample]) -> None:
    for idx in tqdm.trange(len(samples), disable=False, desc="Processing batch item"):
        sample = samples[idx]

        process_single_generation(
            pipe=pipe,
            input_path=sample.video_path,
            control_keys=sample.control_keys,
            control_paths=sample.control_paths,
            control_weights=sample.control_weights_str,
            prompt=sample.prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            resolution=inference_args.resolution_hw,
            num_conditional_frames=inference_args.num_conditional_frames,
            guidance=inference_args.guidance,
            seed=inference_args.seed,
        )


def prepare_calibration_data(args: argparse.Namespace, quant_config) -> list[CalibrationSample]:
    # Collect calibration data
    assert os.path.exists(args.calibration_dataset), f"Calibration dataset does not exist: {args.calibration_dataset}"

    if args.calibration_dataset.endswith(".jsonl"):
        dataset = []
        with open(args.calibration_dataset, 'rt') as fd:
            for line in fd.read().splitlines():
                if line and not line.startswith("#"):
                    dataset.append(json.loads(line))
    else:
        with open(args.calibration_dataset, 'rt') as fd:
            dataset = json.load(fd)
    assert isinstance(dataset, Sequence) and dataset, f"Did not find calibration samples in {args.calibration_dataset}."

    samples: list[CalibrationSample] = []
    for idx, sample in enumerate(dataset):
        try:
            samples.append(CalibrationSample.from_json(sample, modality=args.model.hint_keys))
        except ValueError as ex:
            log.warning(f"Skipping item {idx}: {ex}")
            continue

    num_samples = calibration_samples_required(args, quant_config)
    if args.calibration_mode == "FULL":
        if len(samples) <= num_samples:
            log.warning(
                "Inadequate number of samples detected. Consider using a larger calibration dataset for an improved "
                "optimization outcome."
            )
    else:  # MINIMAL
        if len(samples) < num_samples:
            log.warning(
                "Inadequate number of samples detected. MINIMAL calibration for the given model context requires "
                f"{num_samples} samples."
            )
        samples = random.sample(samples, k=num_samples)
    return samples


def calibrate_dit_denoiser(pipe, args: argparse.Namespace, quant_config):

    samples = prepare_calibration_data(args, quant_config)

    inference_args = argparse.Namespace(**{
        "name": "calibration",
        "resolution_hw": args.resolution,
        "num_conditional_frames": args.num_conditional_frames,
        "guidance": args.guidance,
        "seed": args.seed,
    })

    dit_controlnet = pipe.model.net  # Contains base and control branches

    # Fuse QKV projection
    fuse_qkv_projections(dit_controlnet, args.model.is_multicontrol)

    def forward_loop(dit_controlnet):
        pipe.model.net = dit_controlnet
        generate_video(pipe, inference_args, samples)
        return pipe.model.net

    return mtq.quantize(dit_controlnet, quant_config, forward_loop)


def main(cmdargs) -> str:
    model_variant = cmdargs.model_variant if isinstance(cmdargs.model_variant, list) else [cmdargs.model_variant]
    model_meta = ModelMeta.from_text(model_variant)

    # Base args
    input_file = "optim_dit_args_multicontrol.json" if len(model_variant) > 1 else "optim_dit_args.json"
    with open(os.path.join(SCRIPTS_ROOT, "byoc_utils", input_file), "rt") as f:
        args: dict = json.load(f)
        checkpoint_base_path = '/opt/nim/workspace/'
        args["checkpoint_paths"] = {
            "edge": cmdargs.edge_checkpoint_name or checkpoint_base_path + "general/edge/ecd0ba00-d598-4f94-aa09-e8627899c431_ema_bf16.pt",
            "depth": cmdargs.depth_checkpoint_name or checkpoint_base_path + "general/depth/0f214f66-ae98-43cf-ab25-d65d09a7e68f_ema_bf16.pt",
            "seg": cmdargs.seg_checkpoint_name or checkpoint_base_path + "general/seg/fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab_ema_bf16.pt",
            "vis": cmdargs.vis_checkpoint_name or checkpoint_base_path + "general/blur/20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1_ema_bf16.pt",
        }
        args.update({
            "model": model_meta,
            "output_dir": cmdargs.output_dir,
            "resolution": cmdargs.resolution,
            "calibration_mode": cmdargs.calibration_mode,
            "calibration_dataset": cmdargs.calibration_dataset or model_meta.calibration_dataset,
        })
    if cmdargs.num_gpus > 1:
        args["num_gpus"] = cmdargs.num_gpus

    pipeline_args = PipelineArgs(**args)
    pipe = setup_pipeline(pipeline_args)

    quant_config = QUANTIZATION_MODES[cmdargs.mode.upper()]
    calib_args = argparse.Namespace(**args)
    dit_controlnet = calibrate_dit_denoiser(pipe, calib_args, quant_config)

    # Save checkpoint and return quantized pipeline
    filename_noext = os.path.join(
        cmdargs.output_dir,
        f"dit_controlnet_{model_meta.safe_name}_{cmdargs.resolution}_{cmdargs.mode}"
    )
    filename_pt = f"{filename_noext}.pt"
    log.info(f"Saving quantized checkpoint to: {filename_pt}")
    mto.save(dit_controlnet, filename_pt)
    real_stdout = sys.stdout
    with open(f"{filename_noext}.mtq-rep.txt", 'w') as sys.stdout:
        mtq.print_quant_summary(dit_controlnet)
    sys.stdout = real_stdout

    return filename_pt


if __name__ == "__main__":
    # Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = make_parser()
    cmdargs = parser.parse_args()
    main(cmdargs)
