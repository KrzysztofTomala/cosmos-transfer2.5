# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import logging
import os

from huggingface_hub.utils._auth import get_token


def make_parser():
    parser = argparse.ArgumentParser(
        description="Optimize a fine-tuned Cosmos-Transfer2.5 checkpoint for NIM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_variant", required=True, type=str,
                        help="Model variant to use for control-video-to-world generation, example: `edge`.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Working folder onto which to save quantized checkpoint, ONNX states, and TRT engines.")
    parser.add_argument("--checkpoint_name", type=str, default="",
                        help="Optional filename of custom checkpoint (`*.pt`) to optimise. Defaults to the registered "
                             "post-trained checkpoint.")
    parser.add_argument("--quant_mode", type=str, default="FP8", help="Quantization mode (FP8 or NVFP4)")
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
    parser.add_argument("-O", dest="optimization_level", type=int, default=3, help="TRT optimization level")
    parser.add_argument("--skip_trt_tests", action="store_true", help="Skip TRT testrun")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set the logging level")
    return parser


def quantize_checkpoint(cmdargs) -> str:
    from scripts import quantize_model
    options = [
        "--model_variant", cmdargs.model_variant,
        "--output_dir", cmdargs.output_dir,
        "--checkpoint_name", cmdargs.checkpoint_name,
        "--mode", cmdargs.quant_mode,
        "--resolution", cmdargs.resolution,
        "--calibration_mode", cmdargs.calibration_mode,
        "--calibration_dataset", cmdargs.calibration_dataset,
        "--num_gpus", str(cmdargs.num_gpus),
    ]
    calibrate_args = quantize_model.make_parser().parse_args(options)
    modelopt_checkpoint = quantize_model.main(calibrate_args)
    return modelopt_checkpoint


def convert_to_onnx(cmdargs, modelopt_checkpoint) -> str:
    from scripts import convert_pt_to_onnx
    options = [
        "--model_variant", cmdargs.model_variant,
        "--modelopt_checkpoint", modelopt_checkpoint,
        "--output_dir", cmdargs.output_dir,
        "--mode", cmdargs.quant_mode,
        "--resolution", cmdargs.resolution,
    ]
    export_args = convert_pt_to_onnx.make_parser().parse_args(options)
    onnx_dir = convert_pt_to_onnx.main(export_args)
    return onnx_dir


def build_trt_engines(cmdargs) -> str:
    from scripts import build_trt_engine
    options = [
        "--model_variant", cmdargs.model_variant,
        "--output_dir", cmdargs.output_dir,
        "--mode", cmdargs.quant_mode,
        "-O", str(cmdargs.optimization_level),
        "--resolution", cmdargs.resolution,
    ]
    if cmdargs.skip_trt_tests:
        options.append("--skip_testrun")
    build_args = build_trt_engine.make_parser().parse_args(options)
    trt_dir = build_trt_engine.main(build_args)
    return trt_dir


def main(cmdargs):
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    os.environ.setdefault("LOGURU_LEVEL", args.log_level)

    # Lazy-load after logging level is set
    from cosmos_transfer2._src.imaginaire.utils import log
    log.info(f"Preparing to optimize Cosmos-Transfer2.5-2B. Working folder: {args.output_dir}")

    # Check if is authenticated with HF before download-triggering first-party imports for clean error message
    assert get_token(), "No HF credentials set, set environment variable `HF_TOKEN` or authenticate via `hf auth login`"

    # Quantize custom checkpoint
    modelopt_checkpoint = quantize_checkpoint(cmdargs)
    log.info(f"Quantized model saved to: {os.path.join(cmdargs.output_dir, modelopt_checkpoint)}")

    # Convert DiT blocks to ONNX
    onnx_dir = convert_to_onnx(cmdargs, modelopt_checkpoint)
    log.info(f"ONNX States saved to: {onnx_dir}")

    # build TRT engines from ONNX
    trt_dir = build_trt_engines(cmdargs)
    log.info(f"TRT Engines saved to: {trt_dir}")


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
