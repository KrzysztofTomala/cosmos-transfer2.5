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

from scripts import quantize_model, convert_pt_to_onnx, build_trt_engine
from scripts.byoc_utils.model import VARIANTS
from scripts.byoc_utils.pipeline import QUANTIZATION_MODES


def make_parser():
    parser = argparse.ArgumentParser(
        description="Optimize a fine-tuned Cosmos-Transfer2.5 checkpoint for NIM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_variant", choices=VARIANTS, required=True, type=str,
                        help="Model variant to use for control-video-to-world generation")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Working folder onto which to save quantized checkpoint, ONNX states, and TRT engines.")
    parser.add_argument("--checkpoint_name", type=str, default="",
                        help="Optional filename of custom checkpoint (`*.pt`) to optimise. Defaults to the registered "
                             "post-trained checkpoint.")
    parser.add_argument("--quant_config", type=str, choices=list(QUANTIZATION_MODES.keys()), default="FP8",
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
    parser.add_argument("-O", dest="optimization_level", type=int, default=3, help="TRT optimization level")
    parser.add_argument("--skip_trt_tests", action="store_true", help="Skip TRT testrun")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Set the logging level")
    return parser


def quantize_checkpoint(cmdargs) -> str:
    options = [
        "--model_variant", cmdargs.model_variant,
        "--output_dir", cmdargs.output_dir,
        "--checkpoint_name", cmdargs.checkpoint_name,
        "--mode", cmdargs.quant_mode,
        "--resolution", cmdargs.resolution,
        "--calibration_mode", cmdargs.calibration_mode,
        "--calibration_dataset", cmdargs.calibration_dataset,
        "--num_gpus", cmdargs.num_gpus,
    ]
    calibrate_args = quantize_model.make_parser().parse_args(options)
    modelopt_checkpoint = quantize_model.main(calibrate_args)
    return modelopt_checkpoint


def convert_to_onnx(cmdargs, modelopt_checkpoint) -> str:
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
    options = [
        "--model_variant", cmdargs.model_variant,
        "--output_dir", cmdargs.output_dir,
        "--mode", cmdargs.quant_mode,
        "-O", cmdargs.optimization_level,
        "--resolution", cmdargs.resolution,
    ]
    if cmdargs.skip_trt_tests:
        options.append("--skip_testrun")
    build_args = build_trt_engine.make_parser().parse_args(options)
    trt_dir = build_trt_engine.main(build_args)
    return trt_dir


def main(cmdargs):
    # Set logging level based on command line argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logging.info(f"Preparing to optimize Cosmos-Transfer2.5-2B checkpoint. Working folder: `{args.output_dir}`.")

    # Quantize custom checkpoint
    modelopt_checkpoint = quantize_checkpoint(cmdargs)
    logging.info(f"Quantized model saved to: {modelopt_checkpoint}")

    # Convert DiT blocks to ONNX
    onnx_dir = convert_to_onnx(cmdargs, modelopt_checkpoint)
    logging.info(f"ONNX States saved to: {onnx_dir}")

    # build TRT engines from ONNX
    trt_dir = build_trt_engines(cmdargs)
    logging.info(f"TRT Engines saved to: {trt_dir}")


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
