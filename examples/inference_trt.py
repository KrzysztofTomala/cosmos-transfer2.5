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

from pathlib import Path
from typing import Annotated, Union

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from packages._trt_plugins.context_registry import get_sm_version
import torch

from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    InferenceArguments,
    InferenceOverrides,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
    ],
    tyro.conf.ConsolidateSubcommandArgs,
]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path(s) to the inference parameter file(s).
    If multiple files are provided, run "batch" inference. The model will be loaded once and all samples run sequentially.
    If there are different hint keys across the batch, the multicontrol model will be used regardless of the each sample's hint keys.
    """
    trt_engine_dir: Path
    """Path to TRT engines"""
    trt_engine_dir_format: Path
    """Path format to multicontrol TRT engines (can be empty for single-control use)
    You may specify something like 'output/trt_{}_2B_FP8'.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""

    control: ControlUnion = EdgeConfig()
    """Control help. Run control:edge --help for more information about edge etc."""

    force_multi_control: bool = False
    """Load full multi-control model, let control branches be activated dynamically on a per-sample basis"""


def main(
    args: Args,
):
    inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    if args.setup.benchmark:
        assert len(inference_samples) > 1, "Benchmarking must be run for more than 1 sample."
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference import Control2WorldInference
    if args.force_multi_control:
        # Load full multicontrol model, route modalities according to each sample
        from cosmos_transfer2.config import CONTROL_KEYS
        batch_hint_keys = CONTROL_KEYS.copy()

    inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)
    sm_arch = f"sm{get_sm_version(torch.cuda.current_device())}"
    trt_engine_dir = str(args.trt_engine_dir / sm_arch)
    trt_engine_dir_format = str(args.trt_engine_dir_format / sm_arch)
    inference.inference_pipeline.model.load_trt(trt_engine_dir, trt_engine_dir_format)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
