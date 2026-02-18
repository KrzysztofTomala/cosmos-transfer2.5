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

from types import SimpleNamespace

import torch
import tensorrt as trt

from cosmos_transfer2._src.imaginaire.utils import log
from scripts.byoc_utils.block import BlockMeta
from scripts.byoc_utils.model import SHAPE_SPECS, ModelDimensions, OperationalBounds, shapes_from_spec


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Common scratch space for all loaded TRT engines
_TRT_EXISTING_CONTEXT = SimpleNamespace()
_TRT_EXISTING_CONTEXT.device_memory = None
_TRT_EXISTING_CONTEXT.execution_contexts = []


_trt2pt_dtype = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF:  torch.float16,
    trt.DataType.INT8:  torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL:  torch.bool,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.FP8:   torch.float8_e4m3fn,
    trt.DataType.BF16:  torch.bfloat16,
    trt.DataType.INT64: torch.int64,
}


def create_execution_context_from_pool(engine):
    # Currently each engine only has one profile
    num_profiles = engine.num_optimization_profiles
    req_size = 0
    for profile_idx in range(num_profiles):
        req_size = max(engine.get_device_memory_size_for_profile(profile_idx), req_size)
    if _TRT_EXISTING_CONTEXT.device_memory is None or _TRT_EXISTING_CONTEXT.device_memory.numel() < req_size:
        log.info(f"Reallocating {req_size/1024**3:.2f}G of scratch space")
        # Reallocate new scratch space
        _TRT_EXISTING_CONTEXT.device_memory = torch.empty(req_size, dtype=torch.int8, device='cuda')
        # Update the created contexts
        for context in _TRT_EXISTING_CONTEXT.execution_contexts:
            context.device_memory = _TRT_EXISTING_CONTEXT.device_memory.data_ptr()
        # Clear CUDA caches
        torch.cuda.empty_cache()
    # Create new context & attach the current scratch space
    context = engine.create_execution_context(strategy=trt.tensorrt.ExecutionContextAllocationStrategy.USER_MANAGED)
    context.device_memory = _TRT_EXISTING_CONTEXT.device_memory.data_ptr()
    # Make a record for the context created
    _TRT_EXISTING_CONTEXT.execution_contexts.append(context)
    return context


def trt_set_tensor_check(context, name, tensor, check_shape=True):
    assert tensor.is_contiguous(), f"contiguous tensor expected for `{name}`."
    expected_dtype = _trt2pt_dtype[context.engine.get_tensor_dtype(name)]
    assert expected_dtype == tensor.dtype, \
        f"incompatible dtype for tensor `{name}`: {tensor.dtype}, expected {expected_dtype}."
    if check_shape:
        assert context.set_input_shape(name, tensor.shape), \
            f"incompatible shape for tensor `{name}`: {tensor.shape}, expected {context.engine.get_tensor_shape(name)}."
    context.set_tensor_address(name, tensor.data_ptr())


def trt_engine_from_onnx_block(
    trt_builder: trt.Builder,
    onnx_path: str,
    block_meta: BlockMeta,
    dims: ModelDimensions,
    optimization_level: int = 3,
    explicit_bounds: OperationalBounds = None,
):
    # Prepare build config
    config = trt_builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    resolution_profiles = {} # {resulution: profile_idx}
    for profile_idx, (resolution, model_dim) in enumerate(dims.items()):
        resolution_bounds = (min(model_dim.H, model_dim.W), max(model_dim.H, model_dim.W))
        bounds = explicit_bounds or OperationalBounds(
            T_MIN=1,
            T_MAX=model_dim.T,
            H_MIN=resolution_bounds[0],
            W_MIN=resolution_bounds[0],
            H_MAX=resolution_bounds[1],
            W_MAX=resolution_bounds[1],
        )

        profile = optimization_profile_block(trt_builder, block_meta, model_dim, bounds)
        config.add_optimization_profile(profile)
        resolution_profiles[resolution] = profile_idx

    # Prepare graph and load block
    log.info(f"Loading ONNX block from {onnx_path}")
    trt_explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_strongly_typed = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = trt_builder.create_network(trt_explicit_batch | trt_strongly_typed)
    parser = trt.OnnxParser(network, trt_builder.logger)
    status = parser.parse_from_file(onnx_path)

    if not status:
        log.error("Failed to read ONNX due to:")
        for ierr in range(parser.num_errors):
            log.error(f'- {parser.get_error(ierr)}')
        raise RuntimeError("Failed to build TRT engine (see logs for details)")

    # Build engine
    log.info("Building TRT engine from ONNX")
    engine_serialized = trt_builder.build_serialized_network(network, config)
    log.info(f"Built TRT engine with {len(resolution_profiles)} resolution profiles: {resolution_profiles}")

    return engine_serialized, resolution_profiles


def optimization_profile_block(trt_builder, block_meta: BlockMeta, dims: ModelDimensions, bounds: OperationalBounds):
    profile = trt_builder.create_optimization_profile()

    for key in block_meta.fixed_inputs:
        profile.set_shape(key, **shapes_from_spec(SHAPE_SPECS[key], dims, bounds))

    if block_meta.is_control:
        c_shape = SHAPE_SPECS["control_B_T_H_W_D"].copy()
        if block_meta.block_index > 0:
            c_shape.insert(0, str(block_meta.block_index+1))
        profile.set_shape("c", **shapes_from_spec(c_shape, dims, bounds))

    return profile
