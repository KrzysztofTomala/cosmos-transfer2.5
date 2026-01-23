from typing import Tuple
from types import SimpleNamespace

import logging
import torch

import tensorrt as trt
import tensorrt.plugin as trtp

from .context_registry import get_sm_version, get_a2a_op, get_loc_cp_ranks

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Global context variables
_trt_logger = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(_trt_logger)
pyt_stream = torch.cuda.current_stream()
trt_stream = torch.cuda.Stream()

# Common scratch space for all loaded TRT engines
_trt_existing_context = SimpleNamespace()
_trt_existing_context.device_memory = None
_trt_existing_context.execution_contexts = []


_trt2pt_dtype = {
    # TODO: Merge with export_attn.py
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
    req_size = engine.get_device_memory_size_for_profile(0)
    if _trt_existing_context.device_memory is None or _trt_existing_context.device_memory.numel() < req_size:
        log.info(f"Reallocating {req_size/1024**3:.2f}G of scratch space")
        # Reallocate new scratch space
        _trt_existing_context.device_memory = torch.empty(req_size, dtype=torch.int8, device='cuda')
        # Update the created contexts
        for context in _trt_existing_context.execution_contexts:
            context.device_memory = _trt_existing_context.device_memory.data_ptr()
        # Clear CUDA caches
        torch.cuda.empty_cache()
    # Create new context & attach the current scratch space
    context = engine.create_execution_context(strategy=trt.tensorrt.ExecutionContextAllocationStrategy.USER_MANAGED)
    context.device_memory = _trt_existing_context.device_memory.data_ptr()
    # Make a record for the context created
    _trt_existing_context.execution_contexts.append(context)
    return context

def trt_set_tensor_check(context, name, tensor, check_shape=True):
    assert tensor.is_contiguous(), f"contiguous tensor expected: {name}"
    assert trt_get_tensor_dtype(context, name) == tensor.dtype, f"incompatible dtype for tensor {name}: {tensor.dtype}"
    if check_shape:
        assert context.set_input_shape(name, tensor.shape), f"incompatible shape for tensor {name}: {tensor.shape}, expected: {context.engine.get_tensor_shape(name)}. min,opt,max: {context.engine.get_tensor_profile_shape(name,0)}"
    context.set_tensor_address(name, tensor.data_ptr())

def trt_get_tensor_dtype(context, name):
    return _trt2pt_dtype[context.engine.get_tensor_dtype(name)]

# Utils for attention plugins

def _recast(t):
    t_ = t
    if t.dtype == trt.DataType.BF16:
        # torch.as_tensor would fail inferring BF16 from __cuda_array_interface__ due to lack of such support in NumPy
        # have to manually workaround dtype
        t_._immutable = False
        t_.dtype = trt.DataType.HALF
        t_._immutable = True
        return torch.as_tensor(t_, device='cuda').view(torch.bfloat16)
    elif t.dtype == trt.DataType.FP8:
        # torch.as_tensor would fail inferring FP8 from __cuda_array_interface__ due to lack of such support in NumPy
        # have to manually workaround dtype
        t_._immutable = False
        t_.dtype = trt.DataType.INT8
        t_._immutable = True
        return torch.as_tensor(t_, device='cuda').view(torch.float8_e4m3fn)
    else:
        return torch.as_tensor(t_, device='cuda')

# Attention plugin #1: TE Attention

try:
    @trtp.register("Cosmos::MultiheadAttention")
    def fmha_plugin_v3(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        out_desc = q.like()
        # B S H D -> B S (H D)
        batch, max_seq, num_heads, head_dim = q.shape_expr
        out_desc.shape_expr = [batch, max_seq, num_heads * head_dim]
        # TODO: Allocate TE scratch space within TRT
        return out_desc

    @trtp.impl("Cosmos::MultiheadAttention")
    def fmha_plugin_v3_impl(
        q: trtp.Tensor,
        k: trtp.Tensor,
        v: trtp.Tensor,
        outputs: Tuple[trtp.Tensor],
        stream: int
    ) -> None:
        # Prepare Tensors
        q_t, k_t, v_t, out_t = map(_recast, (q, k, v, outputs[0]))

        # Prepare env
        ext_stream = torch.cuda.ExternalStream(stream)
        cp_group72 = get_loc_cp_ranks()
        if len(cp_group72) < 1:
            cp_group72 = [q_t.device.index]

        with torch.cuda.stream(ext_stream):
            # Prepare Op
            if get_sm_version(q_t.device.index) // 10 != 10:
                op = get_a2a_op(cp_group72)
            else:
                op = get_a2a_op(cp_group72, backend="flashinfer_vx")

            # Execute
            out1 = op(q_t, k_t, v_t)
            out_t.copy_(out1)

except Exception as e:
    log.error(f"Cannot create MultiheadAttention plugin: {e}.")
    raise ImportError(f"Cannot create MultiheadAttention plugin: {e}.")
