import logging
import torch
import torch.distributed as dist
from copy import deepcopy
from types import SimpleNamespace
from typing import List
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch import DotProductAttention, fp8_autocast

from .minimal_a2a import MinimalA2AAttnOp

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

try:
    import sageattention
except ImportError as e:
    log.error(f"Couldn't import SageAttention: {e}")
    sageattention = None

try:
    import flashinfer_vx
except ImportError as e:
    log.error(f"Couldn't import FlashInfer Vision Extension: {e}")
    flashinfer_vx = None


_context_registry = SimpleNamespace()
_context_registry.sm_version = {}
_context_registry.cp_procgrp = {}
_context_registry.cp_scomm = torch.cuda.Stream(priority=-1)
_context_registry.cp_a2a_op = {}
_context_registry.cp_attn_graph = {}
_context_registry.loc_attn_graph_pool = torch.cuda.graph_pool_handle()
_context_registry.loc_cp_ranks = []

def get_cp_stream() -> torch.cuda.Stream:
    return _context_registry.cp_scomm

def set_cp_procgrp(cp_group_list: List[int], procgrp: dist.ProcessGroup) -> None:
    _context_registry.cp_procgrp[tuple(cp_group_list)] = procgrp

def get_cp_procgrp(cp_group_list: List[int]) -> dist.ProcessGroup:
    cp_group = tuple(cp_group_list)
    if cp_group not in _context_registry.cp_procgrp:
        raise RuntimeError(f"CP group {cp_group} not found. Forbidding creation for safety. Only comment out this error when you know what you are doing.")
        _context_registry.cp_procgrp[cp_group] = dist.new_group(ranks=cp_group)
    return _context_registry.cp_procgrp[cp_group]

def set_loc_cp_ranks(cp_group_list):
    _context_registry.loc_cp_ranks = deepcopy(cp_group_list)

def get_loc_cp_ranks():
    return _context_registry.loc_cp_ranks

def get_a2a_op(cp_group_list, backend="sageattn") -> MinimalA2AAttnOp:
    cp_group = tuple(cp_group_list)
    if cp_group not in _context_registry.cp_a2a_op:
        if backend == "sageattn":
            loc_attn = lambda q, k, v: sageattention.sageattn(q, k, v, tensor_layout="NHD")
        elif backend == "flashinfer_vx":
            loc_attn = lambda q, k, v: flashinfer_vx.qattn(q, k, v)
        elif backend == "torch":
            loc_attn = lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)
        _context_registry.cp_a2a_op[cp_group] = MinimalA2AAttnOp(loc_attn)
        if len(cp_group) > 1:
            _context_registry.cp_a2a_op[cp_group].set_context_parallel_group(get_cp_procgrp(cp_group_list),
                                                                             [],
                                                                             get_cp_stream())
    return _context_registry.cp_a2a_op[cp_group]

def get_graphed_attn_op(op_qkv, custom_traits, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    B, S, H, D = q.shape
    graph_traits = (B, S, H, D, custom_traits)
    if graph_traits not in _context_registry.cp_attn_graph:
        _context_registry.cp_attn_graph[graph_traits] = torch.cuda.make_graphed_callables(
            op_qkv,
            (q, k, v),
            num_warmup_iters=4,
            pool=_context_registry.loc_attn_graph_pool,
        )
    return _context_registry.cp_attn_graph[graph_traits]

def get_sm_version(rank: int) -> int:
    if rank not in _context_registry.sm_version:
        gpu_props = torch.cuda.get_device_properties(rank)
        sm_version = gpu_props.major * 10 + gpu_props.minor
        _context_registry.sm_version[rank] = sm_version
    return _context_registry.sm_version[rank]
