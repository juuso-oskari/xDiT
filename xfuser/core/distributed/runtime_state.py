from abc import ABCMeta
import importlib
import inspect
import random
from typing import List, Optional

import numpy as np
import torch
from torch.cuda import manual_seed as device_manual_seed
from torch.cuda import manual_seed_all as device_manual_seed_all
import diffusers
from diffusers import DiffusionPipeline
import torch.distributed
import torch.distributed as dist

try:
    import torch_musa
    from torch_musa.core.random import manual_seed as device_manual_seed
    from torch_musa.core.random import manual_seed_all as device_manual_seed_all
except ModuleNotFoundError:
    pass

import xfuser.envs as envs
from xfuser.envs import PACKAGES_CHECKER

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None
if envs._is_npu():
    from torch.npu import manual_seed as device_manual_seed
    from torch.npu import manual_seed_all as device_manual_seed_all

from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.attention_schedule import AttentionSchedule, GemmPrecisionSchedule
from xfuser.config.config import (
    ParallelConfig,
    RuntimeConfig,
    InputConfig,
    EngineConfig,
)
from xfuser.logger import init_logger
from .parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_pp_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from xfuser.config.args import xFuserArgs

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()
_FP8_COMMS_SAFETY_FACTOR = 0.85


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_manual_seed(seed)
    device_manual_seed_all(seed)


class Fp8CommsModelState:
    """Per-transformer FP8 comms calibration state (one entry per self-attn layer)."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.q_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.k_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.v_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.o_running_max = torch.zeros(num_layers, dtype=torch.float32)
        self.synced = False

    def to_device_(self, device: torch.device):
        self.q_running_max = self.q_running_max.to(device)
        self.k_running_max = self.k_running_max.to(device)
        self.v_running_max = self.v_running_max.to(device)
        self.o_running_max = self.o_running_max.to(device)


class Fp8CommsState:
    """Holds all state for FP8 Ulysses all-to-all communication.

    Per-layer scales live on each attn1 module as compile-friendly buffers; this class
    holds per-model running amaxes during calibration only.
    """
    def __init__(self, fixed_scale: Optional[float] = None):
        self.fixed_scale = fixed_scale
        self._models: dict[int, Fp8CommsModelState] = {}
        self.calibrated_model_ids: set = set()

    def register_model(self, model, num_layers: int) -> None:
        """Register a transformer for per-layer FP8 comms calibration."""
        model_id = id(model)
        if model_id in self._models:
            return
        self._models[model_id] = Fp8CommsModelState(num_layers)
        if self.fixed_scale is not None:
            self.apply_fixed_scales_to_model(model)
            self._models[model_id].synced = True
            self.calibrated_model_ids.add(model_id)

    def get_model_state(self, model) -> Optional[Fp8CommsModelState]:
        return self._models.get(id(model))

    def apply_fixed_scales_to_model(self, model) -> None:
        """Broadcast a fixed scale to all self-attention layer buffers."""
        scale = float(self.fixed_scale)
        for block in model.blocks:
            block.attn1.fp8_q_scale.fill_(scale)
            block.attn1.fp8_k_scale.fill_(scale)
            block.attn1.fp8_v_scale.fill_(scale)
            block.attn1.fp8_o_scale.fill_(scale)

    def update_running_max(
        self,
        model,
        layer_idx: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Update running amaxes in-place for one layer. Safe inside compiled region when unsynced."""
        model_state = self._models.get(id(model))
        if model_state is None or model_state.synced:
            return
        idx = layer_idx.reshape(-1).long()
        q_amax = q.abs().amax().reshape(1)
        k_amax = k.abs().amax().reshape(1)
        v_amax = v.abs().amax().reshape(1)
        model_state.q_running_max.index_copy_(
            0,
            idx,
            torch.maximum(model_state.q_running_max.index_select(0, idx), q_amax),
        )
        model_state.k_running_max.index_copy_(
            0,
            idx,
            torch.maximum(model_state.k_running_max.index_select(0, idx), k_amax),
        )
        model_state.v_running_max.index_copy_(
            0,
            idx,
            torch.maximum(model_state.v_running_max.index_select(0, idx), v_amax),
        )

    def update_o_running_max(
        self,
        model,
        layer_idx: torch.Tensor,
        o_descale: torch.Tensor,
    ):
        """Update o running amaxes in-place for one layer. Safe inside compiled region when unsynced."""
        model_state = self._models.get(id(model))
        if model_state is None or model_state.synced:
            return
        idx = layer_idx.reshape(-1).long()
        o_amax = o_descale.abs().amax().reshape(1)
        model_state.o_running_max.index_copy_(
            0,
            idx,
            torch.maximum(model_state.o_running_max.index_select(0, idx), o_amax),
        )

    def _scatter_scales_to_model(
        self,
        model,
        q_scales: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        o_scales: torch.Tensor,
    ):
        for i, block in enumerate(model.blocks):
            block.attn1.fp8_q_scale.copy_(q_scales[i : i + 1])
            block.attn1.fp8_k_scale.copy_(k_scales[i : i + 1])
            block.attn1.fp8_v_scale.copy_(v_scales[i : i + 1])
            block.attn1.fp8_o_scale.copy_(o_scales[i : i + 1])

    def to_device_(self, device: torch.device):
        for model_state in self._models.values():
            model_state.to_device_(device)


class RuntimeState(metaclass=ABCMeta):
    attention_backend: AttentionBackendType = AttentionBackendType.SDPA_FLASH
    cross_attention_backend: Optional[AttentionBackendType] = None
    fp8_comms: Optional[Fp8CommsState] = None        # None = disabled
    parallel_config: ParallelConfig
    runtime_config: RuntimeConfig
    input_config: InputConfig
    num_pipeline_patch: int
    ready: bool = False

    def __init__(self, config: EngineConfig):
        self.parallel_config = config.parallel_config
        self.runtime_config = config.runtime_config
        self.input_config = InputConfig()
        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        self.ready = False

        self._check_distributed_env(config.parallel_config)
        attention_backend = self._select_attention_backend(config)
        self.set_attention_backend(attention_backend)
        cross_attention_backend = self._select_cross_attention_backend(config)
        self.set_cross_attention_backend(cross_attention_backend)
        self._init_fp8_comms(config)

    def is_ready(self):
        return self.ready

    def _check_distributed_env(
        self,
        parallel_config: ParallelConfig,
    ):
        if not model_parallel_is_initialized():
            logger.warning("Model parallel is not initialized, initializing...")
            if not torch.distributed.is_initialized():
                init_distributed_environment()
            initialize_model_parallel(
                data_parallel_degree=parallel_config.dp_degree,
                classifier_free_guidance_degree=parallel_config.cfg_degree,
                sequence_parallel_degree=parallel_config.sp_degree,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_degree=parallel_config.tp_degree,
                pipeline_parallel_degree=parallel_config.pp_degree,
                fully_shard_degree=parallel_config.fs_degree,
                vae_parallel_size=parallel_config.vae_parallel_size,
                use_parallel_vae=parallel_config.use_parallel_vae,
            )

    def destroy_distributed_env(self):
        if model_parallel_is_initialized():
            destroy_model_parallel()
        destroy_distributed_environment()

    def set_attention_backend(self, attention_backend: str | AttentionBackendType):
        """
        Set the attention backend for the current environment.
        Given attention_backend can be either AttentionBackendType or a string with the name of the backend.
        """
        if isinstance(attention_backend, str):
            try:
                attention_backend = AttentionBackendType[attention_backend.upper()]
            except:
                pass

        if not isinstance(attention_backend, AttentionBackendType):
            raise ValueError(f"Value '{attention_backend}' is not a valid attention backend.")

        self._check_if_backend_compatible_with_current_configuration(attention_backend)
        self.attention_backend = attention_backend
        logger.warning("Using {} as attention backend.".format(self.attention_backend.name))
        if attention_backend in [AttentionBackendType.FLASH_3_FP8, AttentionBackendType.AITER_FP8, AttentionBackendType.NVTE_FP8, AttentionBackendType.FLASH_4_FP4, AttentionBackendType.AITER_MLA, AttentionBackendType.AITER_MXFP4, AttentionBackendType.AITER_F4F4]:
            logger.warning("Low-precision attention backend is enabled. This may cause poor quality outputs, consider using hybrid attention if possible.")


    def _init_fp8_comms(self, config: EngineConfig):
        if not config.runtime_config.use_fp8_comms:
            self.fp8_comms = None
            return
        ulysses_degree = config.parallel_config.sp_config.ulysses_degree or 1
        if ulysses_degree <= 1:
            logger.warning(
                "--use_fp8_comms is set but ulysses_degree <= 1. "
                "FP8 communication will not be applied."
            )
            self.fp8_comms = None
            return
        scale = config.runtime_config.fp8_comms_scale
        if scale is not None:
            logger.warning(f"FP8 communication enabled with fixed scale {scale}.")
        else:
            logger.warning(
                "FP8 communication enabled with dynamic per-layer scaling "
                "(calibrated before inference)."
            )
        self.fp8_comms = Fp8CommsState(fixed_scale=scale)

    def sync_fp8_comms(self, model=None):
        """All-reduce per-layer running amaxes and scatter scales into attn1 buffers.

        Call outside the compiled region after a calibration forward pass.
        """
        fp8_comms = self.fp8_comms
        if fp8_comms is None or fp8_comms.fixed_scale is not None or model is None:
            return
        model_state = fp8_comms.get_model_state(model)
        if model_state is None or model_state.synced:
            return
        if (
            model_state.q_running_max.max() == 0
            and model_state.k_running_max.max() == 0
            and model_state.v_running_max.max() == 0
        ):
            return
        from xfuser.core.distributed.attention_backend import AITER_FP8_DTYPE
        dtype_max = torch.finfo(AITER_FP8_DTYPE).max
        maxes = torch.stack(
            [
                model_state.q_running_max,
                model_state.k_running_max,
                model_state.v_running_max,
                model_state.o_running_max
            ],
            dim=0,
        )
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX, group=PROCESS_GROUP.ULYSSES_PG)
        scales = maxes.clamp(min=1e-6) / (dtype_max * _FP8_COMMS_SAFETY_FACTOR)
        fp8_comms._scatter_scales_to_model(model, scales[0], scales[1], scales[2], scales[3])
        model_state.q_running_max.zero_()
        model_state.k_running_max.zero_()
        model_state.v_running_max.zero_()
        model_state.o_running_max.zero_()
        model_state.synced = True
        fp8_comms.calibrated_model_ids.add(id(model))
        if dist.get_rank() == 0:
            q_scales, k_scales, v_scales, o_scales = scales[0], scales[1], scales[2], scales[3]
            print(
                f"[fp8_comms] {model.__class__.__name__} per-layer scales synced: "
                f"q=[{q_scales.min().item():.6f}, {q_scales.max().item():.6f}] "
                f"k=[{k_scales.min().item():.6f}, {k_scales.max().item():.6f}] "
                f"v=[{v_scales.min().item():.6f}, {v_scales.max().item():.6f}] "
                f"o=[{o_scales.min().item():.6f}, {o_scales.max().item():.6f}] "
                f"(amax q={maxes[0].max().item():.4f} k={maxes[1].max().item():.4f} "
                f"v={maxes[2].max().item():.4f} o={maxes[3].max().item():.4f})"
            )

    def reset_fp8_comms_calibration(self, model=None):
        """Reset per-layer scales for recalibration. No-op if already calibrated."""
        fp8_comms = self.fp8_comms
        if fp8_comms is None or fp8_comms.fixed_scale is not None or model is None:
            return
        model_id = id(model)
        if model_id in fp8_comms.calibrated_model_ids:
            return
        model_state = fp8_comms.get_model_state(model)
        if model_state is None:
            return
        for block in model.blocks:
            block.attn1.fp8_q_scale.fill_(1.0)
            block.attn1.fp8_k_scale.fill_(1.0)
            block.attn1.fp8_v_scale.fill_(1.0)
            block.attn1.fp8_o_scale.fill_(1.0)
        model_state.q_running_max.zero_()
        model_state.k_running_max.zero_()
        model_state.v_running_max.zero_()
        model_state.o_running_max.zero_()
        model_state.synced = False

    def set_cross_attention_backend(self, cross_attention_backend: Optional[str | AttentionBackendType]):
        """
        Set the cross-attention backend. When None, cross-attention will use the main attention_backend.
        """
        if cross_attention_backend is None:
            self.cross_attention_backend = None
            return

        if isinstance(cross_attention_backend, str):
            try:
                cross_attention_backend = AttentionBackendType[cross_attention_backend.upper()]
            except:
                pass

        if not isinstance(cross_attention_backend, AttentionBackendType):
            raise ValueError(f"Value '{cross_attention_backend}' is not a valid attention backend.")

        self._check_if_backend_compatible_with_current_configuration(cross_attention_backend)
        self.cross_attention_backend = cross_attention_backend
        logger.warning("Using {} as cross-attention backend.".format(self.cross_attention_backend.name))

    def get_cross_attention_backend(self) -> AttentionBackendType:
        """
        Returns the backend to use for cross-attention.
        Falls back to the main attention_backend when no separate cross-attention backend is configured.
        """
        if self.cross_attention_backend is not None:
            return self.cross_attention_backend
        return self.attention_backend

    def _select_cross_attention_backend(self, engine_config: Optional[EngineConfig] = None) -> Optional[AttentionBackendType]:
        """
        Select the cross-attention backend from config. Returns None if not explicitly set
        (meaning the main attention_backend will be used).
        """
        if engine_config and engine_config.runtime_config.cross_attention_backend:
            return AttentionBackendType[engine_config.runtime_config.cross_attention_backend.upper()]
        return None

    def _select_attention_backend(self, engine_config: Optional[EngineConfig] = None):
        """
        Select the best attention backend for the current environment.
        """
        if engine_config and engine_config.runtime_config.attention_backend:
            backend = AttentionBackendType[engine_config.runtime_config.attention_backend.upper()]

        elif envs._is_hip():
            if env_info["has_aiter"] and PACKAGES_CHECKER._on_rdna4():
                backend = AttentionBackendType.AITER_FLYDSL
            elif env_info["has_aiter"]:
                backend = AttentionBackendType.AITER
            elif env_info["has_flash_attn"]:
                backend = AttentionBackendType.FLASH
            else:
                backend = AttentionBackendType.SDPA

        elif env_info["has_flash_attn_4"]:
            backend = AttentionBackendType.FLASH_4
        elif env_info["has_flash_attn_3"]:
            backend = AttentionBackendType.FLASH_3
        elif torch.backends.cudnn.is_available():
            backend = AttentionBackendType.CUDNN
        elif env_info["has_flash_attn"]:
            backend = AttentionBackendType.FLASH
        elif env_info["has_npu_flash_attn"]:
            backend = AttentionBackendType.NPU
        else:
            backend = AttentionBackendType.SDPA

        return backend

    def _check_if_backend_compatible_with_current_configuration(self, attention_backend: AttentionBackendType):
        """
        Check if the selected attention backend is compatible with the current configuration.
        """
        if attention_backend in [AttentionBackendType.SDPA,
                                 AttentionBackendType.SDPA_MATH,
                                 AttentionBackendType.FLASH_4,
                                 AttentionBackendType.FLASH_4_FP4,
                                 AttentionBackendType.AITER_FP8,
                                 AttentionBackendType.AITER_MLA,
                                 AttentionBackendType.AITER_SAGE,
                                 AttentionBackendType.AITER_SPARSE_SAGE,
                                 AttentionBackendType.AITER_SPARGE,
                                 AttentionBackendType.AITER_SAGE_V2,
                                 AttentionBackendType.AITER_SPARSE_SAGE_V2,
                                 AttentionBackendType.AITER_SPARGE_ASM,
                                 AttentionBackendType.AITER_SPARGE_ASM_V2,
                                 AttentionBackendType.AITER_SPARGE_ASM_V2_AFFINE_SORTED,
                                 AttentionBackendType.AITER_SPARGE_ASM_FP8,
                                 AttentionBackendType.AITER_SPARGE_ASM_FP8_AFFINE_SORTED,
                                 AttentionBackendType.AITER_SPARGE_V2,
                                 AttentionBackendType.AITER_I8FP8,
                                 AttentionBackendType.AITER_MXFP4,
                                 AttentionBackendType.AITER_F4F4,
                                 AttentionBackendType.AITER_FLYDSL,
                                 AttentionBackendType.FLEX_BLOCK_ATTN,
                                 AttentionBackendType.FLEX_BLOCK_SPARGE]:
            if self.parallel_config.ring_degree > 1:
                # Ring parallelism merges per-rank attention outputs via LSE, so
                # the wrapper must expose return_lse (and, for AITER_SAGE,
                # smooth_k, shipped with the LSE-correction fix needed for
                # correct merging). Pick (module, symbol, required params)
                # for the selected backend and validate the wrapper's signature.
                if attention_backend == AttentionBackendType.AITER_SAGE:
                    module_path = "aiter.ops.triton.attention.fav3_sage"
                    symbol = "fav3_sage_wrapper_func"
                    required = ("return_lse", "smooth_k")
                elif attention_backend == AttentionBackendType.AITER_SAGE_V2:
                    module_path = "aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper"
                    symbol = "fav3_sage_mxfp4_wrapper"
                    required = ("return_lse",)
                else:
                    raise RuntimeError(
                        "Selected attention backend does not support ring parallelism."
                    )
                try:
                    fn = getattr(importlib.import_module(module_path), symbol)
                except ImportError:
                    raise RuntimeError(
                        f"{attention_backend.value} attention is not available, "
                        "please update AITER"
                    ) from None
                try:
                    params = inspect.signature(fn).parameters
                except (AttributeError, TypeError):
                    params = {}
                missing = [p for p in required if p not in params]
                if missing:
                    raise RuntimeError(
                        f"{attention_backend.value} attention is missing {missing} "
                        "required for ring parallelism, please update AITER"
                    )
        if attention_backend == AttentionBackendType.AITER_FP8:
            try:
                from aiter import flash_attn_fp8_pertensor_func
            except ImportError:
                raise RuntimeError("AITER fp8 flash attention is not available, please update AITER")
        elif attention_backend == AttentionBackendType.NVTE_FP8:
            if not env_info.get("has_transformer_engine"):
                raise RuntimeError(
                    "Transformer Engine FP8 attention requires transformer-engine"
                )
        elif attention_backend == AttentionBackendType.AITER_MLA:
            try:
                from aiter import get_ps_metadata_info_v1, get_ps_metadata_v1, mla_prefill_ps_asm_fwd, mla_reduce_v1
            except ImportError:
                raise RuntimeError("AITER MLA attention is not available, please update AITER") from None
        elif attention_backend == AttentionBackendType.AITER_SAGE:
            try:
                import aiter.ops.triton.attention
                from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
            except ImportError:
                raise RuntimeError("AITER Sage attention is not available, please update AITER") from None
        elif attention_backend == AttentionBackendType.AITER_SPARSE_SAGE:
            try:
                from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
            except ImportError:
                raise RuntimeError("AITER Sparse Sage attention is not available, please update AITER") from None
        elif attention_backend == AttentionBackendType.AITER_SAGE_V2:
            try:
                from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import fav3_sage_mxfp4_wrapper
            except ImportError:
                raise RuntimeError("AITER Sage V2 attention is not available, please update AITER") from None
        elif attention_backend == AttentionBackendType.FLASH_4_FP4:
            if not env_info.get("has_flash_attn_4_fp4"):
                raise RuntimeError(
                    "Flash Attention V4 FP4 is not available. Requires Blackwell GPU (SM >= 10.0) "
                    "and the hao-ai-lab/flash-attention-fp4 fork with nvidia-cutlass-dsl."
                )
        elif attention_backend == AttentionBackendType.SAGE:
            if not env_info["has_sage"]:
                raise RuntimeError("SageAttention is not available, please install SageAttention.")
        elif attention_backend == AttentionBackendType.AITER_SPARGE_ASM:
            try:
                from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
                from aiter.ops.mha import flash_attn_i8fp8_sparse_pertensor_func
            except ImportError:
                raise RuntimeError(
                    "AITER Sparge ASM attention is not available; this backend "
                    "needs the hand-written gfx950 .co (fwd_hd128_i8fp8_sparse.co) "
                    "and aiter.ops.mha.flash_attn_i8fp8_sparse_pertensor_func."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_SPARGE_ASM_V2:
            try:
                from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
                from aiter.ops.mha import flash_attn_mxfp4_sparse_pertensor_func
                from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_mxfp4
                from aiter.ops.triton.quant.sage_attention_quant_fp8_input_wrapper import (
                    sage_quant_mxfp4_fp8_input,
                )
            except ImportError:
                raise RuntimeError(
                    "AITER Sparge ASM V2 (mxfp4) attention is not available; this "
                    "backend needs the hand-written gfx950 .co "
                    "(fwd_hd128_mxfp4_sparse.co) and "
                    "aiter.ops.mha.flash_attn_mxfp4_sparse_pertensor_func plus "
                    "aiter.ops.triton.quant.sage_attention_quant_wrappers."
                    "sage_quant_mxfp4."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_SPARGE_ASM_V2_AFFINE_SORTED:
            try:
                from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
                from aiter.ops.mha import (
                    flash_attn_mxfp4_sparse_pertensor_func,
                    fmha_v3_fwd_mxfp4_sparse_sorted,
                )
                from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_mxfp4
                from aiter.ops.triton.quant.sage_attention_quant_fp8_input_wrapper import (
                    sage_quant_mxfp4_fp8_input,
                )
            except ImportError:
                raise RuntimeError(
                    "AITER Sparge ASM V2 Affine Sorted (mxfp4) attention is not "
                    "available; this backend needs the hand-written gfx950 .co "
                    "(fwd_hd128_mxfp4_sparse_sorted.co) and the sorted entry "
                    "aiter.ops.mha.fmha_v3_fwd_mxfp4_sparse_sorted (reached via "
                    "flash_attn_mxfp4_sparse_pertensor_func(dispatch='sorted')) plus "
                    "aiter.ops.triton.quant.sage_attention_quant_wrappers.sage_quant_mxfp4."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_SPARGE_ASM_FP8:
            try:
                from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
                from aiter.ops.mha import flash_attn_fp8_sparse_pertensor_func
            except ImportError:
                raise RuntimeError(
                    "AITER Sparge ASM FP8 attention is not available; this backend "
                    "needs the hand-written gfx950 .co (fwd_hd128_fp8_sparse.co) "
                    "and aiter.ops.mha.flash_attn_fp8_sparse_pertensor_func."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_SPARGE_ASM_FP8_AFFINE_SORTED:
            try:
                from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
                from aiter.ops.mha import (
                    flash_attn_fp8_sparse_pertensor_func,
                    fmha_v3_fwd_fp8_sparse_persistent,
                )
            except ImportError:
                raise RuntimeError(
                    "AITER Sparge ASM FP8 Affine Sorted attention is not available; "
                    "this backend needs the hand-written gfx950 .co "
                    "(fwd_hd128_fp8_sparse_affine_sorted.co) and the persistent "
                    "entry aiter.ops.mha.fmha_v3_fwd_fp8_sparse_persistent (reached "
                    "via flash_attn_fp8_sparse_pertensor_func(dispatch='affine_sorted'))."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_I8FP8:
            try:
                from aiter.ops.mha import flash_attn_i8fp8_pertensor_func
            except ImportError:
                raise RuntimeError(
                    "AITER i8fp8 ASM attention is not available; this backend "
                    "needs the dense gfx950 .co (fwd_hd128_i8fp8.co) and "
                    "aiter.ops.mha.flash_attn_i8fp8_pertensor_func (the dense "
                    "sibling of flash_attn_i8fp8_sparse_pertensor_func)."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_MXFP4:
            try:
                from aiter.ops.mha import flash_attn_mxfp4_pertensor_func
                from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_mxfp4
                from aiter.ops.triton.quant.sage_attention_quant_fp8_input_wrapper import (
                    sage_quant_mxfp4_fp8_input,
                )
            except ImportError:
                raise RuntimeError(
                    "AITER mxfp4 ASM attention is not available; this backend "
                    "needs the dense gfx950 .co (fwd_hd128_mxfp4.co), "
                    "aiter.ops.mha.flash_attn_mxfp4_pertensor_func, "
                    "aiter.ops.triton.quant.sage_attention_quant_wrappers.sage_quant_mxfp4, "
                    "and aiter.ops.triton.quant.sage_attention_quant_fp8_input_wrapper."
                    "sage_quant_mxfp4_fp8_input for fp8-comms input."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_F4F4:
            try:
                from aiter.ops.mha import flash_attn_f4f4_pertensor_func
                from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
                    sage_quant_f4f4,
                    _pack_v_mxfp4_colmajor,
                )
            except ImportError:
                raise RuntimeError(
                    "AITER f4f4 ASM attention is not available; this backend needs "
                    "the f4f4 gfx950 .co (fwd_hd128_f4f4.co) via its dedicated launch "
                    "path aiter.ops.mha.flash_attn_f4f4_pertensor_func (the compile-op "
                    "fmha_v3_fwd_f4f4; no AITER_FMHA_F4F4 env var) and the in-tree fp4 "
                    "quantizer aiter.ops.triton.quant.sage_attention_quant_wrappers."
                    "sage_quant_f4f4 (plus its shared V packer _pack_v_mxfp4_colmajor "
                    "for the mxfp4-comms path)."
                ) from None
        elif attention_backend == AttentionBackendType.AITER_SPARGE:
            msg = "AITER Sparge attention is not available, please update AITER"
            try:
                from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
                if inspect.signature(fav3_sage_wrapper_func).parameters.get("block_lut") is None:
                    raise RuntimeError(msg) from None
            except ImportError:
                raise RuntimeError(msg) from None
        elif attention_backend == AttentionBackendType.AITER_SPARGE_V2:
            msg = "AITER Sparge V2 attention is not available, please update AITER"
            try:
                from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import fav3_sage_mxfp4_wrapper
                if inspect.signature(fav3_sage_mxfp4_wrapper).parameters.get("block_lut") is None:
                    raise RuntimeError(msg) from None
            except ImportError:
                raise RuntimeError(msg) from None
        elif attention_backend == AttentionBackendType.AITER_FLYDSL:
            try:
                from aiter.ops.flydsl import flydsl_flash_attn_func
            except ImportError:
                raise RuntimeError("AITER FlyDSL attention is not available, please update AITER") from None
        elif attention_backend in (AttentionBackendType.FLEX_BLOCK_ATTN,
                                   AttentionBackendType.FLEX_BLOCK_SPARGE):
            if not env_info["has_flex_block_attn"]:
                raise RuntimeError("Flex Block Attention is not available, please install Flex Block Attention.")



class UnetRuntimeState(RuntimeState):

    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        super().__init__(config)
        self.sanity_check()

    def sanity_check(self):
        if self.parallel_config.world_size > 1:
            if not(self.parallel_config.cfg_degree == 2 and self.parallel_config.world_size == 2):
                raise RuntimeError("UnetRuntimeState only supports 2 GPUs with CFG Parallel")


class DiTRuntimeState(RuntimeState):
    patch_mode: bool
    pipeline_patch_idx: int
    vae_scale_factor: int
    vae_scale_factor_spatial: int
    vae_scale_factor_temporal: int
    backbone_patch_size: int
    pp_patches_height: Optional[List[int]]
    pp_patches_start_idx_local: Optional[List[int]]
    pp_patches_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_start_idx_local: Optional[List[int]]
    pp_patches_token_start_end_idx_global: Optional[List[List[int]]]
    pp_patches_token_num: Optional[List[int]]
    max_condition_sequence_length: int
    split_text_embed_in_sp: bool

    def __init__(self, pipeline: DiffusionPipeline, config: EngineConfig):
        self.attention_schedule: Optional[AttentionSchedule] = None
        self.schedule_total_steps: Optional[int] = None
        self.gemm_schedule: Optional[GemmPrecisionSchedule] = None
        self.gemm_schedule_total_steps: Optional[int] = None
        self.use_high_precision_gemm: bool = True
        self.step_counter: Optional[int] = None
        super().__init__(config)
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self._check_model_and_parallel_config(
            pipeline=pipeline, parallel_config=config.parallel_config
        )
        try:
            self._check_pipeline_class_name(pipeline, config)
        except Exception:
            # Keeps backward compatatability with existing pipeline classes.
            pass
        if self.fp8_comms is not None and torch.cuda.is_available():
            self.fp8_comms.to_device_(torch.device("cuda", torch.cuda.current_device()))

    def _check_pipeline_class_name(self, pipeline: DiffusionPipeline, config: EngineConfig):
        self.cogvideox = False
        self.consisid = False
        self.hunyuan_video = False
        if pipeline.__class__.__name__.startswith(("CogVideoX", "ConsisID", "HunyuanVideo", "Wan")):
            if pipeline.__class__.__name__.startswith("CogVideoX"):
                self.cogvideox = True
            elif pipeline.__class__.__name__.startswith("ConsisID"):
                self.consisid = True
            else:
                self.hunyuan_video = True
            self._set_cogvideox_parameters(
                vae_scale_factor_spatial=pipeline.vae_scale_factor_spatial,
                vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
                backbone_patch_size=pipeline.transformer.config.patch_size,
                backbone_in_channel=pipeline.transformer.config.in_channels,
                backbone_inner_dim=pipeline.transformer.config.num_attention_heads
                * pipeline.transformer.config.attention_head_dim,
            )
        elif pipeline.__class__.__name__.startswith("ZImage"):
            self._set_model_parameters(
                vae_scale_factor=pipeline.vae_scale_factor,
                backbone_patch_size=pipeline.transformer.config.all_patch_size,
                backbone_in_channel=pipeline.transformer.config.in_channels,
                backbone_inner_dim=pipeline.transformer.config.n_heads
                * pipeline.transformer.config.axes_dims[-1]
            )
        else:
            vae_scale_factor = getattr(pipeline, "vae_scale_factor", 0)
            if pipeline.__class__.__name__.startswith("Flux") and diffusers.__version__ >= '0.32':
                vae_scale_factor *= 2
            self._set_model_parameters(
                vae_scale_factor=vae_scale_factor,
                backbone_patch_size=pipeline.transformer.config.patch_size,
                backbone_in_channel=pipeline.transformer.config.in_channels,
                backbone_inner_dim=pipeline.transformer.config.num_attention_heads
                * pipeline.transformer.config.attention_head_dim,
            )

    def has_attention_schedule(self) -> bool:
        """True if a per-step attention schedule is active (e.g. for warmup/compile logic)."""
        return self.attention_schedule is not None

    def has_gemm_schedule(self) -> bool:
        """True if a per-step GEMM precision schedule is active (e.g. for warmup/compile logic)."""
        return self.gemm_schedule is not None

    def _get_active_total_steps(self) -> Optional[int]:
        attn_steps = self.schedule_total_steps
        gemm_steps = self.gemm_schedule_total_steps
        if attn_steps is not None and gemm_steps is not None and attn_steps != gemm_steps:
            raise RuntimeError(
                f"Attention and GEMM schedules must use the same total steps; got {attn_steps} and {gemm_steps}."
            )
        return attn_steps or gemm_steps

    def increment_step_counter(self):
        """
        Advance the denoising step and set per-step scheduled backends/modes when active.
        When the entire denoising process is over, the step counter is reset to 0.
        """
        if self.step_counter is None:
            return

        active_total_steps = self._get_active_total_steps()
        if active_total_steps is None:
            return

        current_step = self.step_counter
        if self.attention_schedule is not None:
            self.attention_backend = self.attention_schedule.get_backend(current_step)
        if self.gemm_schedule is not None:
            self.use_high_precision_gemm = self.gemm_schedule.is_high_precision(current_step)

        self.step_counter = self.step_counter + 1

        if self.step_counter >= active_total_steps:
            self.step_counter = 0

    def set_attention_schedule(
        self,
        attention_schedule: AttentionSchedule,
        total_steps: int,
    ) -> None:
        """
        Set a per-step attention schedule.
        When set, increment_step_counter() will use the attention_schedule to set attention_backend each step.
        """
        for backend in set(attention_schedule.backends):
            self._check_if_backend_compatible_with_current_configuration(backend)
        self.attention_schedule = attention_schedule
        self.schedule_total_steps = torch.tensor(total_steps, dtype=torch.int)
        self.step_counter = torch.tensor(0, dtype=torch.int)
        logger.warning("Per-step attention schedule enabled (total_steps=%d).", total_steps)

    def set_gemm_schedule(
        self,
        gemm_schedule: GemmPrecisionSchedule,
        total_steps: int,
    ) -> None:
        """
        Set a per-step GEMM precision schedule.
        When set, increment_step_counter() will update use_high_precision_gemm each step.
        """
        self.gemm_schedule = gemm_schedule
        self.gemm_schedule_total_steps = torch.tensor(total_steps, dtype=torch.int)
        self.step_counter = torch.tensor(0, dtype=torch.int)
        logger.warning("Per-step GEMM schedule enabled (total_steps=%d).", total_steps)

    def set_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        max_condition_sequence_length: Optional[int] = None,
        split_text_embed_in_sp: bool = True,
    ):
        self.input_config.num_inference_steps = (
            num_inference_steps or self.input_config.num_inference_steps
        )
        self.max_condition_sequence_length = max_condition_sequence_length
        self.split_text_embed_in_sp = split_text_embed_in_sp
        if self.runtime_config.warmup_steps > self.input_config.num_inference_steps:
            self.runtime_config.warmup_steps = self.input_config.num_inference_steps
        if seed is not None and seed != self.input_config.seed:
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            not self.ready
            or (height and self.input_config.height != height)
            or (width and self.input_config.width != width)
            or (batch_size and self.input_config.batch_size != batch_size)
        ):
            self._input_size_change(height, width, batch_size)

        self.ready = True

    def set_video_input_parameters(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        split_text_embed_in_sp: bool = True,
    ):
        self.input_config.num_inference_steps = (
            num_inference_steps or self.input_config.num_inference_steps
        )
        if self.runtime_config.warmup_steps > self.input_config.num_inference_steps:
            self.runtime_config.warmup_steps = self.input_config.num_inference_steps
        self.split_text_embed_in_sp = split_text_embed_in_sp
        if seed is not None and seed != self.input_config.seed:
            self.input_config.seed = seed
            set_random_seed(seed)
        if (
            not self.ready
            or (height and self.input_config.height != height)
            or (width and self.input_config.width != width)
            or (num_frames and self.input_config.num_frames != num_frames)
            or (batch_size and self.input_config.batch_size != batch_size)
        ):
            self._video_input_size_change(height, width, num_frames, batch_size)

        self.ready = True

    def _set_cogvideox_parameters(
        self,
        vae_scale_factor_spatial: int,
        vae_scale_factor_temporal: int,
        backbone_patch_size: int,
        backbone_inner_dim: int,
        backbone_in_channel: int,
    ):
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel

    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0

    def _check_model_and_parallel_config(
        self,
        pipeline: DiffusionPipeline,
        parallel_config: ParallelConfig,
    ):
        num_heads = self._get_model_attention_heads(pipeline)
        ulysses_degree = parallel_config.sp_config.ulysses_degree
        if num_heads % ulysses_degree != 0 or num_heads < ulysses_degree:
            raise RuntimeError(
                f"transformer backbone has {num_heads} heads, which is not "
                f"divisible by or smaller than ulysses_degree "
                f"{ulysses_degree}."
            )

    def _get_model_attention_heads(self, pipeline: DiffusionPipeline) -> int:
        # Allows models to override their number of attention heads, for Ulysses Anything
        if hasattr(pipeline.transformer.config, "num_attention_heads"):
            return pipeline.transformer.config.num_attention_heads
        elif "num_attention_heads" in pipeline.transformer.config:
            return pipeline.transformer.config.num_attention_heads
        elif "n_heads" in pipeline.transformer.config:
            return pipeline.transformer.config.n_heads
        else:
            raise RuntimeError(
                "Cannot find the number of attention heads in transformer config. Model is not supported."
            )

    def _set_model_parameters(
        self,
        vae_scale_factor: int,
        backbone_patch_size: int,
        backbone_inner_dim: int,
        backbone_in_channel: int,
    ):
        self.vae_scale_factor = vae_scale_factor
        self.backbone_patch_size = backbone_patch_size
        self.backbone_inner_dim = backbone_inner_dim
        self.backbone_in_channel = backbone_in_channel

    def _input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.batch_size = batch_size or self.input_config.batch_size
        self._calc_patches_metadata()
        self._reset_recv_buffer()

    def _video_input_size_change(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.input_config.height = height or self.input_config.height
        self.input_config.width = width or self.input_config.width
        self.input_config.num_frames = num_frames or self.input_config.num_frames
        self.input_config.batch_size = batch_size or self.input_config.batch_size
        if self.cogvideox:
            self._calc_cogvideox_patches_metadata()
        elif self.consisid:
            self._calc_consisid_patches_metadata()
        elif self.hunyuan_video:
            # TODO: implement the hunyuan video patches metadata
            pass
        else:
            self._calc_patches_metadata()
        self._reset_recv_buffer()

    def _calc_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor = self.vae_scale_factor
        latents_height = self.input_config.height // vae_scale_factor
        latents_width = self.input_config.width // vae_scale_factor

        if latents_height % num_sp_patches != 0:
            raise ValueError(
                "The height of the input is not divisible by the number of sequence parallel devices"
            )

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1)
            // (patch_size * num_sp_patches)
        ) * (patch_size * num_sp_patches)
        # get the number of pipeline that matches patch height requirements
        num_pipeline_patch = (
            latents_height + pipeline_patches_height - 1
        ) // pipeline_patches_height
        if num_pipeline_patch != self.num_pipeline_patch:
            logger.warning(
                f"Pipeline patches num changed from "
                f"{self.num_pipeline_patch} to {num_pipeline_patch} due "
                f"to input size and parallelisation requirements"
            )
        pipeline_patches_height_list = [
            pipeline_patches_height for _ in range(num_pipeline_patch - 1)
        ]
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met "
            )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]

        pp_patches_height = [
            sp_patches_height[sp_patch_idx]
            for sp_patches_height in pp_sp_patches_height
        ]
        pp_patches_start_idx_local = [0] + [
            sum(pp_patches_height[:i]) for i in range(1, len(pp_patches_height) + 1)
        ]
        pp_patches_start_end_idx_global = [
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_token_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )
        self.pp_patches_token_num = pp_patches_token_num

    def _calc_cogvideox_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor_spatial = self.vae_scale_factor_spatial
        latents_height = self.input_config.height // vae_scale_factor_spatial
        latents_width = self.input_config.width // vae_scale_factor_spatial
        latents_frames = (
            self.input_config.num_frames - 1
        ) // self.vae_scale_factor_temporal + 1

        if latents_height % num_sp_patches != 0:
            raise ValueError(
                "The height of the input is not divisible by the number of sequence parallel devices"
            )

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1)
            // (patch_size * num_sp_patches)
        ) * (patch_size * num_sp_patches)
        # get the number of pipeline that matches patch height requirements
        num_pipeline_patch = (
            latents_height + pipeline_patches_height - 1
        ) // pipeline_patches_height
        if num_pipeline_patch != self.num_pipeline_patch:
            logger.warning(
                f"Pipeline patches num changed from "
                f"{self.num_pipeline_patch} to {num_pipeline_patch} due "
                f"to input size and parallelisation requirements"
            )
        pipeline_patches_height_list = [
            pipeline_patches_height for _ in range(num_pipeline_patch - 1)
        ]
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met "
            )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]

        pp_patches_height = [
            sp_patches_height[sp_patch_idx]
            for sp_patches_height in pp_sp_patches_height
        ]
        pp_patches_start_idx_local = [0] + [
            sum(pp_patches_height[:i]) for i in range(1, len(pp_patches_height) + 1)
        ]
        pp_patches_start_end_idx_global = [
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_token_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )
        self.pp_patches_token_num = pp_patches_token_num

    def _calc_consisid_patches_metadata(self):
        num_sp_patches = get_sequence_parallel_world_size()
        sp_patch_idx = get_sequence_parallel_rank()
        patch_size = self.backbone_patch_size
        vae_scale_factor_spatial = self.vae_scale_factor_spatial
        latents_height = self.input_config.height // vae_scale_factor_spatial
        latents_width = self.input_config.width // vae_scale_factor_spatial
        latents_frames = (
            self.input_config.num_frames - 1
        ) // self.vae_scale_factor_temporal + 1

        if latents_height % num_sp_patches != 0:
            raise ValueError(
                "The height of the input is not divisible by the number of sequence parallel devices"
            )

        self.num_pipeline_patch = self.parallel_config.pp_config.num_pipeline_patch
        # Pipeline patches
        pipeline_patches_height = (
            latents_height + self.num_pipeline_patch - 1
        ) // self.num_pipeline_patch
        # make sure pipeline_patches_height is a multiple of (num_sp_patches * patch_size)
        pipeline_patches_height = (
            (pipeline_patches_height + (num_sp_patches * patch_size) - 1)
            // (patch_size * num_sp_patches)
        ) * (patch_size * num_sp_patches)
        # get the number of pipeline that matches patch height requirements
        num_pipeline_patch = (
            latents_height + pipeline_patches_height - 1
        ) // pipeline_patches_height
        if num_pipeline_patch != self.num_pipeline_patch:
            logger.warning(
                f"Pipeline patches num changed from "
                f"{self.num_pipeline_patch} to {num_pipeline_patch} due "
                f"to input size and parallelisation requirements"
            )
        pipeline_patches_height_list = [
            pipeline_patches_height for _ in range(num_pipeline_patch - 1)
        ]
        the_last_pp_patch_height = latents_height - pipeline_patches_height * (
            num_pipeline_patch - 1
        )
        if the_last_pp_patch_height % (patch_size * num_sp_patches) != 0:
            raise ValueError(
                f"The height of the last pipeline patch is {the_last_pp_patch_height}, "
                f"which is not a multiple of (patch_size * num_sp_patches): "
                f"{patch_size} * {num_sp_patches}. Please try to adjust 'num_pipeline_patches "
                f"or sp_degree argument so that the condition are met "
            )
        pipeline_patches_height_list.append(the_last_pp_patch_height)

        # Sequence parallel patches
        # len: sp_degree * num_pipeline_patches
        flatten_patches_height = [
            pp_patch_height // num_sp_patches
            for _ in range(num_sp_patches)
            for pp_patch_height in pipeline_patches_height_list
        ]
        flatten_patches_start_idx = [0] + [
            sum(flatten_patches_height[:i])
            for i in range(1, len(flatten_patches_height) + 1)
        ]
        pp_sp_patches_height = [
            flatten_patches_height[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]
        pp_sp_patches_start_idx = [
            flatten_patches_start_idx[
                pp_patch_idx * num_sp_patches : (pp_patch_idx + 1) * num_sp_patches + 1
            ]
            for pp_patch_idx in range(num_pipeline_patch)
        ]

        pp_patches_height = [
            sp_patches_height[sp_patch_idx]
            for sp_patches_height in pp_sp_patches_height
        ]
        pp_patches_start_idx_local = [0] + [
            sum(pp_patches_height[:i]) for i in range(1, len(pp_patches_height) + 1)
        ]
        pp_patches_start_end_idx_global = [
            sp_patches_start_idx[sp_patch_idx : sp_patch_idx + 2]
            for sp_patches_start_idx in pp_sp_patches_start_idx
        ]
        pp_patches_token_start_end_idx_global = [
            [
                (latents_width // patch_size) * (start_idx // patch_size),
                (latents_width // patch_size) * (end_idx // patch_size),
            ]
            for start_idx, end_idx in pp_patches_start_end_idx_global
        ]
        pp_patches_token_num = [
            end - start for start, end in pp_patches_token_start_end_idx_global
        ]
        pp_patches_token_start_idx_local = [
            sum(pp_patches_token_num[:i]) for i in range(len(pp_patches_token_num) + 1)
        ]
        self.num_pipeline_patch = num_pipeline_patch
        self.pp_patches_height = pp_patches_height
        self.pp_patches_start_idx_local = pp_patches_start_idx_local
        self.pp_patches_start_end_idx_global = pp_patches_start_end_idx_global
        self.pp_patches_token_start_idx_local = pp_patches_token_start_idx_local
        self.pp_patches_token_start_end_idx_global = (
            pp_patches_token_start_end_idx_global
        )
        self.pp_patches_token_num = pp_patches_token_num

    def _reset_recv_buffer(self):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype=self.runtime_config.dtype)

    def _reset_recv_skip_buffer(self, num_blocks_per_stage):
        batch_size = self.input_config.batch_size
        batch_size = batch_size * (2 // self.parallel_config.cfg_degree)
        hidden_dim = self.backbone_inner_dim
        num_patches_tokens = [
            end - start for start, end in self.pp_patches_token_start_end_idx_global
        ]
        patches_shape = [
            [num_blocks_per_stage, batch_size, tokens, hidden_dim]
            for tokens in num_patches_tokens
        ]
        feature_map_shape = [
            num_blocks_per_stage,
            batch_size,
            sum(num_patches_tokens),
            hidden_dim,
        ]
        # reset pipeline communicator buffer
        get_pp_group().set_skip_tensor_recv_buffer(
            patches_shape_list=patches_shape,
            feature_map_shape=feature_map_shape,
        )



class ExternalRuntimeState(RuntimeState):
    """
    Runtime state for running xDiT components outside xDiT.
    This can be used to test individual components in tests without
    having to setup a full distributed environment.
    """
    def __init__(self):
        # Creating config with default params
        config, _ = xFuserArgs().create_config()
        super().__init__(config)


    def _check_distributed_env(self, parallel_config):
        pass


# _RUNTIME: Optional[RuntimeState] = None
# TODO: change to RuntimeState after implementing the unet
_RUNTIME: Optional[DiTRuntimeState] = None


def runtime_state_is_initialized():
    return _RUNTIME is not None


def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME


def initialize_runtime_state(pipeline: Optional[DiffusionPipeline] = None, engine_config: Optional[EngineConfig] = None):
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning(
            "Runtime state is already initialized, reinitializing with pipeline..."
        )
    if hasattr(pipeline, "transformer"):
        _RUNTIME = DiTRuntimeState(pipeline=pipeline, config=engine_config)
    elif hasattr(pipeline, "unet"):
        _RUNTIME = UnetRuntimeState(pipeline=pipeline, config=engine_config)
    elif not pipeline:
        _RUNTIME = ExternalRuntimeState()

