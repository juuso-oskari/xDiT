# This file implements USP with torch version >= '2.5.0'
import os
import torch
import torch.distributed as dist
import functools

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
    get_runtime_state,
)
from xfuser.core.distributed.runtime_state import _FP8_COMMS_SAFETY_FACTOR

from packaging.version import parse
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.attention_backend import (
    ATTENTION_FUNCTION_REGISTRY,
    AttentionBackendType,
)
from xfuser.core.sparge_attention.head_balance import (
    apply_head_balance,
    revert_head_balance,
)

# Sparge backends whose kernel cost can be load-balanced across Ulysses ranks.
# These all build a block mask via _build_sparge_block_mask and write the
# per-head cost into the head-balance "cost sink". Non-sparge backends are
# excluded so head balancing is a clean no-op for them.
_HEAD_BALANCE_BACKENDS = frozenset({
    AttentionBackendType.AITER_SPARGE,
    AttentionBackendType.AITER_SPARGE_ASM,
    AttentionBackendType.AITER_SPARGE_ASM_V2,
    AttentionBackendType.AITER_SPARGE_ASM_V2_AFFINE_SORTED,
    AttentionBackendType.AITER_SPARGE_ASM_FP8,
    AttentionBackendType.AITER_SPARGE_ASM_FP8_AFFINE_SORTED,
    AttentionBackendType.AITER_SPARGE_V2,
    AttentionBackendType.FLEX_BLOCK_SPARGE,
})
_ATTENTION_BACKENDS_SUPPORTING_PRE_HADAMARD_ROTATION = frozenset({
    AttentionBackendType.AITER_FP8,
    AttentionBackendType.AITER_SPARGE_ASM_FP8,
    AttentionBackendType.AITER_SPARGE_ASM_FP8_AFFINE_SORTED,
})
# Attention backends whose Ulysses all-to-all can ship MXFP4 (fp4) payloads.
# Both route through _aiter_mxfp4_attn_call's mxfp4_pre_quantized path; AITER_F4F4
# additionally repacks V to per-channel fp4 in the kernel call.
_MXFP4_COMMS_BACKENDS = frozenset({"AITER_MXFP4", "AITER_F4F4"})
_FP8_LOG_SCALES = bool(os.environ.get("XFUSER_FP8_LOG_SCALES"))
_FP8_NCCL_NEEDS_VIEW = parse(torch.__version__).release < parse("2.11.0").release
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def ring_attn(attention_function, query, key, value, dropout_p=0.0, is_causal=False, joint_attn_kwargs=None, attention_kwargs=None):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "joint_attn_kwargs": joint_attn_kwargs,
        "attention_kwargs": attention_kwargs,
    }
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.flatten()
    # NCCL does not support FP8 collectives before PyTorch 2.11, view as uint8 (same width) for the transfer.
    if _FP8_NCCL_NEEDS_VIEW and x_dtype in _FP8_DTYPES:
        x = x.view(torch.uint8)
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.view(x_dtype).reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _per_tensor_quant(x: torch.Tensor, scale_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to FP8 using a fixed pre-allocated scale tensor. Returns (x_fp8, descale)."""
    import aiter
    fp8_dtype = aiter.dtypes.fp8
    return aiter.per_tensor_quant(x, scale=scale_t, quant_dtype=fp8_dtype, dtypeMax=torch.finfo(fp8_dtype).max)


def _fp8_comms_input_all_to_all(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> tuple:
    """Quantize Q/K/V to FP8 using per-layer scales and run interleaved input all-to-alls.

    Returns (query, key, value, attn_kwargs_update, (q_scale, k_scale, v_scale), qkv_amaxes).
    """
    q_fp8, q_descale = _per_tensor_quant(query, q_scale)
    query = _ft_c_input_all_to_all(q_fp8)
    k_fp8, k_descale = _per_tensor_quant(key, k_scale)
    key = _ft_c_input_all_to_all(k_fp8)
    v_fp8, v_descale = _per_tensor_quant(value, v_scale)
    value = _ft_c_input_all_to_all(v_fp8)

    qkv_amaxes = (
        (q_descale.item(), k_descale.item(), v_descale.item())
        if _FP8_LOG_SCALES else None
    )

    attn_kwargs_update = {
        "pre_quantized": True,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
    }
    return query, key, value, attn_kwargs_update, (q_scale, k_scale, v_scale), qkv_amaxes


def _mxfp4_comms_input_all_to_all(query, key, value, is_f4f4=False):
    """MXFP4 (fp4) Ulysses input all-to-all for the AITER_MXFP4 attention path.

    Quantizes Q/K to mxfp4 *before* the all-to-all and ships the packed fp4
    tensors (head_dim/2 bytes) + their E8M0 block scales (head_dim/32) instead of
    bf16 (head_dim*2 bytes) -> ~4x smaller Q/K transfer. The kernel then consumes
    the pre-quantized Q/K directly (``mxfp4_pre_quantized``), so no
    dequantize/re-quantize happens on-device after the comms.

    Q/K use ``smooth_rotate_downcast_qk`` (Hadamard rotate + mxfp4 downcast, no
    K-smoothing). The mxfp4 block scale is per-32-element block *along head_dim*,
    a purely local reduction, so the result is bit-identical whether quantized
    before or after the head/sequence all-to-all (which only permutes b/h/s).

    V is left in bf16 over the wire and quantized after the gather: its
    per-channel fp8 scale is an amax *over the sequence*, which is only correct
    once the full (post-a2a) sequence is present on the rank.

    Returns ``(query_fp4, key_fp4, value_bf16, attn_kwargs_update)``.
    """
    from xfuser.core.distributed.attention_backend import (
        HADAMARD_MATRIX,
        AITER_SAGE_V2_BLOCK_R,
        _AITER_SPARGE_ASM_BLOCK_M,
        _pack_v_fp4_colmajor,
        _pack_v_fp8_perchannel,
    )
    from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
        smooth_rotate_downcast_qk,
    )

    head_dim = query.shape[-1]
    # Fold the softmax scale (head_dim**-0.5) and log2(e) into Q exactly like the
    # baseline mxfp4 kernel path (_aiter_mxfp4_attn_call -> sage_quant_mxfp4).
    sm_scale = (head_dim ** -0.5) * 1.4426950408889634
    R = HADAMARD_MATRIX[query.device]

    import aiter, os

    # V: naive direct fp8 cast (descale = 1). fp8 is floating-point, so its
    # exponent already spans V's range (validated: identical cosine to a
    # per-channel amax) -> purely local, no all-reduce, half the bf16 bytes.

    quant_kwargs = dict(
        BLOCK_SIZE_M=_AITER_SPARGE_ASM_BLOCK_M,
        hadamard_rotation=True,
        R=R,
        BLOCK_R=AITER_SAGE_V2_BLOCK_R,
        q_smoothing=False,
        layout="bhsd",
        sm_scale=sm_scale,
    )

    # V a2a. MXFP4_COMMS_V_BF16 selects the V wire format for BOTH mxfp4 and f4f4:
    #   set  -> ship bf16, then quantize post-gather (per-channel fp8 for mxfp4,
    #           per-channel fp4 for f4f4) -- most accurate.
    #   unset-> ship naive fp8 (half the bytes): mxfp4 consumes fp8 directly with a
    #           descale=1 dummy scale; f4f4 upcasts fp8->bf16 before its fp4 repack
    #           (a lossy fp8 roundtrip, but the smaller V transfer can win).
    # Either way V is issued FIRST so its a2a overlaps the Q/K quant + the exposed
    # Q/K collective below.
    v_bf16_comms = os.environ.get("MXFP4_COMMS_V_BF16", "0") != "0"
    ship_v_bf16 = v_bf16_comms
    value = _ft_c_input_all_to_all(value if ship_v_bf16 else value.to(aiter.dtypes.fp8))

    # q, k quant
    q_fp4, q_scale, k_fp4, k_scale, _ = smooth_rotate_downcast_qk(query, key, **quant_kwargs)

    # MXFP4_COMMS_PACK=1 -> pack fp4+scale into one collective per Q/K.
    # else -> unpacked (each fp4/scale its own collective).
    if os.environ.get("MXFP4_COMMS_PACK", "0") != "0":
        query = _ft_c_input_all_to_all(torch.cat([q_fp4, q_scale], dim=-1))
        key = _ft_c_input_all_to_all(torch.cat([k_fp4, k_scale], dim=-1))  # packed K last
        attn_kwargs_update = {"mxfp4_pre_quantized": True, "mxfp4_fp4_width": q_fp4.shape[-1]}
    else:
        query = _ft_c_input_all_to_all(q_fp4)
        q_scale = _ft_c_input_all_to_all(q_scale)
        key = _ft_c_input_all_to_all(k_fp4)
        k_scale = _ft_c_input_all_to_all(k_scale)  # exposed tail
        attn_kwargs_update = {"mxfp4_pre_quantized": True, "q_scale": q_scale, "k_scale": k_scale}

    # --- V quant at the tail. The quant kernels launch on the compute stream while
    # the (still-unwaited) Q/K + k_scale collective is in flight on the comm stream,
    # so V quantization overlaps the exposed K a2a. V's per-channel scale is an amax
    # over the *full* (post-a2a) sequence, so it must be computed here post-gather.
    # Both backends branch identically on the V wire format (v_bf16_comms).
    if is_f4f4:
        # f4f4 V pack (post-gather). AITER_F4F4_MXFP4_V selects microscaled mxfp4-V
        # (per-(channel, 32-kv-block) E8M0) vs the default per-channel fp4. This MUST
        # match the deployed fwd_hd128_f4f4.co build or the kernel reads garbage; it
        # mirrors the fresh-path sage_quant_f4f4 toggle. The packer reads strides (no
        # .contiguous()) and returns a 128-padded view.
        if v_bf16_comms:
            v_bshd = torch.permute(value, [0, 2, 1, 3])                     # bf16 [b, s, h, d]
        else:
            # naive fp8 comm -> upcast fp8->bf16 for the fp4 repack.
            v_bshd = torch.permute(value, [0, 2, 1, 3]).to(torch.bfloat16)  # [b, s, h, d]
        if os.environ.get("AITER_F4F4_MXFP4_V", "0") != "0":
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import _pack_v_mxfp4_colmajor
            value, v_descale = _pack_v_mxfp4_colmajor(v_bshd)               # microscaled (E8M0 block) V
        else:
            value, v_descale = _pack_v_fp4_colmajor(v_bshd)                 # per-channel fp4 V
        attn_kwargs_update["mxfp4_v_descale"] = v_descale
    else:
        # mxfp4: per-channel fp8 V.
        if v_bf16_comms:
            # per-channel amax fp8 quant over the full gathered sequence (bhsd in/out).
            value, v_scale = _pack_v_fp8_perchannel(value)                 # bhsd fp8
        else:
            # naive fp8 comm (descale = 1): dummy per-channel ones scale (bhsd: h = dim 1).
            v_scale = torch.ones(
                value.shape[0], value.shape[1], value.shape[3],
                device=value.device, dtype=torch.float32,
            )                                                              # [b, h, d]
        attn_kwargs_update["mxfp4_v_scale"] = v_scale
    return query, key, value, attn_kwargs_update


# DistriFusion-style temporal stale-KV cache for the mxfp4-comms Ulysses path.
# Keyed by (layer, CFG parity); holds the previous same-branch step's *gathered*
# (post-a2a) K/V so a later step can attend against slightly-stale K/V while its
# own fresh K/V is (eventually) gathered in the background. Consecutive denoising
# steps are highly similar (input temporal redundancy), so reuse is ~lossless.
_MXFP4_STALE_KV_CACHE: dict = {}


def _mxfp4_stale_kv_swap(layer, key, value, mxfp4_kwargs):
    """DistriFusion-style fresh-local + stale-remote K/V for the mxfp4-comms path.

    Assembles attention K/V as: this rank's own (fresh) sequence block + the
    previous same-CFG-branch step's (stale) K/V for every other block, then runs
    one *normal* full-attention call (no LSE merge needed). The fresh local block
    anchors the current step, avoiding the compounding drift that fully-stale K/V
    produced (which was pure noise). Probe form: still gathers fresh (blocking)
    and caches it for the next same-parity step.
    """
    import os
    from xfuser.core.distributed import (
        get_ulysses_parallel_world_size,
        get_ulysses_parallel_rank,
    )
    rs = get_runtime_state()
    sc = getattr(rs, "step_counter", None)
    if sc is None:
        return key, value, mxfp4_kwargs
    step = int(sc)
    warmup_fwd = 2 * int(os.environ.get("MXFP4_COMMS_STALE_WARMUP", "4"))
    ck = (id(layer), step % 2)  # parity ~ cond/uncond CFG branch
    kscale = mxfp4_kwargs.get("k_scale")
    prev = _MXFP4_STALE_KV_CACHE.get(ck)
    _MXFP4_STALE_KV_CACHE[ck] = (key, value, kscale)  # fresh full gather -> next step
    if prev is None or step < warmup_fwd:
        return key, value, mxfp4_kwargs
    P = get_ulysses_parallel_world_size()
    r = get_ulysses_parallel_rank()
    S = key.shape[2]  # gathered layout [B, H/P, S_full, D]
    if P <= 1 or S % P != 0:
        return key, value, mxfp4_kwargs
    s = S // P
    lo, hi = r * s, (r + 1) * s  # this rank's own (fresh) sequence block
    sk, sv, sks = prev
    k_used = sk.clone(); k_used[:, :, lo:hi, :] = key[:, :, lo:hi, :]
    v_used = sv.clone(); v_used[:, :, lo:hi, :] = value[:, :, lo:hi, :]
    nk = dict(mxfp4_kwargs)
    if kscale is not None and sks is not None:
        ks_used = sks.clone(); ks_used[:, :, lo:hi, :] = kscale[:, :, lo:hi, :]
        nk["k_scale"] = ks_used
    return k_used, v_used, nk


def _fp8_comms_output_all_to_all(out: torch.Tensor, o_scale_t: torch.Tensor | None) -> torch.Tensor:
    """Quantize attention output to FP8, run output all-to-all, dequantize back."""
    restore_dtype = out.dtype if out.dtype not in _FP8_DTYPES else torch.bfloat16
    if out.dtype not in _FP8_DTYPES:
        out_fp8, out_descale = _per_tensor_quant(out, o_scale_t)
    else:
        out_fp8, out_descale = out, o_scale_t
    return (_ft_c_output_all_to_all(out_fp8).float() * out_descale).to(restore_dtype)


def _mxfp4_comms_output_all_to_all(out: torch.Tensor) -> torch.Tensor:
    """Naive-fp8 output all-to-all for the mxfp4-comms path.

    Casts the attention output to fp8 (descale=1) before the output gather,
    halving the bf16 payload. Safe because out = softmax(QK)*V is a convex
    combination of V, so |out| <= max|V|, and V is already shipped as naive fp8;
    descale=1 is identical on every rank, so the post-a2a tensor (which mixes
    fp8 chunks from all ranks) needs no per-rank rescale.
    """
    import aiter
    restore_dtype = out.dtype if out.dtype not in _FP8_DTYPES else torch.bfloat16
    out_fp8 = out.to(aiter.dtypes.fp8)
    out_fp8 = _ft_c_output_all_to_all(out_fp8)
    return out_fp8.to(restore_dtype)


def _combined_qkv_all_to_all(q, k, v):
    """Concatenate query, key, value tensors and perform a single all-to-all communication."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    # [3, b, h, s, d]
    qkv = torch.stack([q, k, v], dim=0)
    # [3, b, P, h/P, s, d]
    qkv = qkv.view(3, b, world_size, h // world_size, s, d)
    # [P, 3, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [3, b, h/P, P*s, d]  — reshape directly avoids the intermediate
    # contiguous copy that the separate permute+view required.
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(3, b, h // world_size, -1, d)

    q, k, v = torch.unbind(qkv, dim=0)
    return q, k, v


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
    return joint_key, joint_value

def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor

def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value

def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend
    func = ATTENTION_FUNCTION_REGISTRY.get(attention_backend, None)
    if func is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(func)

def concat_joint_tensors_decorator(func):
    """
    Decorator to handle joint tensor concatenation
    This is needed for ring attention with 'rear' joint_strategy, as it
    needs to concat the joint tensors before calling the attention function
    but only on the last step.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query, key, value = args[0:3]
        is_causal = kwargs.get("is_causal")
        dropout_p = kwargs.get("dropout_p")
        joint_attn_kwargs = kwargs.get("joint_attn_kwargs", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)

        if joint_attn_kwargs is not None:
            joint_strategy = joint_attn_kwargs.get("joint_strategy", None)
            joint_key = joint_attn_kwargs.get("joint_key", None)
            joint_value = joint_attn_kwargs.get("joint_value", None)
            step = joint_attn_kwargs.get("step", 0)
            total_steps = joint_attn_kwargs.get("total_steps", 1)
            if (joint_strategy == "front" and step == 0) or (joint_strategy == "rear" and step == total_steps - 1):
                key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
                value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)
            joint_attn_kwargs["step"] = step + 1 # In place increment step

        return func(query, key, value, dropout_p=dropout_p, is_causal=is_causal, attention_kwargs=attention_kwargs)
    return wrapper

def USP(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        joint_query: torch.Tensor | None = None,
        joint_key: torch.Tensor | None = None,
        joint_value: torch.Tensor | None = None,
        joint_strategy: str | None = None,
        attn_layer=None,
        combine_qkv_a2a: bool | None = None,
        use_fp8_comms: bool = False,
        backend=None,
        attention_kwargs: dict | None = None,
        head_balance_layer=None,
        fp8_q_scale: torch.Tensor | None = None,
        fp8_k_scale: torch.Tensor | None = None,
        fp8_v_scale: torch.Tensor | None = None,
        fp8_o_scale: torch.Tensor | None = None,
        fp8_comms_synced: bool = False,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.

    ``head_balance_layer`` (optional): a stable per-layer handle (e.g. the
    attention module). When provided and --use_spargeattn_head_balance is set, the
    Ulysses head dimension is permuted so each rank gets a cost-balanced subset
    of heads (block-sparse load balancing); the permutation is inverted on the
    output. No-op for non-sparse backends (no cost is published) and for ring/
    joint paths.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False

    attention_function = _get_attention_function(backend=backend)

    hb_uly = get_ulysses_parallel_world_size()
    hb_backend = backend if backend is not None else get_runtime_state().attention_backend
    query, key, value, hb_applied, attention_kwargs = apply_head_balance(
        query, key, value, head_balance_layer,
        enabled=get_runtime_state().runtime_config.use_spargeattn_head_balance,
        ulysses_world_size=hb_uly,
        ring_world_size=get_ring_parallel_world_size(),
        is_sparge_backend=hb_backend in _HEAD_BALANCE_BACKENDS,
        joint_strategy=joint_strategy,
        attention_kwargs=attention_kwargs,
    )

    joint_attn_kwargs = None
    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)
        joint_attn_kwargs = {
            "joint_value": joint_value,
            "joint_key": joint_key,
            "joint_strategy": joint_strategy,
            "step": 0,
            "total_steps": get_ring_parallel_world_size(),

        }

    qkv_scales = None
    qkv_amaxes = None
    if get_ulysses_parallel_world_size() > 1:
        if use_fp8_comms:
            fp8_comms_backend = backend if backend is not None else get_runtime_state().attention_backend
            if fp8_comms_backend in _ATTENTION_BACKENDS_SUPPORTING_PRE_HADAMARD_ROTATION:
                from xfuser.core.distributed.attention_backend import (
                    FP8_HADAMARD_MATRIX,
                    _fp8_hadamard_rotate,
                )
                R = FP8_HADAMARD_MATRIX[query.device]
                query = _fp8_hadamard_rotate(query, R).contiguous()
                key = _fp8_hadamard_rotate(key, R).contiguous()
            if fp8_q_scale is None or fp8_k_scale is None or fp8_v_scale is None:
                raise RuntimeError(
                    "FP8 comms requires per-layer scale buffers (fp8_q_scale, fp8_k_scale, fp8_v_scale)."
                )
            query, key, value, attn_kwargs_update, qkv_scales, qkv_amaxes = _fp8_comms_input_all_to_all(
                query, key, value, fp8_q_scale, fp8_k_scale, fp8_v_scale,
            )
            attention_kwargs = (attention_kwargs or {}) | attn_kwargs_update
        elif (
            get_runtime_state().runtime_config.use_mxfp4_comms
            and getattr(hb_backend, "name", None) in _MXFP4_COMMS_BACKENDS
        ):
            # Shared mxfp4 Q/K comms packing for both AITER_MXFP4 and AITER_F4F4; V is
            # quantized here too (per its backend's V path) so it overlaps the K a2a.
            is_f4f4 = getattr(hb_backend, "name", None) == "AITER_F4F4"
            query, key, value, mxfp4_kwargs = _mxfp4_comms_input_all_to_all(
                query, key, value, is_f4f4=is_f4f4,
            )
            if os.environ.get("MXFP4_COMMS_STALE_KV", "0") != "0" and head_balance_layer is not None:
                key, value, mxfp4_kwargs = _mxfp4_stale_kv_swap(head_balance_layer, key, value, mxfp4_kwargs)
            attention_kwargs = (attention_kwargs or {}) | mxfp4_kwargs
        elif combine_qkv_a2a and query.shape == key.shape == value.shape:
            query, key, value = _combined_qkv_all_to_all(query, key, value)
        else:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)

    if get_sequence_parallel_world_size() == 1: # No SP
        out, _ = attention_function(query,
                                    key,
                                    value,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    joint_attn_kwargs=joint_attn_kwargs,
                                    attention_kwargs=attention_kwargs)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(attention_function,
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        joint_attn_kwargs=joint_attn_kwargs,
                        attention_kwargs=attention_kwargs)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out, _ = attention_function(query,
                                        key,
                                        value,
                                        dropout_p=dropout_p,
                                        is_causal=is_causal,
                                        joint_attn_kwargs=joint_attn_kwargs,
                                        attention_kwargs=attention_kwargs)
        else: # USP
            out = ring_attn(attention_function,
                            query,
                            key,
                            value,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                            joint_attn_kwargs=joint_attn_kwargs,
                            attention_kwargs=attention_kwargs)
        if use_fp8_comms:
            dtype_max = 448.0
            out_amax = out.abs().amax()
            if _FP8_LOG_SCALES and qkv_amaxes is not None:
                out_amax = out_amax.item()
                rank = dist.get_rank()
                q_amax, k_amax, v_amax = qkv_amaxes
                print(f"[fp8_scales rank{rank}] q_amax={q_amax:.4f} k_amax={k_amax:.4f} v_amax={v_amax:.4f} out_amax={out_amax:.4f}")
            if fp8_o_scale is None:
                raise RuntimeError(
                    "FP8 comms requires per-layer scale buffers fp8_o_scale"
                )
            out = _fp8_comms_output_all_to_all(out, fp8_o_scale)
        elif (
            get_runtime_state().runtime_config.use_mxfp4_comms
            and getattr(hb_backend, "name", None) in _MXFP4_COMMS_BACKENDS
            and os.environ.get("MXFP4_COMMS_FP8_OUT", "1") != "0"
        ):
            out = _mxfp4_comms_output_all_to_all(out)
        else:
            out = _ft_c_output_all_to_all(out)
        if hb_applied:
            # Restore global head order on the output, gather this step's per-head
            # costs across the Ulysses group, and plan next step's permutation.
            out = revert_head_balance(
                out, attention_kwargs, head_balance_layer, hb_uly
            )

    return out


def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        use_fp8_comms: bool = False,  # accepted for call-site uniformity with USP(), never applied
        backend=None,
        attention_kwargs=None,
        head_balance_layer=None,
        fp8_q_scale: torch.Tensor | None = None,
        fp8_k_scale: torch.Tensor | None = None,
        fp8_v_scale: torch.Tensor | None = None,
    ):
    """
    Runs attention call without any parallelism.
    This can be used when the logic necessitates no Ulysses or Ring parallelism in any case.
    Explicit backend can be provided to specify the attention backend to use.

    ``head_balance_layer`` is accepted for call-site signature parity with
    ``USP`` but ignored here: with no Ulysses parallelism there is no head
    sharding to balance.
    """
    attention_function = _get_attention_function(backend=backend)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
    )
    return out

