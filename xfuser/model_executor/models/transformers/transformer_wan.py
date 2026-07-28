import torch
import math
import os
from typing import Optional, Union, Dict, Any, Tuple

from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttention
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.usp import (
    USP,
    attention,
)
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_runtime_state,
    get_ulysses_parallel_world_size,
)
from xfuser.core.distributed.attention_backend import SUPPORTS_PRE_QUANTIZATION_BACKENDS
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister
)
from xfuser.model_executor.layers.mxfp6_linear import xFuserMXFP6Linear
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]

@xFuserAttentionProcessorRegister.register(WanAttnProcessor)
class xFuserWanAttnProcessor(WanAttnProcessor):

    def __init__(
        self,
        use_ulysses_parallel_attention: bool = True,
        is_cross_attention: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if use_ulysses_parallel_attention:
            self.attention_function = USP
        else:
            self.attention_function = attention
        self.is_cross_attention = is_cross_attention
        # attention_kwargs is the shared, mutable dict used by sparse backends
        # (SSTA / sparge) to receive layout info like `thw`. Cross-attention and
        # the I2V image-context sub-call below are dense, so they don't read it.
        self.attention_kwargs = attention_kwargs

    def _maybe_mxfp4_out_layer(self, attn: "WanAttention", has_img: bool):
        """Return the o_proj (xFuserMXFP4Linear) when the fused mxfp4 output-comms path
        is eligible, else None (fall back to bf16/fp8 output all-to-all).

        Enabled only when: MXFP4_COMMS_MXFP4_OUT is set, this is the sequence-parallel
        self-attention, mxfp4 comms is on (Ulysses>1), and o_proj is an mxfp4 GEMM with
        Hadamard rotation whose block divides head_dim (so per-head rotation before the
        head-gather a2a is identical to rotating the flattened model_dim). Disabled for
        I2V (``has_img``: the bf16 image-context add must precede o_proj)."""
        if os.environ.get("MXFP4_COMMS_MXFP4_OUT", "0") == "0":
            return None
        if self.is_cross_attention or has_img:
            return None
        rs = get_runtime_state()
        if not getattr(rs.runtime_config, "use_mxfp4_comms", False):
            return None
        if get_ulysses_parallel_world_size() <= 1:
            return None

        from xfuser.model_executor.layers.mxfp4_linear import (
            xFuserMXFP4Linear,
            xFuserHybridMXFP4Linear,
        )
        o_proj = attn.to_out[0]
        if isinstance(o_proj, xFuserHybridMXFP4Linear):
            # Per-step high/low switch: only fuse on low-precision (mxfp4) steps.
            if getattr(rs, "use_high_precision_gemm", True):
                return None
            o_proj = o_proj.low_precision_linear
        if not isinstance(o_proj, xFuserMXFP4Linear):
            return None
        if getattr(o_proj, "_hadamard_R", None) is None:
            return None
        block_r = getattr(o_proj, "_hadamard_block_r", 0)
        head_dim = o_proj.in_features // attn.heads
        if block_r <= 0 or head_dim % block_r != 0:
            return None
        return o_proj

    def _apply_fused_out_comms(self, attn, o_proj, attn_out):
        """Epilogue for the fused mxfp4 output-comms path.

        ``attn_out = (a_fp4 [b, h, S/P, d/2], a_scale [b, h, S/P, d/32])`` (uint8) -- the
        attention output was Hadamard-rotated + mxfp4-packed (fused Triton kernel) before the
        head-gather a2a. Mirror the bf16 path's transpose(1,2)+flatten(2,3) on both, then feed
        the pre-quantized activation straight into o_proj (skips the rotate+quant it would
        otherwise redo). Stays inside the compiled graph (the fused kernel + e8m0 shuffle +
        prequant GEMM are all compile-safe), so the output a2a overlaps surrounding compute."""
        a_fp4, a_scale = attn_out
        a_fp4 = a_fp4.transpose(1, 2).flatten(2, 3)       # [b, S/P, model_dim/2]
        a_scale = a_scale.transpose(1, 2).flatten(2, 3)   # [b, S/P, model_dim/32]
        batch_shape = a_fp4.shape[:-1]
        hidden_states = o_proj.forward_prequantized(
            a_fp4.reshape(-1, a_fp4.shape[-1]),
            a_scale.reshape(-1, a_scale.shape[-1]),
        ).view(*batch_shape, -1)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _run_shared_fp6_projections(
        input_tensor: torch.Tensor,
        projections: tuple[xFuserMXFP6Linear, ...],
    ) -> Optional[tuple[torch.Tensor, ...]]:
        if not projections or not all(isinstance(proj, xFuserMXFP6Linear) for proj in projections):
            return None
        in_features = projections[0].in_features
        if input_tensor.shape[-1] != in_features or any(proj.in_features != in_features for proj in projections):
            return None

        original_shape = input_tensor.shape
        input_2d = input_tensor.view(-1, in_features)
        a_packed, a_scale = projections[0].pack_activation(input_2d)
        M = input_2d.shape[0]
        return tuple(
            proj.forward_packed_2d(a_packed, a_scale, M, input_tensor.dtype).view(
                *original_shape[:-1], proj.out_features
            )
            for proj in projections
        )

    def _get_qkv_projections(self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # encoder_hidden_states is only passed for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                # In self-attention layers, we can fuse the entire QKV projection into a single linear
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                # In cross-attention layers, we can only fuse the KV projections into a single linear
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            if encoder_hidden_states is hidden_states:
                shared_qkv = self._run_shared_fp6_projections(
                    hidden_states, (attn.to_q, attn.to_k, attn.to_v)
                )
                if shared_qkv is not None:
                    return shared_qkv
            query = attn.to_q(hidden_states)
            shared_kv = self._run_shared_fp6_projections(
                encoder_hidden_states, (attn.to_k, attn.to_v)
            )
            if shared_kv is not None:
                key, value = shared_kv
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def _get_added_kv_projections(self, attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
        if attn.fused_projections:
            key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
        else:
            shared_kv = self._run_shared_fp6_projections(
                encoder_hidden_states_img, (attn.add_k_proj, attn.add_v_proj)
            )
            if shared_kv is not None:
                key_img, value_img = shared_kv
            else:
                key_img = attn.add_k_proj(encoder_hidden_states_img)
                value_img = attn.add_v_proj(encoder_hidden_states_img)
        return key_img, value_img


    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        backend = None
        if self.is_cross_attention:
            # We allow specifying a different backend for cross-attention than the main attention backend
            # as some backends may have too much overhead for cross-attention.
            backend = get_runtime_state().get_cross_attention_backend()

        activation_dtype = hidden_states.dtype

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        query, key, value = self._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        runtime_state = get_runtime_state()
        use_fp8_comms = (
            not self.is_cross_attention
            and runtime_state.fp8_comms is not None
            and runtime_state.attention_backend in SUPPORTS_PRE_QUANTIZATION_BACKENDS
        )

        # update running max for dynamic scale calibration -- pure in-place tensor ops, no graph break
        fp8_comms_synced = False
        if not self.is_cross_attention and runtime_state.fp8_comms is not None:
            fp8_owner = getattr(attn, "fp8_comms_owner", None)
            if fp8_owner is not None and hasattr(attn, "fp8_comms_layer_idx"):
                runtime_state.fp8_comms.update_running_max(
                    fp8_owner,
                    attn.fp8_comms_layer_idx,
                    query,
                    key,
                    value,
                )
                fp8_comms_synced = runtime_state.fp8_comms.get_model_state(fp8_owner).synced
        use_fp8_comms = use_fp8_comms and fp8_comms_synced

        fp8_q_scale = getattr(attn, "fp8_q_scale", None)
        fp8_k_scale = getattr(attn, "fp8_k_scale", None)
        fp8_v_scale = getattr(attn, "fp8_v_scale", None)
        fp8_o_scale = getattr(attn, "fp8_o_scale", None)
        fp8_kwargs = (
            {
                "fp8_q_scale": fp8_q_scale,
                "fp8_k_scale": fp8_k_scale,
                "fp8_v_scale": fp8_v_scale,
                "fp8_o_scale": fp8_o_scale,
            }
            if use_fp8_comms
            else {}
        )

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = self.attention_function(
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                backend=backend,
                attention_kwargs=self.attention_kwargs,
                use_fp8_comms=use_fp8_comms,
                **fp8_kwargs,
            ).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.to(activation_dtype)

        mxfp4_out_layer = self._maybe_mxfp4_out_layer(attn, hidden_states_img is not None)

        attn_out = self.attention_function(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            backend=backend,
            use_fp8_comms=use_fp8_comms,
            attention_kwargs=self.attention_kwargs,
            head_balance_layer=attn,
            mxfp4_out_layer=mxfp4_out_layer,
            **fp8_kwargs,
        )

        if isinstance(attn_out, tuple):
            # Fused mxfp4 output comms (see _apply_fused_out_comms). Runs eagerly: the
            # epilogue is post-collective (USP already graph-broke) and uses exotic
            # fp4/e8m0 dtype reinterprets that inductor can't compile.
            return self._apply_fused_out_comms(attn, mxfp4_out_layer, attn_out)

        hidden_states = attn_out.transpose(1, 2)

        # update running max for dynamic scale calibration -- pure in-place tensor ops, no graph break
        if not self.is_cross_attention and runtime_state.fp8_comms is not None:
            fp8_owner = getattr(attn, "fp8_comms_owner", None)
            if fp8_owner is not None and hasattr(attn, "fp8_comms_layer_idx"):
                runtime_state.fp8_comms.update_o_running_max(
                    fp8_owner,
                    attn.fp8_comms_layer_idx,
                    hidden_states,
                )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(activation_dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserWanTransformer3DWrapper(WanTransformer3DModel):


    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
           patch_size,
           num_attention_heads,
           attention_head_dim,
           in_channels,
           out_channels,
           text_dim,
           freq_dim,
           ffn_dim,
           num_layers,
           cross_attn_norm,
           qk_norm,
           eps,
           image_dim,
           added_kv_proj_dim,
           rope_max_seq_len,
           pos_embed_seq_len,
        )
        self.attention_kwargs = attention_kwargs
        for layer_idx, block in enumerate(self.blocks):
            block.attn1.processor = xFuserWanAttnProcessor(attention_kwargs=self.attention_kwargs)
            block.attn2.processor = xFuserWanAttnProcessor(use_ulysses_parallel_attention=False, is_cross_attention=True)
            # Per-layer head permutation buffer for the Ulysses block-sparse head
            # balancer (read/updated in-place inside USP; identity = no balancing).
            # Registered pre-compile and non-persistent so it stays out of the
            # state_dict and is captured as a graph input by torch.compile.
            block.attn1.register_buffer(
                "head_perm",
                torch.arange(num_attention_heads, dtype=torch.long),
                persistent=False,
            )
            # Per-layer FP8 comms scales (updated during calibration, frozen before compile).
            block.attn1.register_buffer(
                "fp8_q_scale", torch.ones(1, dtype=torch.float32), persistent=False
            )
            block.attn1.register_buffer(
                "fp8_k_scale", torch.ones(1, dtype=torch.float32), persistent=False
            )
            block.attn1.register_buffer(
                "fp8_v_scale", torch.ones(1, dtype=torch.float32), persistent=False
            )
            block.attn1.register_buffer(
                "fp8_o_scale", torch.zeros(1, dtype=torch.float32), persistent=False
            )
            block.attn1.register_buffer(
                "fp8_comms_layer_idx",
                torch.tensor([layer_idx], dtype=torch.long),
                persistent=False,
            )
            # Plain attribute (not nn.Module child) — assigning self directly would
            # register the transformer as a submodule of attn1 and recurse forever.
            object.__setattr__(block.attn1, "fp8_comms_owner", self)

    def register_fp8_comms_state(self, fp8_comms) -> None:
        """Register this transformer with runtime FP8 comms state (after runtime init)."""
        if fp8_comms is None:
            return
        fp8_comms.register_model(self, len(self.blocks))


    def _chunk_and_pad_sequence(self, x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat([x,
                        torch.zeros(
                            pad_shape,
                            dtype=x.dtype,
                            device=x.device,
                        )], dim=dim)
        x = torch.chunk(x,
                        sp_world_size,
                        dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0


        get_runtime_state().increment_step_counter()

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        if self.attention_kwargs is not None:
            self.attention_kwargs["thw"] = (
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
            )

        # 1. RoPE
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # We only reach this for Wan2.1, when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Part of sequence parallel: given the resolution, we may need to pad the sequence length to match this prior to chunking
        pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        hidden_states = self._chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)

        if ts_seq_len is not None: # (wan2.2 ti2v)
            temb = self._chunk_and_pad_sequence(temb, sp_world_rank, sp_world_size, pad_amount, dim=1)
            timestep_proj = self._chunk_and_pad_sequence(timestep_proj, sp_world_rank, sp_world_size, pad_amount, dim=1)

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, pad_amount):
            freqs = self._chunk_and_pad_sequence(freqs, sp_world_rank, sp_world_size, pad_amount, dim=1)
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)


        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = self._gather_and_unpad(hidden_states, pad_amount, dim=-2)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)