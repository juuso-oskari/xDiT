import os
import torch
import torch.nn as nn
import math
try:
    import aiter
    from aiter.ops.shuffle import shuffle_weight
except ImportError:
    pass # Error will be thrown in base_model.py, if mxfp4 gemms are enabled but AITER is not available.
from typing import Optional
from xfuser.core.distributed.runtime_state import get_runtime_state


# ---------------------------------------------------------------------------
# Hadamard (QuaRot/SpinQuant-style) incoherence preprocessing for MXFP4 GEMMs.
#
# For Y = X @ W.T, insert a block-diagonal orthonormal rotation R on the shared
# in_features axis:  Y = (X @ R) @ (W @ R).T, which is *exact* in real arithmetic
# because each block satisfies R @ R.T == I. Rotating spreads per-channel
# outliers across each block, shrinking the per-32 block amax that MXFP4's shared
# E8M0 scale must cover -> markedly lower fp4 quant error at equal bit-width.
#
# The weight rotation is folded in offline (once, at quantize time -> free at
# runtime); only the activation rotation runs online, and it is cheap
# (~block_r/N of the GEMM's flops).
# ---------------------------------------------------------------------------
_MXFP4_HADAMARD_ENABLED = os.environ.get("XFUSER_MXFP4_GEMM_HADAMARD", "0") != "0"
try:
    _MXFP4_HADAMARD_BLOCK_R = int(os.environ.get("XFUSER_MXFP4_GEMM_HADAMARD_BLOCK_R", "32"))
except ValueError:
    _MXFP4_HADAMARD_BLOCK_R = 32

_HADAMARD_CACHE: dict = {}


def _build_hadamard(block_r: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Normalized Hadamard matrix R (block_r x block_r, R @ R.T == I, block_r a
    power of two). Prefers aiter's create_hadamard_matrix; falls back to a local
    Sylvester construction so the rotation also works without that kernel."""
    try:
        from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
            create_hadamard_matrix,
        )
        return (create_hadamard_matrix(block_r, dtype=dtype) / (block_r ** 0.5)).to(dtype)
    except Exception:
        assert block_r > 0 and (block_r & (block_r - 1)) == 0, "block_r must be a power of 2"
        H = torch.ones((1, 1), dtype=torch.float32)
        while H.shape[0] < block_r:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return (H / (block_r ** 0.5)).to(dtype)


def _get_hadamard(block_r: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    key = (str(device), block_r, dtype)
    R = _HADAMARD_CACHE.get(key)
    if R is None:
        R = _build_hadamard(block_r, dtype).to(device)
        _HADAMARD_CACHE[key] = R
    return R


def _hadamard_rotate(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Block-diagonal orthonormal rotation along the last (in_features) axis, in
    contiguous blocks of R.shape[-1]. Applied identically to weights (offline)
    and activations (online) so the GEMM output is unchanged up to quant error."""
    d = x.shape[-1]
    br = R.shape[-1]
    R = R.to(x.dtype)
    if br == d:
        return torch.matmul(x, R)
    return torch.matmul(x.unflatten(-1, (d // br, br)), R).flatten(-2)


@torch.library.custom_op("mylib::mxfp4_gemm", mutates_args=())
def _mxfp4_gemm(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    a_quant, a_scale = quant_func(a, shuffle=True)
    output = aiter.gemm_a4w4(a_quant, w_quant, a_scale, w_scale, bpreshuffle=True, bias=bias)
    return output

@_mxfp4_gemm.register_fake
def _(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Fake implementation for torch.compile shape inference
    """
    M, _ = a.shape
    N, _ = w_quant.shape
    
    # Return fake tensor with correct shape
    return torch.empty(M, N, dtype=a.dtype, device=a.device)

class xFuserMXFP4Linear(nn.Module):
    """
    Custom Linear layer using MXFP4 GEMM operation
    
    Drop-in replacement for nn.Linear.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        # Hadamard incoherence rotation (0 => disabled). Requires block_r | in_features
        # so the block-diagonal rotation tiles the contraction axis exactly.
        self._hadamard_block_r = (
            _MXFP4_HADAMARD_BLOCK_R
            if (_MXFP4_HADAMARD_ENABLED and _MXFP4_HADAMARD_BLOCK_R > 0
                and in_features % _MXFP4_HADAMARD_BLOCK_R == 0)
            else 0
        )
        # Buffer slot for the (device-resident) rotation matrix; filled at quantize
        # time. persistent=False: it is a derived constant, not part of the checkpoint.
        self.register_buffer("_hadamard_R", None, persistent=False)

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self.mm = self._run_mxfp4_gemm
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform (same as nn.Linear)"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def load_and_quantize_weights(
        self, 
        weights: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ) -> None:
        """
        Load pre-trained weights and quantize them.
        
        Args:
            weights: Full-precision weight tensor [out_features, in_features]
            bias: Optional bias tensor [out_features]
        """
        with torch.no_grad():
            # Temporarily restore weight parameter if it was deleted
            if self.weight is None:
                self.weight = nn.Parameter(
                    torch.empty_like(weights, device=weights.device, dtype=weights.dtype)
                )
            
            self.weight.data.copy_(weights.data)
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias.data)
        
        self._quantize_weights()
    
    def _quantize_weights(self) -> None:
        """
        Quantize weights to FP4 and register quantized tensors as buffers.
        
        This ensures proper device movement with .to(), .cuda(), CPU offload,
        and distributed training frameworks (FSDP, DDP).
        """
        if self.weight is None:
            raise RuntimeError(
                "Cannot quantize: weight parameter is None."
                "Call load_and_quantize_weights() or reset_parameters() first."
            )
        
        # Fold the Hadamard rotation into the weights offline (free at runtime).
        # The activation must be rotated by the same R online (see forward()).
        weight = self.weight
        if self._hadamard_block_r:
            R = _get_hadamard(self._hadamard_block_r, weight.device, weight.dtype)
            self._hadamard_R = R
            weight = _hadamard_rotate(weight, R)

        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_quant, weight_scale = quant_func(weight, shuffle=True)
        weight_shuffle = shuffle_weight(weight_quant, layout=(16, 16))
        
        # Register quantized tensors as buffers for proper state management
        # persistent=True ensures they're saved in state_dict
        self.register_buffer('weight_shuffle', weight_shuffle, persistent=True)
        self.register_buffer('weight_scale', weight_scale, persistent=True)
        
        # Properly remove the original weight parameter to save memory
        # This maintains module structure while freeing memory
        delattr(self, 'weight')
        self.register_parameter('weight', None)

    def _run_mxfp4_gemm(self, a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.ops.mylib.mxfp4_gemm(a, w_quant, w_scale, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using MXFP4 GEMM
        """

        if not hasattr(self, "weight_shuffle"):
            self._quantize_weights()

        # Save original shape
        original_shape = input.shape
        
        # Flatten all batch dimensions: [..., in_features] -> [M, in_features]
        input_2d = input.view(-1, self.in_features)

        # Online activation rotation, matching the offline weight rotation. Cheap
        # (~block_r/out_features of the GEMM flops); no-op when Hadamard disabled.
        if self._hadamard_R is not None:
            input_2d = _hadamard_rotate(input_2d, self._hadamard_R)

        output = self.mm(
            input_2d,
            self.weight_shuffle,
            self.weight_scale,
            None
        )
        if self.bias is not None:
            output = output + self.bias
        
        # Reshape back to original batch dimensions
        # [M, N] -> [..., out_features]
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output
    
    def extra_repr(self):
        """String representation (for print(model))"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class xFuserHybridMXFP4Linear(nn.Module):
    """
    Hybrid linear layer that switches per diffusion step between
    high precision (FP8-quantized nn.Linear path) and low precision (MXFP4 GEMM path).
    """

    def __init__(
        self,
        high_precision_linear: nn.Module,
        low_precision_linear: xFuserMXFP4Linear,
    ) -> None:
        super().__init__()
        self.high_precision_linear = high_precision_linear
        self.low_precision_linear = low_precision_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        runtime_state = get_runtime_state()
        use_high_precision = getattr(runtime_state, "use_high_precision_gemm", True)
        if use_high_precision:
            return self.high_precision_linear(input)
        return self.low_precision_linear(input)

    def extra_repr(self):
        return "hybrid_gemm_schedule=True"
