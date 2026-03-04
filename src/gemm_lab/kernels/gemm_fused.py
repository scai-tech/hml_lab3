from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _check_inputs(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor | None) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("Fused Triton kernel requires Triton to be installed.")
    if x.dim() != 2 or weight_t.dim() != 2:
        raise ValueError("Expected rank-2 tensors for x and weight_t")
    if x.shape[1] != weight_t.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(x.shape)} x {tuple(weight_t.shape)}")
    if not x.is_cuda or not weight_t.is_cuda:
        raise ValueError("Expected CUDA tensors for fused Triton kernel")
    if not x.is_contiguous() or not weight_t.is_contiguous():
        raise ValueError("Expected contiguous x and weight_t")
    if x.dtype != weight_t.dtype:
        raise ValueError("x and weight_t dtype must match")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Supported dtypes are fp16/bf16/fp32")
    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != weight_t.shape[1]:
            raise ValueError("Bias must be shape [out_features]")
        if not bias.is_cuda or bias.dtype != x.dtype or not bias.is_contiguous():
            raise ValueError("Bias must be contiguous CUDA tensor with same dtype")
    return x.shape[0], weight_t.shape[1], x.shape[1]


if triton is not None:

    @triton.jit
    def _fused_linear_bias_relu_kernel(
        a_ptr,
        b_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias,
        HAS_BIAS: tl.constexpr,
        DO_RELU: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
            b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

            a_mask = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
            b_mask = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)

            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a_tile, b_tile)

        if HAS_BIAS:
            b_ptrs = bias_ptr + offs_n * stride_bias
            b = tl.load(b_ptrs, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += b[None, :]

        if DO_RELU:
            acc = tl.maximum(acc, 0.0)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_relu(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    relu: bool = True,
    debug: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    M, N, K = _check_inputs(x, weight_t, bias)

    if debug:
        print("[gemm_fused] launching Triton fused linear+bias+relu kernel")

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_linear_bias_relu_kernel[grid](
        x,
        weight_t,
        bias if bias is not None else y,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        y.stride(0),
        y.stride(1),
        0 if bias is None else bias.stride(0),
        HAS_BIAS=bias is not None,
        DO_RELU=relu,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


class FusedLinearReLU(nn.Module):
    """Linear layer with fused optional ReLU in one call path."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        relu: bool = True,
        device: str = "cuda",
        debug: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        self.relu = relu
        self.debug = debug
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1.0 / (self.weight.shape[1] ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_relu(
            x,
            self.weight.t().contiguous(),
            self.bias,
            relu=self.relu,
            debug=self.debug,
        )
