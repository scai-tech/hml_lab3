from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Expected rank-2 tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Triton GEMM expects CUDA tensors.")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Starter kernel expects contiguous inputs.")
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match.")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Supported dtypes: fp16, bf16, fp32")
    return a.shape[0], b.shape[1], a.shape[1]


if triton is not None:

    @triton.jit
    def _gemm_kernel_naive(
        a_ptr,
        b_ptr,
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
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        lane = tl.arange(0, BLOCK_SIZE * BLOCK_SIZE)
        offs_m = pid_m * BLOCK_SIZE + (lane % BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + (lane // BLOCK_SIZE)

        acc = tl.zeros((BLOCK_SIZE * BLOCK_SIZE,), dtype=tl.float32)
        for k in range(0, K):
            a = tl.load(a_ptr + offs_m * stride_am + k * stride_ak, mask=offs_m < M, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + k * stride_bk + offs_n * stride_bn, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += a * b

        tl.store(
            c_ptr + offs_m * stride_cm + offs_n * stride_cn,
            acc,
            mask=(offs_m < M) & (offs_n < N),
        )


def triton_gemm_naive(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_size: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    M, N, K = _check_inputs(a, b)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    _gemm_kernel_naive[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
