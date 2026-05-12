"""Triton kernels for grouped MoE matmuls and fused SwiGLU projection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_TORCH_TO_TL_DTYPE: dict[torch.dtype, "tl.dtype"] = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _grouped_matmul_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    offsets_ptr,
    lengths_ptr,
    E,
    K,
    N,
    stride_am,
    stride_ak,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Triton kernel for grouped expert matmul tiles."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    if pid_e >= E:
        return
    row_start = tl.load(offsets_ptr + pid_e)
    rows = tl.load(lengths_ptr + pid_e)
    block_row = pid_m * BLOCK_M
    if block_row >= rows:
        return

    offs_m = block_row + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_base = 0
    while k_base < K:
        k_offsets = k_base + offs_k
        a_ptrs = (
            a_ptr
            + (row_start + offs_m[:, None]) * stride_am
            + k_offsets[None, :] * stride_ak
        )
        w_ptrs = (
            w_ptr
            + pid_e * stride_w_e
            + k_offsets[:, None] * stride_w_k
            + offs_n[None, :] * stride_w_n
        )
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < rows) & (k_offsets[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, w)
        k_base += BLOCK_K

    c_ptrs = (
        c_ptr + (row_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N))


def grouped_matmul(
    a_packed: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply expert-specific linear projections to packed token rows on CUDA."""
    if a_packed.dim() != 2 or weights.dim() != 3:
        raise ValueError("grouped_matmul expects a_packed [M,K] and weights [E,K,N]")
    if not a_packed.is_cuda or not weights.is_cuda:
        raise ValueError("grouped_matmul requires CUDA tensors")
    if offsets.dim() != 1 or lengths.dim() != 1:
        raise ValueError("offsets/lengths must be 1D")
    E, K, N = weights.shape
    M = a_packed.shape[0]
    if a_packed.shape[1] != K:
        raise ValueError(
            f"grouped_matmul shape mismatch: a{tuple(a_packed.shape)} w{tuple(weights.shape)}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    c = torch.empty((M, N), device=a_packed.device, dtype=out_dtype)
    max_rows = int(lengths.max().item()) if lengths.numel() else 0
    if N >= 4096:
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        num_warps = 8
        num_stages = 4
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        num_warps = 4
        num_stages = 4
    grid = (triton.cdiv(max_rows, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
    _grouped_matmul_kernel[grid](
        a_packed,
        weights,
        c,
        offsets,
        lengths,
        E,
        K,
        N,
        a_packed.stride(0),
        a_packed.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.jit
def _grouped_swiglu_w2_kernel(
    h_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    offsets_ptr,
    lengths_ptr,
    E,
    K,
    N,
    stride_hm,
    stride_hk,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    stride_b_e,
    stride_b_n,
    stride_cm,
    stride_cn,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    W_DTYPE: tl.constexpr,
):
    """Triton kernel for grouped SwiGLU-output projection tiles."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_e = tl.program_id(2)

    if pid_e >= E:
        return
    row_start = tl.load(offsets_ptr + pid_e)
    rows = tl.load(lengths_ptr + pid_e)
    block_row = pid_m * BLOCK_M
    if block_row >= rows:
        return

    offs_m = block_row + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_base = 0
    while k_base < K:
        k_offsets = k_base + offs_k

        h_glu_ptrs = (
            h_ptr
            + (row_start + offs_m[:, None]) * stride_hm
            + k_offsets[None, :] * stride_hk
        )
        h_lin_ptrs = (
            h_ptr
            + (row_start + offs_m[:, None]) * stride_hm
            + (k_offsets[None, :] + K) * stride_hk
        )

        h_glu = tl.load(
            h_glu_ptrs,
            mask=(offs_m[:, None] < rows) & (k_offsets[None, :] < K),
            other=0.0,
        )
        h_lin = tl.load(
            h_lin_ptrs,
            mask=(offs_m[:, None] < rows) & (k_offsets[None, :] < K),
            other=0.0,
        )

        h_glu = tl.where(h_glu > LIMIT, LIMIT, h_glu)
        h_lin = tl.where(h_lin > LIMIT, LIMIT, h_lin)
        h_lin = tl.where(h_lin < -LIMIT, -LIMIT, h_lin)

        h_glu_f = h_glu.to(tl.float32)
        h_lin_f = h_lin.to(tl.float32)
        act = h_glu_f * tl.sigmoid(ALPHA * h_glu_f) * (h_lin_f + 1.0)
        # tl.dot requires matching input dtypes. The SwiGLU math is computed in
        # fp32 for stability, then cast back to the projection weight dtype.
        act = act.to(W_DTYPE)

        w_ptrs = (
            w_ptr
            + pid_e * stride_w_e
            + k_offsets[:, None] * stride_w_k
            + offs_n[None, :] * stride_w_n
        )
        w = tl.load(
            w_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(act, w)
        k_base += BLOCK_K

    b_ptrs = b_ptr + pid_e * stride_b_e + offs_n * stride_b_n
    b = tl.load(b_ptrs, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    c_ptrs = (
        c_ptr + (row_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N))


def grouped_swiglu_w2(
    h_pre: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Fuse packed SwiGLU activation with the second expert projection on CUDA."""
    if h_pre.dim() != 2 or weights.dim() != 3:
        raise ValueError("grouped_swiglu_w2 expects h_pre [M,2K] and weights [E,K,N]")
    if not h_pre.is_cuda or not weights.is_cuda or not bias.is_cuda:
        raise ValueError("grouped_swiglu_w2 requires CUDA tensors")
    if offsets.dim() != 1 or lengths.dim() != 1:
        raise ValueError("offsets/lengths must be 1D")
    E, K, N = weights.shape
    if h_pre.shape[1] != 2 * K:
        raise ValueError(
            f"grouped_swiglu_w2 shape mismatch: h{tuple(h_pre.shape)} w{tuple(weights.shape)}"
        )
    try:
        w_dtype_tl = _TORCH_TO_TL_DTYPE[weights.dtype]
    except KeyError as exc:
        raise ValueError(
            f"grouped_swiglu_w2 unsupported weights dtype: {weights.dtype}"
        ) from exc
    if out_dtype is None:
        out_dtype = torch.float32
    c = torch.empty((h_pre.shape[0], N), device=h_pre.device, dtype=out_dtype)

    max_rows = int(lengths.max().item()) if lengths.numel() else 0
    if N >= 2048:
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        num_warps = 8
        num_stages = 4
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        num_warps = 4
        num_stages = 4
    grid = (triton.cdiv(max_rows, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
    _grouped_swiglu_w2_kernel[grid](
        h_pre,
        weights,
        bias,
        c,
        offsets,
        lengths,
        E,
        K,
        N,
        h_pre.stride(0),
        h_pre.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        bias.stride(0),
        bias.stride(1),
        c.stride(0),
        c.stride(1),
        ALPHA=alpha,
        LIMIT=limit,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        W_DTYPE=w_dtype_tl,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
