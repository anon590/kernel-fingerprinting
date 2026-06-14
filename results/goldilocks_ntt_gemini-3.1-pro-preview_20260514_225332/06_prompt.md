## Task: goldilocks_ntt

Forward Number-Theoretic Transform of length N = 2^log_N over the Goldilocks prime field p = 2^64 - 2^32 + 1. Convention:
  Y[k] = sum_{n=0}^{N-1} X[n] * omega_N^(k * n)   (mod p)
where omega_N is the primitive N-th root of unity in Goldilocks (the host computes it from the standard plonky2 / risc0 root g_root_2^32 = 1753635133440165772 and uploads tw[i] = omega_N^i for i in [0, N/2) as the twiddle buffer).

Storage is row-major ulong[N]; one independent NTT per task dispatch (batch=1). Bit-exact correctness against a Python bigint reference — any mismatched output element rejects the candidate.

The host implements one butterfly stage per kernel dispatch and ping-pongs (in_data, out_data) across log_N dispatches. Per-stage indexing contract: at stage s, the butterfly pair index k in [0, N/2) decomposes as
  j = k >> s, r = k & ((1 << s) - 1),
the read offsets are (j * 2^s + r) and (j * 2^s + r + N/2), and the write offsets are (j * 2^(s+1) + r) and the same + 2^s. The twiddle for this butterfly is tw[r * (N >> (s + 1))].

All arithmetic is over Goldilocks; outputs MUST be canonical (< p), i.e. a value in [p, 2^64) is treated as a mismatch even if its residue class matches.

## Required kernel signature(s)

```
kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]]);

Dispatch geometry (host-fixed, identical across stages):
  threadsPerGrid        = (N/2, 1, 1)
  threadsPerThreadgroup = (min(N/2, 256), 1, 1)
Each thread owns exactly one butterfly pair; guard against k >= N/2 (the grid is rounded up to a multiple of the TG width). The host invokes the kernel log_N times in one command buffer with stage_idx = 0, 1, ..., log_N - 1, ping-ponging (in_data, out_data) between two device buffers; the final NTT result lands in the buffer selected by the parity of log_N. Twiddle and log_N buffers are bound once.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

#define P_GOLD  0xFFFFFFFF00000001ul
#define EPSILON 0x00000000FFFFFFFFul

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong x_lo = (uint)p00 | (mid << 32);
    ulong x_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;
    
    if (t0 > x_lo) {
        t0 -= EPSILON;
    }

    ulong t2 = t0 + t1;
    if (t2 < t0) {
        t2 += EPSILON;
    } else if (t2 >= P_GOLD) {
        t2 -= P_GOLD;
    }
    return t2;
}

inline ulong gold_add(ulong u, ulong v) {
    ulong out0 = u + v;
    if (out0 < u) {
        out0 += EPSILON;
    } else if (out0 >= P_GOLD) {
        out0 -= P_GOLD;
    }
    return out0;
}

inline ulong gold_sub(ulong u, ulong v) {
    ulong out1 = u - v;
    if (out1 > u) {
        out1 -= EPSILON;
    }
    return out1;
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k                         [[thread_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_idx                    [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]])
{
    uint half_N = 1u << (log_N - 1u);
    bool active = k < half_N;

    uint s = stage_idx;
    uint mask = (1u << s) - 1u;
    uint not_mask = ~mask;

    ulong out0 = 0, out1 = 0;
    if (active) {
        ulong u = in_data[k];
        ulong v = in_data[k + half_N];

        if (s != 0u) {
            uint tw_idx = (k & mask) << (log_N - 1u - s);
            v = gold_mul(v, twiddles[tw_idx]);
        }

        out0 = gold_add(u, v);
        out1 = gold_sub(u, v);
    }

    // Coalesce writes using threadgroup memory for narrow butterfly spans
    if ((1u << s) < tg_size) {
        threadgroup ulong shared_out[512];
        
        if (active) {
            uint local_o0 = ((tid & not_mask) << 1u) | (tid & mask);
            uint local_o1 = local_o0 | (1u << s);
            shared_out[local_o0] = out0;
            shared_out[local_o1] = out1;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint base_out = (tg_idx * tg_size) << 1u;
        uint active_pairs = (half_N > tg_idx * tg_size) ? (half_N - tg_idx * tg_size) : 0;
        uint valid_elements = min(active_pairs, tg_size) << 1u;
        
        // Cooperatively write back the continuous block generated by active threads
        if (tid < valid_elements) {
            out_data[base_out + tid] = shared_out[tid];
        }
        if (tid + tg_size < valid_elements) {
            out_data[base_out + tid + tg_size] = shared_out[tid + tg_size];
        }
    } else {
        // Native coalescing applies when stride spans beyond threadgroup dimension
        if (active) {
            uint o0 = ((k & not_mask) << 1u) | (k & mask);
            uint o1 = o0 | (1u << s);
            out_data[o0] = out0;
            out_data[o1] = out1;
        }
    }
}
```

Result of previous attempt:
             N2_14: correct, 0.15 ms, 30.3 GB/s (15.2% of 200 GB/s)
             N2_16: correct, 0.36 ms, 58.7 GB/s (29.4% of 200 GB/s)
             N2_18: correct, 1.08 ms, 87.0 GB/s (43.5% of 200 GB/s)
  score (gmean of fraction): 0.2686

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

#define P_GOLD  0xFFFFFFFF00000001ul
#define EPSILON 0x00000000FFFFFFFFul

// 64x64 -> 128 unsigned multiplication with purely 32-bit splits
inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    // Standard 32x32 -> 64 hardware multiplication
    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    // Fast tracking of carries via 32-bit truncation (free register sub-selects)
    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong x_lo = (uint)p00 | (mid << 32);
    ulong x_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    // Branchless Goldilocks reduction
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0;

    // Equivalent to x_hi_lo * EPSILON but executed in fast shift/sub arithmetic
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint half_N = 1u << (log_N - 1u);
    if (k >= half_N) return;

    uint s = stage_idx;
    uint mask = (1u << s) - 1u;
    uint not_mask = ~mask;

    // Linear memory access profile: regardless of stage, Thread k ALWAYS maps 
    // strictly to k and k + N/2. Coalescing here is unconditionally 100%.
    ulong u = in_data[k];
    ulong v = in_data[k + half_N];

    // Uniformly bypass all multiplication overhead during stage 0
    if (s != 0u) {
        uint tw_idx = (k & mask) << (log_N - 1u - s);
        v = gold_mul(v, twiddles[tw_idx]);
    }

    // Direct mapping to the standard interleaved Cooley-Tukey pattern, without
    // runtime index multiplication (only bitwise ops and shifts).
    uint o0 = ((k & not_mask) << 1u) | (k & mask);
    uint o1 = o0 | (1u << s);

    // gold_add
    ulong out0 = u + v;
    out0 += (out0 < u) ? EPSILON : 0;
    out_data[o0] = (out0 >= P_GOLD) ? (out0 - P_GOLD) : out0;

    // gold_sub
    ulong out1 = u - v;
    out1 -= (out1 > u) ? EPSILON : 0;
    out_data[o1] = out1;
}
```

Incumbent result:
             N2_14: correct, 0.09 ms, 49.3 GB/s (24.6% of 200 GB/s)
             N2_16: correct, 0.18 ms, 115.6 GB/s (57.8% of 200 GB/s)
             N2_18: correct, 1.06 ms, 89.3 GB/s (44.6% of 200 GB/s)
  score (gmean of fraction): 0.3990

## History

- iter  0: compile=OK | correct=True | score=0.28455122587807063
- iter  1: compile=OK | correct=True | score=0.28021139272830276
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=0.2828191873116982
- iter  4: compile=OK | correct=True | score=0.39897535842765014
- iter  5: compile=OK | correct=True | score=0.2685764552207826

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
