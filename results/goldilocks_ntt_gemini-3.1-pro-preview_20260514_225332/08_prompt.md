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

// Safe 64-bit shuffle utilizing native 32-bit uint2 hardware shuffles
inline ulong shuffle_ulong(ulong val, ushort lane) {
    uint2 v = as_type<uint2>(val);
    return as_type<ulong>(uint2(simd_shuffle(v.x, lane), simd_shuffle(v.y, lane)));
}

// 64x64 -> 128 unsigned multiplication with native 64-bit lo-word and optimized hi-word logic
inline ulong gold_mul(ulong a, ulong b) {
    ulong x_lo = a * b; // Natively gives exactly the lower 64 bits modulo 2^64
    
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;

    // Must cast mulhi to ulong to ensure sum does not overflow 32-bit
    ulong mid = (ulong)mulhi(a0, b0) + (uint)p01 + (uint)p10;
    ulong x_hi = (ulong)a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0;

    // Equivalent to x_hi_lo * EPSILON but via shift/sub
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

    // Input memory accesses map 100% contiguously to thread execution widths
    ulong u = in_data[k];
    ulong v = in_data[k + half_N];

    if (s != 0u) {
        uint tw_idx = (k & mask) << (log_N - 1u - s);
        v = gold_mul(v, twiddles[tw_idx]);
    }

    // gold_add
    ulong out0 = u + v;
    out0 += (out0 < u) ? EPSILON : 0;
    out0 = (out0 >= P_GOLD) ? (out0 - P_GOLD) : out0;

    // gold_sub
    ulong out1 = u - v;
    out1 -= (out1 > u) ? EPSILON : 0;

    // In early stages (s <= 5), outputs natively scatter. We fully coalesce memory accesses 
    // by exchanging values across the 32-lane SIMD warp to assemble consecutive ulong2 blocks.
    // For s >= 6, structural block widths naturally guarantee full contiguous strides per warp.
    if (s <= 5u) {
        if (s > 0u) {
            uint lane = k & 31u;
            uint m = lane & mask;
            
            bool use_out1 = (m >= (1u << (s - 1u)));
            uint src_m0 = use_out1 ? (2u * m - (1u << s)) : (2u * m);
            uint src_lane = (lane & not_mask) + src_m0;
            
            ulong v0_0 = shuffle_ulong(out0, (ushort)src_lane);
            ulong v0_1 = shuffle_ulong(out1, (ushort)src_lane);
            ulong v1_0 = shuffle_ulong(out0, (ushort)(src_lane + 1u));
            ulong v1_1 = shuffle_ulong(out1, (ushort)(src_lane + 1u));
            
            out0 = use_out1 ? v0_1 : v0_0;
            out1 = use_out1 ? v1_1 : v1_0;
        }
        // Vectorized 128-bit contiguous stores replacing scalar isolated outputs
        ((device ulong2*)out_data)[k] = ulong2(out0, out1);
    } else {
        uint o0 = ((k & not_mask) << 1u) | (k & mask);
        uint o1 = o0 | (1u << s);
        out_data[o0] = out0;
        out_data[o1] = out1;
    }
}
```

Result of previous attempt:
             N2_14: correct, 0.16 ms, 29.3 GB/s (14.7% of 200 GB/s)
             N2_16: correct, 0.40 ms, 52.2 GB/s (26.1% of 200 GB/s)
             N2_18: correct, 1.24 ms, 76.2 GB/s (38.1% of 200 GB/s)
  score (gmean of fraction): 0.2444

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
- iter  6: compile=FAIL | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.24438647316282536

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
