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

constant ulong P_GOLD      = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON     = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_POS   = 0x0001000000000000ul;  // 2^48, square root of -1
constant ulong ROOT4_NEG   = 0xFFFEFFFF00000001ul;  // p - 2^48

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? EPSILON : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    t -= (t > a) ? EPSILON : 0ul;
    return t;
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

// 64x64 -> 128 using explicit 32-bit products and mulhi().
inline ulong2 umul128_u32(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);

    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);

    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);

    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    uint mid1   = p00_hi + p01_lo;
    uint carry1 = (mid1 < p00_hi) ? 1u : 0u;

    uint mid2   = mid1 + p10_lo;
    uint carry2 = (mid2 < mid1) ? 1u : 0u;

    ulong lo = ((ulong)mid2 << 32) | (ulong)p00_lo;

    ulong hi = ((ulong)p11_hi << 32) | (ulong)p11_lo;
    hi += (ulong)p01_hi + (ulong)p10_hi + (ulong)carry1 + (ulong)carry2;

    return ulong2(lo, hi);
}

// Reduce x_lo + x_hi * 2^64 modulo p, returning canonical representative.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong hi_lo = x_hi & EPSILON;
    ulong hi_hi = x_hi >> 32;

    ulong t0 = x_lo - hi_hi;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    // hi_lo * (2^32 - 1), with hi_lo < 2^32.
    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128_u32(a, b);
    return gold_reduce128(p.x, p.y);
}

// Multiply by 2^48 modulo Goldilocks.
inline ulong gold_mul_root4_pos(ulong x) {
    return gold_reduce128(x << 48, x >> 16);
}

inline ulong gold_mul_maybe_root4(ulong x, ulong w) {
    if (w == ROOT4_POS) {
        return gold_mul_root4_pos(x);
    } else if (w == ROOT4_NEG) {
        return gold_neg(gold_mul_root4_pos(x));
    } else {
        return gold_mul(x, w);
    }
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

    // Stage 0 has twiddle 1 for every butterfly.
    if (s == 0u) {
        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        uint o0 = k << 1;
        out_data[o0]      = gold_add(u, x);
        out_data[o0 + 1u] = gold_sub(u, x);
        return;
    }

    uint span_half = 1u << s;
    uint r         = k & (span_half - 1u);

    uint o0 = (k << 1) - r;
    uint o1 = o0 + span_half;

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    ulong v;

    // Early stages have many twiddle=1 and twiddle=root4 cases.
    // For later stages the hit rate is tiny, so use the straight path.
    if (s <= 4u) {
        if (r == 0u) {
            v = x;
        } else {
            uint tw_idx = r << (log_N - 1u - s);

            if (r == (span_half >> 1)) {
                v = gold_mul_maybe_root4(x, twiddles[tw_idx]);
            } else {
                v = gold_mul(x, twiddles[tw_idx]);
            }
        }
    } else {
        uint tw_idx = r << (log_N - 1u - s);
        v = gold_mul(x, twiddles[tw_idx]);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```

Result of previous attempt:
             N2_14: correct, 0.16 ms, 28.2 GB/s (14.1% of 200 GB/s)
             N2_16: correct, 0.38 ms, 54.5 GB/s (27.3% of 200 GB/s)
             N2_18: correct, 1.11 ms, 85.2 GB/s (42.6% of 200 GB/s)
  score (gmean of fraction): 0.2540

## Current best (incumbent)

```metal
// Naive seed for the Goldilocks NTT (forward, length N = 2^log_N).
//
// One butterfly stage per kernel dispatch. The host ping-pongs
// (in_data, out_data) across log_N dispatches; the final result lands
// in the buffer determined by the parity of log_N.
//
// Convention (matches the reference in metal_zk/reference/goldilocks.py):
//   Y[k] = sum_n X[n] * omega_N^(k * n)   (mod p)
// where omega_N is the primitive N-th root of unity in Goldilocks
// (p = 2^64 - 2^32 + 1). The twiddle buffer holds tw[i] = omega_N^i for
// i in [0, N/2); stage s reads tw[r * (N >> (s+1))].
//
// Per-thread work:
//   thread k in [0, N/2) -> one butterfly pair.
//   j = k >> s, r = k & ((1<<s) - 1)
//   read offsets : i0 = j*(1<<s) + r           (low half)
//                  i1 = i0 + N/2                (high half)
//   write offsets: o0 = j*(1<<(s+1)) + r
//                  o1 = o0 + (1<<s)
//   twiddle      : tw[r * (N >> (s+1))]        // = omega_(2^(s+1))^r
//   butterfly    : u = in[i0]; v = gold_mul(in[i1], tw);
//                  out[o0] = gold_add(u, v);
//                  out[o1] = gold_sub(u, v);
//
// Dispatch (host-provided):
//   threadsPerGrid       = (N/2, 1, 1)
//   threadsPerThreadgroup= (min(N/2, 256), 1, 1)
//
// Buffer layout (must be preserved by candidate kernels):
//   buffer 0: device const ulong *in_data   (length N)
//   buffer 1: device       ulong *out_data  (length N)
//   buffer 2: device const ulong *twiddles  (length N/2, omega_N^i)
//   buffer 3: const uint &stage_idx         (in [0, log_N))
//   buffer 4: const uint &log_N

#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD   = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON  = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;     // u64 overflow -> add 2^32 - 1 (mod p)
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;     // u64 underflow -> subtract 2^32 - 1
    return t;
}

// 64x64 -> 128 unsigned multiplication via 4-way 32x32 split.
// Returns (lo, hi).
inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    // Each (ulong)u32 * (ulong)u32 fits in a 64-bit product.
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid    = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo     = (p00 & EPSILON) | (mid << 32);
    ulong hi     = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Goldilocks reduction of a 128-bit value (x_lo, x_hi):
//   x mod p = x_lo + (x_hi & 2^32-1) * (2^32 - 1) - (x_hi >> 32)
// Implemented with overflow / underflow fixups; one final canonicalize.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    // t0 = x_lo - x_hi_hi
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;     // borrow -> -= (2^32 - 1) (mod p)

    // t1 = x_hi_lo * (2^32 - 1)
    ulong t1 = x_hi_lo * EPSILON;     // fits in u64 (both factors < 2^32)

    // t2 = t0 + t1
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;       // carry -> += (2^32 - 1) (mod p)

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
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

    uint s         = stage_idx;
    uint span_half = 1u << s;          // 2^s
    uint span      = span_half << 1u;  // 2^(s+1)

    uint mask = span_half - 1u;
    uint j    = (s == 0u) ? k : (k >> s);
    uint r    = k & mask;              // mask = 0 when s = 0 -> r = 0

    uint i0 = j * span_half + r;
    uint i1 = i0 + half_N;
    uint o0 = j * span + r;
    uint o1 = o0 + span_half;

    // tw_stride = N / span = 2^(log_N - s - 1)
    uint tw_stride = 1u << (log_N - 1u - s);
    ulong w  = twiddles[r * tw_stride];

    ulong u  = in_data[i0];
    ulong v  = gold_mul(in_data[i1], w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```

Incumbent result:
             N2_14: correct, 0.14 ms, 33.6 GB/s (16.8% of 200 GB/s)
             N2_16: correct, 0.12 ms, 170.4 GB/s (85.2% of 200 GB/s)
             N2_18: correct, 1.02 ms, 92.7 GB/s (46.4% of 200 GB/s)
  score (gmean of fraction): 0.4050

## History

- iter  0: compile=OK | correct=True | score=0.404984691100057
- iter  1: compile=OK | correct=True | score=0.2540411746181689

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
