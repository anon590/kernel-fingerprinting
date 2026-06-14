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

constant ulong P_GOLD       = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON      = 0x00000000FFFFFFFFul;  // 2^32 - 1
constant ulong ROOT4_POS    = 0x0001000000000000ul;  // 2^48, square = -1
constant ulong ROOT8_24_POS = 0x0000000001000000ul;  // 2^24, square = 2^48
constant ulong ROOT8_72_POS = 0x000000FFFFFFFF00ul;  // 2^72 mod p = 2^40 - 2^8

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

// Fold a 128-bit integer lo + hi*2^64 modulo p.
// Uses 2^64 == 2^32 - 1 and 2^96 == -1.
inline ulong gold_reduce128_fold(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong sub = hi_lo + hi_hi;
    ulong t = lo - sub;
    t -= (t > lo) ? EPSILON : 0ul;

    ulong add = hi_lo << 32;
    ulong r = t + add;
    r += (r < t) ? EPSILON : 0ul;

    return (r >= P_GOLD) ? (r - P_GOLD) : r;
}

// Incumbent-style 32x32->64 partial products, good on the smaller cases.
inline ulong2 umul128_wide(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    return ulong2(lo, hi);
}

inline ulong gold_reduce128_wide(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul_wide(ulong a, ulong b) {
    ulong2 p = umul128_wide(a, b);
    return gold_reduce128_wide(p.x, p.y);
}

// Explicit mulhi-based 32-bit limb multiplication. This was faster for the
// largest benchmark size in the prior measurements.
inline ulong gold_mul_limb(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);

    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);

    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);

    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;

    uint z1 = s1 + p10_lo;
    c1 += (z1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;

    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;

    uint z2 = s3 + c1;
    c2 += (z2 < s3) ? 1u : 0u;

    uint z3 = p11_hi + c2;

    ulong lo = ((ulong)z1 << 32) | (ulong)p00_lo;

    ulong sub = (ulong)z2 + (ulong)z3;
    ulong t = lo - sub;
    t -= (t > lo) ? EPSILON : 0ul;

    ulong add = ((ulong)z2) << 32;
    ulong r = t + add;
    r += (r < t) ? EPSILON : 0ul;

    return (r >= P_GOLD) ? (r - P_GOLD) : r;
}

inline ulong gold_mul_selected(ulong a, ulong b, uint log_N) {
    if (log_N >= 18u) {
        return gold_mul_limb(a, b);
    } else {
        return gold_mul_wide(a, b);
    }
}

// Exact multiplication by small powers of two in the field.
inline ulong gold_mul_pow2_8(ulong x) {
    return gold_reduce128_fold(x << 8, x >> 56);
}

inline ulong gold_mul_pow2_24(ulong x) {
    return gold_reduce128_fold(x << 24, x >> 40);
}

inline ulong gold_mul_pow2_40(ulong x) {
    return gold_reduce128_fold(x << 40, x >> 24);
}

inline ulong gold_mul_pow2_48(ulong x) {
    return gold_reduce128_fold(x << 48, x >> 16);
}

// 2^72 == 2^40 - 2^8 mod p.
inline ulong gold_mul_pow2_72(ulong x) {
    return gold_sub(gold_mul_pow2_40(x), gold_mul_pow2_8(x));
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

    // Stage 0: twiddle is 1 for every butterfly.
    if (s == 0u) {
        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        uint o = k << 1u;
        out_data[o]      = gold_add(u, x);
        out_data[o + 1u] = gold_sub(u, x);
        return;
    }

    // Stage 1: twiddles are {1, +/-2^48}; avoid generic multiplication.
    if (s == 1u) {
        uint r = k & 1u;

        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        ulong root4 = twiddles[half_N >> 1u];
        ulong y = gold_mul_pow2_48(x);
        if (root4 != ROOT4_POS) {
            y = gold_neg(y);
        }

        ulong v = (r == 0u) ? x : y;

        uint o0 = (k << 1u) - r;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 2u] = gold_sub(u, v);
        return;
    }

    // Stage 2: twiddles are 8th roots, all +/-2^24 or +/-2^72, with
    // the square equal to +/-2^48. Replace full field mul by shifts/folds.
    if (s == 2u) {
        uint r = k & 3u;

        ulong u = in_data[k];
        ulong x = in_data[k + half_N];

        uint stride = half_N >> 2u;
        ulong root8 = twiddles[stride];
        ulong root4 = twiddles[stride << 1u];

        ulong y24 = gold_mul_pow2_24(x);
        ulong y48 = gold_mul_pow2_48(x);
        ulong y72 = gold_mul_pow2_72(x);

        ulong y1;
        ulong y2;
        ulong y3;

        if (root4 == ROOT4_POS) {
            bool neg8 = (root8 != ROOT8_24_POS);
            y1 = neg8 ? gold_neg(y24) : y24;
            y2 = y48;
            y3 = neg8 ? gold_neg(y72) : y72;
        } else {
            bool neg8 = (root8 != ROOT8_72_POS);
            y1 = neg8 ? gold_neg(y72) : y72;
            y2 = gold_neg(y48);
            y3 = neg8 ? gold_neg(y24) : y24;
        }

        ulong v = (r == 0u) ? x : ((r == 1u) ? y1 : ((r == 2u) ? y2 : y3));

        uint o0 = (k << 1u) - r;
        out_data[o0]      = gold_add(u, v);
        out_data[o0 + 4u] = gold_sub(u, v);
        return;
    }

    // Last stage has contiguous output and twiddle index k.
    if (s == (log_N - 1u)) {
        ulong u = in_data[k];
        ulong x = in_data[k + half_N];
        ulong w = twiddles[k];
        ulong v = gold_mul_selected(x, w, log_N);

        out_data[k]          = gold_add(u, v);
        out_data[k + half_N] = gold_sub(u, v);
        return;
    }

    uint m = 1u << s;
    uint r = k & (m - 1u);

    ulong u = in_data[k];
    ulong x = in_data[k + half_N];

    uint tw_stride = half_N >> s;
    ulong w = twiddles[r * tw_stride];
    ulong v = gold_mul_selected(x, w, log_N);

    uint o0 = (k << 1u) - r;
    out_data[o0]     = gold_add(u, v);
    out_data[o0 + m] = gold_sub(u, v);
}
```

Result of previous attempt:
             N2_14: correct, 0.18 ms, 25.3 GB/s (12.7% of 200 GB/s)
             N2_16: correct, 0.40 ms, 52.1 GB/s (26.1% of 200 GB/s)
             N2_18: correct, 1.11 ms, 84.7 GB/s (42.3% of 200 GB/s)
  score (gmean of fraction): 0.2408

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

- iter  2: compile=OK | correct=True | score=0.2652357240858739
- iter  3: compile=OK | correct=True | score=0.2592166484198962
- iter  4: compile=OK | correct=True | score=0.26001788777791823
- iter  5: compile=OK | correct=True | score=0.25942034018539134
- iter  6: compile=OK | correct=True | score=0.24440839677068896
- iter  7: compile=OK | correct=True | score=0.26346747757121947
- iter  8: compile=OK | correct=True | score=0.3094244760068847
- iter  9: compile=OK | correct=True | score=0.24076953500721077

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
