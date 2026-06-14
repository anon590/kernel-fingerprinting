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

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul; // 2^32 - 1

// Canonical reduction: x in [0, 2*p) -> [0, p).
static inline ulong gold_canonical(ulong x) {
    return x - select((ulong)0, P_GOLD, x >= P_GOLD);
}

// Add mod p, inputs in [0, p), output in [0, p).
static inline ulong gold_add(ulong a, ulong b) {
    ulong t  = a + b;
    ulong c1 = select((ulong)0, EPSILON, t < a);   // wrap past 2^64
    t += c1;
    return t - select((ulong)0, P_GOLD, t >= P_GOLD);
}

// Sub mod p, inputs in [0, p), output in [0, p).
static inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return t - select((ulong)0, EPSILON, t > a);
}

// Reduce 128-bit value (lo + 2^64 * hi) mod Goldilocks.
// Identity: 2^64 ≡ 2^32 - 1, 2^96 ≡ -1 (mod p).
// Split hi = H1*2^32 + H0; result ≡ lo + H0*(2^32 - 1) - H1.
static inline ulong gold_reduce128(ulong lo, ulong hi) {
    uint2  hi_split = as_type<uint2>(hi);
    ulong  H0 = (ulong)hi_split.x;          // low 32 bits
    ulong  H1 = (ulong)hi_split.y;          // high 32 bits

    // t0 = lo - H1 (mod 2^64); borrow => subtract EPSILON.
    ulong t0 = lo - H1;
    t0 -= select((ulong)0, EPSILON, t0 > lo);

    // t1 = H0 * (2^32 - 1) = (H0 << 32) - H0, fits in u64 since H0 < 2^32.
    ulong t1 = (H0 << 32) - H0;

    // t2 = t0 + t1, with carry correction.
    ulong t2 = t0 + t1;
    t2 += select((ulong)0, EPSILON, t2 < t0);

    return t2 - select((ulong)0, P_GOLD, t2 >= P_GOLD);
}

static inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;
    ulong hi = mulhi(a, b);
    return gold_reduce128(lo, hi);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lg     = log_N;
    uint half_N = 1u << (lg - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;

    uint i0 = (j << s) + r;
    uint i1 = i0 + half_N;
    uint o0 = (j << (s + 1u)) + r;
    uint o1 = o0 + span_half;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r == 0 always; twiddle is 1.
        v = gold_canonical(x);
    } else {
        uint tw_stride = 1u << (lg - 1u - s);
        ulong w = twiddles[r * tw_stride];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```

Result of previous attempt:
             N2_14: correct, 0.14 ms, 32.8 GB/s (16.4% of 200 GB/s)
             N2_16: correct, 0.38 ms, 55.2 GB/s (27.6% of 200 GB/s)
             N2_18: correct, 1.07 ms, 88.6 GB/s (44.3% of 200 GB/s)
  score (gmean of fraction): 0.2717

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

// 128-bit reduction mod Goldilocks p = 2^64 - 2^32 + 1.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;        // both < 2^32, fits in u64

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    // Use Metal builtins: a*b gives low 64 bits, mulhi gives high 64.
    ulong lo = a * b;
    ulong hi = mulhi(a, b);
    return gold_reduce128(lo, hi);
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
    uint span_half = 1u << s;
    uint span      = span_half << 1u;

    uint mask = span_half - 1u;
    uint j    = k >> s;
    uint r    = k & mask;

    uint i0 = j * span_half + r;
    uint i1 = i0 + half_N;
    uint o0 = j * span + r;
    uint o1 = o0 + span_half;

    ulong u = in_data[i0];
    ulong x = in_data[i1];

    ulong v;
    if (s == 0u) {
        // r == 0 -> twiddle is omega^0 == 1
        v = gold_canonical(x);
    } else {
        uint tw_stride = 1u << (log_N - 1u - s);
        ulong w = twiddles[r * tw_stride];
        v = gold_mul(x, w);
    }

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```

Incumbent result:
             N2_14: correct, 0.14 ms, 32.1 GB/s (16.1% of 200 GB/s)
             N2_16: correct, 0.37 ms, 57.0 GB/s (28.5% of 200 GB/s)
             N2_18: correct, 0.51 ms, 184.1 GB/s (92.1% of 200 GB/s)
  score (gmean of fraction): 0.3481

## History

- iter  0: compile=OK | correct=True | score=0.33689900830783004
- iter  1: compile=OK | correct=True | score=0.3480917564016398
- iter  2: compile=OK | correct=True | score=0.33914027917250167
- iter  3: compile=OK | correct=True | score=0.27117774688676166
- iter  4: compile=OK | correct=True | score=0.27160687297626684
- iter  5: compile=OK | correct=True | score=0.28395414279500275
- iter  6: compile=OK | correct=True | score=0.2643226366693955
- iter  7: compile=OK | correct=True | score=0.2717164571121996

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
