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

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return select(x, x - P_GOLD, x >= P_GOLD);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t = select(t, t + EPSILON, t < a);
    return select(t, t - P_GOLD, t >= P_GOLD);
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return select(t, t - EPSILON, t > a);
}

// 64x64 -> 128 multiply using Metal's native 32x32->high intrinsic.
// Low 64 bits come from a plain a*b. High 64 bits computed via mulhi
// on 32-bit halves plus cross products.
inline ulong umul_hi(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    // p00_hi = high 32 bits of (a0 * b0)
    uint  p00_hi = mulhi(a0, b0);
    ulong p01    = (ulong)a0 * (ulong)b1;   // 64-bit
    ulong p10    = (ulong)a1 * (ulong)b0;   // 64-bit
    ulong p11    = (ulong)a1 * (ulong)b1;   // 64-bit

    // mid = p00_hi + low32(p01) + low32(p10)  (fits in 34 bits)
    ulong mid = (ulong)p00_hi + (p01 & EPSILON) + (p10 & EPSILON);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return hi;
}

// Goldilocks multiplication: 64x64 -> reduce mod p = 2^64 - 2^32 + 1.
// Uses 2^64 ≡ 2^32 - 1 (mod p), i.e.
//   hi*2^64 + lo ≡ lo + hi_lo*(2^32 - 1) - hi_hi (mod p)
inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;            // implicit low 64 bits
    ulong hi = umul_hi(a, b);

    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    // t0 = lo - hi_hi (mod 2^64), correct underflow by subtracting EPSILON
    ulong t0 = lo - hi_hi;
    t0 = select(t0, t0 - EPSILON, t0 > lo);

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo, fits in u64
    ulong t1 = (hi_lo << 32) - hi_lo;

    // t2 = t0 + t1, correct overflow by adding EPSILON
    ulong t2 = t0 + t1;
    t2 = select(t2, t2 + EPSILON, t2 < t0);

    return gold_canonical(t2);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lN     = log_N;
    uint half_N = 1u << (lN - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;            // 2^s
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;
    uint jh = j << s;                    // j * 2^s

    uint i0 = jh + r;
    uint i1 = i0 + half_N;
    uint o0 = (jh << 1) + r;             // j * 2^(s+1) + r
    uint o1 = o0 + span_half;

    uint tw_shift = lN - 1u - s;
    ulong w = twiddles[(ulong)r << tw_shift];

    ulong u = in_data[i0];
    ulong v = gold_mul(in_data[i1], w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```

Result of previous attempt:
             N2_14: correct, 0.14 ms, 33.8 GB/s (16.9% of 200 GB/s)
             N2_16: correct, 0.36 ms, 57.5 GB/s (28.7% of 200 GB/s)
             N2_18: correct, 0.55 ms, 170.7 GB/s (85.3% of 200 GB/s)
  score (gmean of fraction): 0.3460

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
             N2_14: correct, 0.14 ms, 32.4 GB/s (16.2% of 200 GB/s)
             N2_16: correct, 0.20 ms, 102.7 GB/s (51.4% of 200 GB/s)
             N2_18: correct, 0.55 ms, 173.1 GB/s (86.6% of 200 GB/s)
  score (gmean of fraction): 0.4163

## History

- iter  0: compile=OK | correct=True | score=0.4163073121460872
- iter  1: compile=OK | correct=True | score=0.3675537227423188
- iter  2: compile=OK | correct=True | score=0.2683190300975542
- iter  3: compile=OK | correct=True | score=0.2678704613916114
- iter  4: compile=OK | correct=True | score=0.285666151720687
- iter  5: compile=OK | correct=True | score=0.3459816265110771

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
