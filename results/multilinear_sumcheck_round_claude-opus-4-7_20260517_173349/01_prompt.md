## Task: multilinear_sumcheck_round

One degree-d sumcheck round on a product polynomial g(x) = f_0(x) * f_1(x) * ... * f_{d-1}(x), where each f_i: {0,1}^k -> F_p is multilinear, stored as a length 2^k_log table of evaluations on the Boolean hypercube. The kernel folds the FIRST variable: it emits (A) the univariate round polynomial h(X) = sum_{x' in {0,1}^(k-1)} prod_i f_i(X, x'), represented by its d+1 evaluations h(0), h(1), ..., h(d); and (B) the folded factor tables f_i_new[j] = f_i(r, j) for j in [0, 2^(k-1)), where r is the verifier-supplied round challenge in [0, p).

Layout convention. The variable being folded is the most significant bit of the hypercube index, so for j in [0, 2^(k-1)) the X = 0 and X = 1 slices are
  f_i^(0)[j] = f_in[i * 2^k_log + j]
  f_i^(1)[j] = f_in[i * 2^k_log + j + 2^(k-1)]
The multilinear extension along the first variable, evaluated at any X in F_p, is the unique affine interpolant
  f_i(X, j) = f_i^(0)[j] + X * (f_i^(1)[j] - f_i^(0)[j])   (mod p)
so the kernel must produce, in one round,
  h(t)       = sum_{j in [0, 2^(k-1))} prod_i f_i(t, j)
               for t in {0, 1, ..., d}
  f_i_new[j] = f_i(r, j)
               for i in [0, d) and j in [0, 2^(k-1)).

Two-kernel pipeline (host issues both in ONE compute command encoder; the serial encoder gives an implicit barrier so kernel B observes A's writes):
  Dispatch 1 (sumcheck_round_h): each threadgroup owns     256 consecutive pair indices in [0, half) where     half = 2^(k_log - 1). For each pair index j the     thread contributes the d+1 per-pair products     prod_i f_i(t, j); the threadgroup cooperatively     reduces 256 contributions per t into one tile sum     and writes d+1 contiguous ulongs to     partial[tgid * (d+1) + t]. Threads with gid >=     half contribute 0 (additive identity for the sum).
  Dispatch 2 (sumcheck_fold): one thread per output     (poly_i, j); writes one folded coefficient to     f_out[poly_i * half + j]. Guard against gid >= d *     half (the grid is rounded up to a multiple of the TG     width).

The host then sums partial[0..K-1] per t on the CPU (K = ceil(half / 256), ~1 KB total -- intentionally untimed) to obtain h_evals[0..d+1], and cross-checks the sumcheck consistency identity h(0) + h(1) == sum_x prod_i f_i(x). A candidate whose h_evals matches a same-buggy reference but indexes the linear extension the wrong way silently fails this identity.

Field selection (constant prime_kind):
  0 = Goldilocks   p = 2^64 - 2^32 + 1
  1 = BabyBear     p = 2^31 - 2^27 + 1 = 2013265921
Both reductions, the per-pair t-loop, and the threadgroup geometry must dispatch on the RUNTIME values of prime_kind, d_deg, and k_log. Baking any of them in as a compile-time constant -- a specific reduction macro, a fixed unroll over t, a hardcoded buffer stride, ... -- violates the kernel contract.

All field elements (f_in, partial, f_out, r) are canonical uint64 in [0, p); a non-canonical output is treated as a correctness failure even if its residue class matches the reference.

## Required kernel signature(s)

```
kernel void sumcheck_round_h(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *partial    [[buffer(1)]],
    constant uint      &k_log      [[buffer(2)]],
    constant uint      &d_deg      [[buffer(3)]],
    constant uint      &prime_kind [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]);

kernel void sumcheck_fold(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *f_out      [[buffer(1)]],
    constant ulong     &r          [[buffer(2)]],
    constant uint      &k_log      [[buffer(3)]],
    constant uint      &d_deg      [[buffer(4)]],
    constant uint      &prime_kind [[buffer(5)]],
    uint gid [[thread_position_in_grid]]);

Dispatch geometry (host-fixed):
  sumcheck_round_h:
    threadsPerGrid        = (K * 256, 1, 1)   K = ceil(half / 256)
    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256
  sumcheck_fold:
    threadsPerGrid        = (d * half rounded up to TG width, 1, 1)
    threadsPerThreadgroup = (min(d * half, 256), 1, 1)

The 256-wide threadgroup is part of the host-kernel contract for sumcheck_round_h: K = ceil(half / 256) is baked into the host-side partial[] allocation, so the kernel must emit exactly one (d+1)-element tile sum per 256 consecutive pair indices.
```

## Baseline: naive seed kernel

```metal
// Naive seed for one degree-d multilinear sumcheck round (Z13).
//
// Given:
//   * d  factor multilinear polynomials f_0, ..., f_{d-1} on {0,1}^k,
//     stored as flat tables of length 2^k each. Element j of f_i is
//     f_in[i * (1 << k_log) + j].
//   * a verifier round challenge r in [0, p).
//
// Compute, in one round:
//   (A) h(X) = sum_{x' in {0,1}^(k-1)} prod_i f_i(X, x'),
//       represented by its d+1 evaluations h(0), h(1), ..., h(d).
//   (B) the folded tables f_i_new[j] = f_i(r, j)
//                          = f_i^(0)[j] + r * (f_i^(1)[j] - f_i^(0)[j])
//       for j in [0, 2^(k-1)), where f_i^(0)[j] = f_in[i * 2^k + j] and
//       f_i^(1)[j] = f_in[i * 2^k + j + 2^(k-1)].
//
// Bit-exact correctness invariants the host cross-checks:
//   * h(0) + h(1) == sum_x prod_i f_i(x)             (sumcheck identity)
//   * h(r)       == sum_y prod_i f_i_new[y]          (round closure)
//
// Two-kernel pipeline (host issues both inside ONE compute encoder; the
// serial mode gives an implicit read-after-write barrier between them):
//
//   1) sumcheck_round_h
//        threadsPerGrid        = (K * TG_WIDTH, 1, 1)   K = ceil(half / TG_WIDTH)
//        threadsPerThreadgroup = (TG_WIDTH = 256, 1, 1)        // FIXED
//        Each threadgroup owns 256 consecutive pair indices in
//        [0, half), where half = 2^(k_log - 1). For its pair index j
//        each thread contributes the d+1 per-pair products
//        prod_i f_i(t, j) for t in {0, 1, ..., d}; threads with
//        gid >= half contribute zeros (additive identity for the
//        sum). The threadgroup reduces its 256 per-t contributions
//        into one tile sum per t and writes d+1 contiguous ulongs
//        to partial[tgid * (d+1) + t]. The host folds K partials
//        per t on the CPU to obtain h(t); that K-element host fold
//        is sub-millisecond and intentionally untimed.
//
//   2) sumcheck_fold
//        threadsPerGrid        = (d * half rounded up, 1, 1)
//        threadsPerThreadgroup = (min(d * half, 256), 1, 1)
//        Each thread owns one output (poly_i, j) and writes one
//        folded coefficient: f_out[poly_i * half + j].
//
// Field selection (constant prime_kind):
//   0 = Goldilocks   p = 2^64 - 2^32 + 1
//   1 = BabyBear     p = 2^31 - 2^27 + 1 = 2013265921
// Both reductions, the per-pair t-loop, and the threadgroup geometry
// must dispatch on the RUNTIME values of prime_kind, d_deg, and
// k_log; baking any of them in as a compile-time constant violates
// the kernel contract.
//
// Buffer layout (host-fixed, must be preserved by candidate kernels):
//
//   sumcheck_round_h:
//     buffer 0: device const ulong *f_in       (length d * 2^k_log)
//     buffer 1: device       ulong *partial    (length K * (d+1))
//     buffer 2: constant uint &k_log
//     buffer 3: constant uint &d_deg           (1..MAX_D)
//     buffer 4: constant uint &prime_kind      (0 = Goldilocks, 1 = BabyBear)
//
//   sumcheck_fold:
//     buffer 0: device const ulong *f_in       (length d * 2^k_log)
//     buffer 1: device       ulong *f_out      (length d * 2^(k_log - 1))
//     buffer 2: constant ulong &r              (round challenge, canonical < p)
//     buffer 3: constant uint  &k_log
//     buffer 4: constant uint  &d_deg
//     buffer 5: constant uint  &prime_kind
//
// All field elements (inputs, partials, outputs) are canonical uint64
// in [0, p); a non-canonical output is treated as a correctness
// failure even if its residue class matches the reference.

#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;        // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;        // 2^32 - 1
constant ulong P_BB    = 2013265921ul;                // 2^31 - 2^27 + 1

constant uint  TG_WIDTH = 256u;
constant uint  MAX_D    = 3u;

// ---------------------- Goldilocks helpers ----------------------------

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

inline ulong2 umul128(ulong a, ulong b) {
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

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

// ---------------------- BabyBear helpers ------------------------------
// All BabyBear elements fit in 31 bits, so a * b fits in 62 bits and
// the % operator is well-defined on uint64.

inline ulong bb_add(ulong a, ulong b) {
    ulong t = a + b;
    return (t >= P_BB) ? (t - P_BB) : t;
}

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

// ---------------------- Generic dispatch ------------------------------

inline ulong mod_add(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_add(a, b) : bb_add(a, b);
}

inline ulong mod_sub(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

inline ulong mod_mul(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

// ----------------------------------------------------------------------
// Kernel A: per-tile partial sums of h(t) for t = 0..d.
//
// Per thread, per pair index j in [0, half):
//   for each factor i in [0, d):
//     f0_i    = f_in[i * 2^k + j]
//     f1_i    = f_in[i * 2^k + j + half]
//     delta_i = f1_i - f0_i                       (mod p)
//     for t in {0, 1, ..., d}:
//         f_i(t, j) = f0_i + t * delta_i          (mod p)
//   accumulate per-t product over i.
// The threadgroup reduces its 256 per-t contributions into one tile
// sum per t; the result is written at partial[tgid * (d+1) + t].
// ----------------------------------------------------------------------

kernel void sumcheck_round_h(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *partial    [[buffer(1)]],
    constant uint      &k_log      [[buffer(2)]],
    constant uint      &d_deg      [[buffer(3)]],
    constant uint      &prime_kind [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong scratch[TG_WIDTH * (MAX_D + 1)];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    ulong term[MAX_D + 1];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

    if (gid < half_n) {
        // Per-t product accumulator over the d factors. Init to 1.
        ulong prod_t[MAX_D + 1];
        for (uint t = 0u; t <= d; ++t) prod_t[t] = 1ul;

        for (uint i = 0u; i < d; ++i) {
            ulong f0 = f_in[i * base + gid];
            ulong f1 = f_in[i * base + gid + half_n];
            ulong delta = mod_sub(f1, f0, prime_kind);
            for (uint t = 0u; t <= d; ++t) {
                ulong ft = mod_add(
                    f0,
                    mod_mul((ulong)t, delta, prime_kind),
                    prime_kind);
                prod_t[t] = mod_mul(prod_t[t], ft, prime_kind);
            }
        }
        for (uint t = 0u; t <= d; ++t) term[t] = prod_t[t];
    }

    for (uint t = 0u; t <= d; ++t) {
        scratch[tid * (MAX_D + 1u) + t] = term[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TG_WIDTH >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            for (uint t = 0u; t <= d; ++t) {
                ulong a = scratch[tid * (MAX_D + 1u) + t];
                ulong b = scratch[(tid + stride) * (MAX_D + 1u) + t];
                scratch[tid * (MAX_D + 1u) + t] = mod_add(a, b, prime_kind);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        for (uint t = 0u; t <= d; ++t) {
            partial[tgid * (d + 1u) + t] = scratch[t];
        }
    }
}

// ----------------------------------------------------------------------
// Kernel B: fold each factor table along the first variable.
//
// Per thread (gid = poly_i * half + j):
//   f0    = f_in[poly_i * 2^k + j]
//   f1    = f_in[poly_i * 2^k + j + half]
//   delta = f1 - f0                          (mod p)
//   f_out[poly_i * half + j] = f0 + r * delta (mod p)
// ----------------------------------------------------------------------

kernel void sumcheck_fold(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *f_out      [[buffer(1)]],
    constant ulong     &r          [[buffer(2)]],
    constant uint      &k_log      [[buffer(3)]],
    constant uint      &d_deg      [[buffer(4)]],
    constant uint      &prime_kind [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    uint total = d * half_n;
    if (gid >= total) return;

    uint poly_i = gid / half_n;
    uint j      = gid % half_n;

    ulong f0    = f_in[poly_i * base + j];
    ulong f1    = f_in[poly_i * base + j + half_n];
    ulong delta = mod_sub(f1, f0, prime_kind);
    ulong fold  = mod_add(f0,
                          mod_mul(r, delta, prime_kind),
                          prime_kind);
    f_out[poly_i * half_n + j] = fold;
}
```

Measured baseline (seed):
       gold_k14_d2: correct, 0.27 ms, 1.5 GB/s (0.7% of 200 GB/s)
       gold_k16_d2: correct, 0.48 ms, 3.3 GB/s (1.6% of 200 GB/s)
       gold_k18_d2: correct, 1.08 ms, 5.8 GB/s (2.9% of 200 GB/s)
  score (gmean of fraction): 0.0152

## Your task

Write an improved Metal kernel that produces bit-exact-correct
results AND runs faster than the seed across all problem sizes. The
fitness score is the geometric mean of `achieved / ceiling` across
sizes; score 0 if any size fails bit-exact correctness.

Output ONE fenced ```metal``` code block containing the kernel(s).
Preserve the kernel name(s) and buffer indices exactly.
