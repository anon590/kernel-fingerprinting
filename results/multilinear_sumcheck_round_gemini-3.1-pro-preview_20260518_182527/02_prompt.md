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

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;

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
    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo  = (uint)p00 | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = (uint)(x_hi >> 32);
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

template<uint prime_kind>
inline ulong mod_add_t(ulong a, ulong b) {
    return (prime_kind == 0u) ? gold_add(a, b) : bb_add(a, b);
}

template<uint prime_kind>
inline ulong mod_sub_t(ulong a, ulong b) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

template<uint prime_kind>
inline ulong mod_mul_t(ulong a, ulong b) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

inline ulong simd_shuffle_down_ulong(ulong val, ushort offset) {
    uint2 v = as_type<uint2>(val);
    v = simd_shuffle_down(v, offset);
    return as_type<ulong>(v);
}

template<uint prime_kind>
inline void sumcheck_round_h_impl(
    device const ulong *f_in,
    device       ulong *partial,
    uint k_log,
    uint d_deg,
    uint gid, uint tid, uint tgid)
{
    uint half_n = 1u << (k_log - 1u);
    ulong term0 = 0, term1 = 0, term2 = 0, term3 = 0;
    
    if (gid < half_n) {
        if (d_deg >= 1) { 
            uint f0_idx = gid;
            uint f1_idx = f0_idx | half_n;
            ulong f0 = f_in[f0_idx];
            ulong f1 = f_in[f1_idx];
            ulong delta = mod_sub_t<prime_kind>(f1, f0);
            
            term0 = f0;
            term1 = f1;
            if (d_deg >= 2) {
                ulong f2 = mod_add_t<prime_kind>(f1, delta);
                term2 = f2;
                if (d_deg >= 3) {
                    ulong f3 = mod_add_t<prime_kind>(f2, delta);
                    term3 = f3;
                }
            }
        }
        if (d_deg >= 2) {
            uint f0_idx = (1u << k_log) | gid;
            uint f1_idx = f0_idx | half_n;
            ulong f0 = f_in[f0_idx];
            ulong f1 = f_in[f1_idx];
            ulong delta = mod_sub_t<prime_kind>(f1, f0);
            
            term0 = mod_mul_t<prime_kind>(term0, f0);
            term1 = mod_mul_t<prime_kind>(term1, f1);
            ulong f2 = mod_add_t<prime_kind>(f1, delta);
            term2 = mod_mul_t<prime_kind>(term2, f2);
            if (d_deg >= 3) {
                ulong f3 = mod_add_t<prime_kind>(f2, delta);
                term3 = mod_mul_t<prime_kind>(term3, f3);
            }
        }
        if (d_deg >= 3) {
            uint f0_idx = (2u << k_log) | gid;
            uint f1_idx = f0_idx | half_n;
            ulong f0 = f_in[f0_idx];
            ulong f1 = f_in[f1_idx];
            ulong delta = mod_sub_t<prime_kind>(f1, f0);
            
            term0 = mod_mul_t<prime_kind>(term0, f0);
            term1 = mod_mul_t<prime_kind>(term1, f1);
            ulong f2 = mod_add_t<prime_kind>(f1, delta);
            term2 = mod_mul_t<prime_kind>(term2, f2);
            ulong f3 = mod_add_t<prime_kind>(f2, delta);
            term3 = mod_mul_t<prime_kind>(term3, f3);
        }
    }
    
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        term0 = mod_add_t<prime_kind>(term0, simd_shuffle_down_ulong(term0, offset));
        term1 = mod_add_t<prime_kind>(term1, simd_shuffle_down_ulong(term1, offset));
        term2 = mod_add_t<prime_kind>(term2, simd_shuffle_down_ulong(term2, offset));
        term3 = mod_add_t<prime_kind>(term3, simd_shuffle_down_ulong(term3, offset));
    }
    
    threadgroup ulong scratch[32];
    
    if (tid % 32 == 0) {
        uint warp_id = tid / 32;
        scratch[warp_id * 4 + 0] = term0;
        scratch[warp_id * 4 + 1] = term1;
        scratch[warp_id * 4 + 2] = term2;
        scratch[warp_id * 4 + 3] = term3;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < 32) {
        if (tid < 8) {
            term0 = scratch[tid * 4 + 0];
            term1 = scratch[tid * 4 + 1];
            term2 = scratch[tid * 4 + 2];
            term3 = scratch[tid * 4 + 3];
        } else {
            term0 = 0; term1 = 0; term2 = 0; term3 = 0;
        }
        
        for (ushort offset = 4; offset > 0; offset >>= 1) {
            term0 = mod_add_t<prime_kind>(term0, simd_shuffle_down_ulong(term0, offset));
            term1 = mod_add_t<prime_kind>(term1, simd_shuffle_down_ulong(term1, offset));
            term2 = mod_add_t<prime_kind>(term2, simd_shuffle_down_ulong(term2, offset));
            term3 = mod_add_t<prime_kind>(term3, simd_shuffle_down_ulong(term3, offset));
        }
        
        if (tid == 0) {
            partial[tgid * (d_deg + 1) + 0] = term0;
            partial[tgid * (d_deg + 1) + 1] = term1;
            if (d_deg >= 2) partial[tgid * (d_deg + 1) + 2] = term2;
            if (d_deg >= 3) partial[tgid * (d_deg + 1) + 3] = term3;
        }
    }
}

template<uint prime_kind>
inline void sumcheck_fold_impl(
    device const ulong *f_in,
    device       ulong *f_out,
    ulong r,
    uint k_log,
    uint d_deg,
    uint gid)
{
    uint half_n = 1u << (k_log - 1u);
    uint total = d_deg << (k_log - 1u);
    if (gid >= total) return;

    uint poly_i = gid >> (k_log - 1u);
    uint j      = gid & (half_n - 1u);

    uint base_idx = (poly_i << k_log) | j;

    ulong f0    = f_in[base_idx];
    ulong f1    = f_in[base_idx | half_n];
    
    ulong delta = mod_sub_t<prime_kind>(f1, f0);
    ulong fold_val = mod_add_t<prime_kind>(f0, mod_mul_t<prime_kind>(r, delta));
    
    f_out[gid] = fold_val;
}

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
    if (prime_kind == 0u) {
        sumcheck_round_h_impl<0u>(f_in, partial, k_log, d_deg, gid, tid, tgid);
    } else {
        sumcheck_round_h_impl<1u>(f_in, partial, k_log, d_deg, gid, tid, tgid);
    }
}

kernel void sumcheck_fold(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *f_out      [[buffer(1)]],
    constant ulong     &r          [[buffer(2)]],
    constant uint      &k_log      [[buffer(3)]],
    constant uint      &d_deg      [[buffer(4)]],
    constant uint      &prime_kind [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (prime_kind == 0u) {
        sumcheck_fold_impl<0u>(f_in, f_out, r, k_log, d_deg, gid);
    } else {
        sumcheck_fold_impl<1u>(f_in, f_out, r, k_log, d_deg, gid);
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:159:5: error: variables in the threadgroup address space cannot be declared in a non-qualified function
    threadgroup ulong scratch[32];
    ^
" UserInfo={NSLocalizedDescription=program_source:159:5: error: variables in the threadgroup address space cannot be declared in a non-qualified function
    threadgroup ulong scratch[32];
    ^
}

## Current best (incumbent)

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

Incumbent result:
       gold_k14_d2: correct, 0.14 ms, 2.7 GB/s (1.4% of 200 GB/s)
       gold_k16_d2: correct, 0.76 ms, 2.1 GB/s (1.0% of 200 GB/s)
       gold_k18_d2: correct, 1.42 ms, 4.4 GB/s (2.2% of 200 GB/s)
  score (gmean of fraction): 0.0146

## History

- iter  0: compile=OK | correct=True | score=0.014646783785662443
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
