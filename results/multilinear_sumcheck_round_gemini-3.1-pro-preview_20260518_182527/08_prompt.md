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

// ==========================================
// Field Helpers
// ==========================================

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
    ulong lo = a * b;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong mid = (ulong)mulhi(a0, b0) + (uint)p01 + (uint)p10;
    ulong hi  = (ulong)a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// Optimized 32-bit x 64-bit -> 96-bit multiplication for small multiplier 'a' (like t)
inline ulong2 umul128_32(uint a, ulong b) {
    ulong lo = (ulong)a * b;
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    ulong p01 = (ulong)a * b1;
    ulong mid = (ulong)mulhi(a, b0) + (uint)p01;
    ulong hi  = (p01 >> 32) + (mid >> 32);
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

// ==========================================
// Generic Dispatch
// ==========================================

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

template<uint prime_kind>
inline ulong mod_mul_32_t(uint a, ulong b) {
    if (prime_kind == 0u) {
        ulong2 prod = umul128_32(a, b);
        return gold_reduce128(prod.x, prod.y);
    } else {
        // P_BB fits in 32-bit; a product doesn't overflow a 64-bit uint here
        return ((ulong)a * b) % P_BB;
    }
}

// ==========================================
// SIMD Reduction Utility
// ==========================================

inline ulong simd_shuffle_xor_ulong(ulong val, ushort mask) {
    uint2 v = as_type<uint2>(val);
    v.x = simd_shuffle_xor(v.x, mask);
    v.y = simd_shuffle_xor(v.y, mask);
    return as_type<ulong>(v);
}

template<uint prime_kind>
inline ulong warp_reduce(ulong val) {
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 16));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 8));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 4));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 2));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 1));
    return val;
}

// ==========================================
// Core Kernels Implementation
// ==========================================

template<uint prime_kind>
inline void sumcheck_round_h_impl(
    device const ulong *f_in,
    device       ulong *partial,
    uint k_log,
    uint d_deg,
    uint gid, uint tid, uint tgid,
    threadgroup ulong *scratch)
{
    uint half_n = 1u << (k_log - 1u);
    
    // Support runtime d_deg bounds up to 31
    ulong terms[32];
    for (uint t = 0; t < 32; t++) {
        terms[t] = (gid < half_n) ? 1 : 0;
    }
    
    if (gid < half_n) {
        for (uint i = 0; i < d_deg; i++) {
            uint base = i << k_log;
            ulong f0 = f_in[base | gid];
            ulong f1 = f_in[base | half_n | gid];
            ulong delta = mod_sub_t<prime_kind>(f1, f0);
            
            terms[0] = mod_mul_t<prime_kind>(terms[0], f0);
            terms[1] = mod_mul_t<prime_kind>(terms[1], f1);
            
            for (uint t = 2; t <= d_deg; t++) {
                uint t_val = t;
                ulong t_delta = mod_mul_32_t<prime_kind>(t_val, delta);
                ulong eval = mod_add_t<prime_kind>(f0, t_delta);
                terms[t] = mod_mul_t<prime_kind>(terms[t], eval);
            }
        }
    }
    
    uint lane_id = tid & 31u;
    uint warp_id = tid >> 5u;
    
    // Each thread writes exactly one output element sum sequentially across threads
    for (uint t = 0; t <= d_deg; t++) {
        ulong val = terms[t];
        val = warp_reduce<prime_kind>(val);
        
        if (lane_id == 0) {
            scratch[warp_id] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong sum = (tid < 8u) ? scratch[tid] : 0;
            sum = warp_reduce<prime_kind>(sum);
            if (tid == 0) {
                partial[tgid * (d_deg + 1u) + t] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    uint total  = d_deg << (k_log - 1u);
    if (gid >= total) return;

    // Zero-division translation maps perfectly using pure masks
    uint mask = half_n - 1u;
    uint base_idx = ((gid & ~mask) << 1u) | (gid & mask);

    ulong f0    = f_in[base_idx];
    ulong f1    = f_in[base_idx | half_n];
    
    ulong delta = mod_sub_t<prime_kind>(f1, f0);
    ulong fold_val = mod_add_t<prime_kind>(f0, mod_mul_t<prime_kind>(r, delta));
    
    f_out[gid] = fold_val;
}

// ==========================================
// Kernel Entry Points
// ==========================================

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
    threadgroup ulong scratch[32];
    if (prime_kind == 0u) {
        sumcheck_round_h_impl<0u>(f_in, partial, k_log, d_deg, gid, tid, tgid, scratch);
    } else {
        sumcheck_round_h_impl<1u>(f_in, partial, k_log, d_deg, gid, tid, tgid, scratch);
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
       gold_k14_d2: correct, 0.13 ms, 3.1 GB/s (1.5% of 200 GB/s)
       gold_k16_d2: correct, 0.30 ms, 5.2 GB/s (2.6% of 200 GB/s)
       gold_k18_d2: correct, 0.48 ms, 13.1 GB/s (6.5% of 200 GB/s)
  score (gmean of fraction): 0.0296

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;

// ==========================================
// Field Helpers
// ==========================================

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
    ulong lo = a * b;
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong mid = (ulong)mulhi(a0, b0) + (uint)p01 + (uint)p10;
    ulong hi  = (ulong)a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
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

// ==========================================
// Generic Dispatch
// ==========================================

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

// ==========================================
// SIMD Reduction Utility
// ==========================================

inline ulong simd_shuffle_xor_ulong(ulong val, ushort mask) {
    uint2 v = as_type<uint2>(val);
    v.x = simd_shuffle_xor(v.x, mask);
    v.y = simd_shuffle_xor(v.y, mask);
    return as_type<ulong>(v);
}

template<uint prime_kind>
inline ulong warp_reduce(ulong val) {
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 16));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 8));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 4));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 2));
    val = mod_add_t<prime_kind>(val, simd_shuffle_xor_ulong(val, 1));
    return val;
}

// ==========================================
// Core Kernels Implementation
// ==========================================

template<uint prime_kind>
inline void sumcheck_round_h_impl(
    device const ulong *f_in,
    device       ulong *partial,
    uint k_log,
    uint d_deg,
    uint gid, uint tid, uint tgid,
    threadgroup ulong *scratch)
{
    uint half_n = 1u << (k_log - 1u);
    ulong term0 = 0, term1 = 0, term2 = 0, term3 = 0;
    
    if (gid < half_n) {
        ulong f0 = f_in[gid];
        ulong f1 = f_in[half_n | gid];
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
        
        if (d_deg >= 2) {
            uint base1 = 1u << k_log;
            ulong f0_1 = f_in[base1 | gid];
            ulong f1_1 = f_in[base1 | half_n | gid];
            ulong delta_1 = mod_sub_t<prime_kind>(f1_1, f0_1);
            
            term0 = mod_mul_t<prime_kind>(term0, f0_1);
            term1 = mod_mul_t<prime_kind>(term1, f1_1);
            ulong f2_1 = mod_add_t<prime_kind>(f1_1, delta_1);
            term2 = mod_mul_t<prime_kind>(term2, f2_1);
            if (d_deg >= 3) {
                ulong f3_1 = mod_add_t<prime_kind>(f2_1, delta_1);
                term3 = mod_mul_t<prime_kind>(term3, f3_1);
            }
        }
        
        if (d_deg >= 3) {
            uint base2 = 2u << k_log;
            ulong f0_2 = f_in[base2 | gid];
            ulong f1_2 = f_in[base2 | half_n | gid];
            ulong delta_2 = mod_sub_t<prime_kind>(f1_2, f0_2);
            
            term0 = mod_mul_t<prime_kind>(term0, f0_2);
            term1 = mod_mul_t<prime_kind>(term1, f1_2);
            ulong f2_2 = mod_add_t<prime_kind>(f1_2, delta_2);
            term2 = mod_mul_t<prime_kind>(term2, f2_2);
            ulong f3_2 = mod_add_t<prime_kind>(f2_2, delta_2);
            term3 = mod_mul_t<prime_kind>(term3, f3_2);
        }
    }
    
    // Stage 1: Intra-warp SIMD reduction
    term0 = warp_reduce<prime_kind>(term0);
    term1 = warp_reduce<prime_kind>(term1);
    if (d_deg >= 2) term2 = warp_reduce<prime_kind>(term2);
    if (d_deg >= 3) term3 = warp_reduce<prime_kind>(term3);
    
    uint lane_id = tid & 31u;
    uint warp_id = tid >> 5u;
    
    if (lane_id == 0) {
        scratch[warp_id * 4 + 0] = term0;
        scratch[warp_id * 4 + 1] = term1;
        if (d_deg >= 2) scratch[warp_id * 4 + 2] = term2;
        if (d_deg >= 3) scratch[warp_id * 4 + 3] = term3;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Stage 2: Warp 0 aggregates the warp results
    if (tid < 32) {
        if (tid < 8) {
            term0 = scratch[tid * 4 + 0];
            term1 = scratch[tid * 4 + 1];
            if (d_deg >= 2) term2 = scratch[tid * 4 + 2];
            if (d_deg >= 3) term3 = scratch[tid * 4 + 3];
        } else {
            term0 = 0; term1 = 0; term2 = 0; term3 = 0;
        }
        
        term0 = warp_reduce<prime_kind>(term0);
        term1 = warp_reduce<prime_kind>(term1);
        if (d_deg >= 2) term2 = warp_reduce<prime_kind>(term2);
        if (d_deg >= 3) term3 = warp_reduce<prime_kind>(term3);
        
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

    // Direct mapping perfectly mirrors index layouts without expensive integer division
    uint poly_i = gid >> (k_log - 1u);
    uint j      = gid & (half_n - 1u);

    uint base_idx = (poly_i << k_log) | j;

    ulong f0    = f_in[base_idx];
    ulong f1    = f_in[base_idx | half_n];
    
    ulong delta = mod_sub_t<prime_kind>(f1, f0);
    ulong fold_val = mod_add_t<prime_kind>(f0, mod_mul_t<prime_kind>(r, delta));
    
    f_out[gid] = fold_val;
}

// ==========================================
// Kernel Entry Points
// ==========================================

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
    threadgroup ulong scratch[32];
    if (prime_kind == 0u) {
        sumcheck_round_h_impl<0u>(f_in, partial, k_log, d_deg, gid, tid, tgid, scratch);
    } else {
        sumcheck_round_h_impl<1u>(f_in, partial, k_log, d_deg, gid, tid, tgid, scratch);
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

Incumbent result:
       gold_k14_d2: correct, 0.04 ms, 9.2 GB/s (4.6% of 200 GB/s)
       gold_k16_d2: correct, 0.08 ms, 19.7 GB/s (9.8% of 200 GB/s)
       gold_k18_d2: correct, 0.12 ms, 53.6 GB/s (26.8% of 200 GB/s)
  score (gmean of fraction): 0.1066

## History

- iter  0: compile=OK | correct=True | score=0.014646783785662443
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.10655517912362927
- iter  3: compile=OK | correct=True | score=0.07007763352938885
- iter  4: compile=OK | correct=True | score=0.07297757293633618
- iter  5: compile=FAIL | correct=False | score=N/A
- iter  6: compile=OK | correct=True | score=0.07750887758690553
- iter  7: compile=OK | correct=True | score=0.029645977877498328

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
