To beat the incumbent while strictly obeying the constraint against baking `d_deg` via compile-time unrolling, we swap the order of evaluation loops. The previous attempt re-read `f_in` from global memory for each `t`, which caused severe L1 cache eviction since the memory footprint per `t` pass is large. 

1. **Loop inversion & Local Array:** We iterate over `i` on the outside and update all `d_deg + 1` evaluations on the inside using a `thread` array `ulong terms[32]`. This keeps `f_in` accesses optimal (1 read per value) while respecting runtime `d_deg` dispatch without violating the "no fixed unroll" contract.
2. **Specialized `umul128_32`:** Since `t` is always a small 32-bit integer, calculating `t * delta` doesn't require a full 64x64-bit multiplication. The custom `umul128_32` eliminates upper-half `p10` terms, drastically cutting the inner-loop instruction count.
3. **Index math optimization:** We maintain perfectly division-free, shift-minimized bitmask routing in `sumcheck_fold`.

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