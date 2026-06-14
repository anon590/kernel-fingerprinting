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