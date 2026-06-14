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

template<uint prime_kind>
inline ulong mod_mul_32_t(uint a, ulong b) {
    if (prime_kind == 0u) {
        // LLVM optimizes the high-half arithmetic perfectly for 32-bit `a`
        ulong2 prod = umul128((ulong)a, b);
        return gold_reduce128(prod.x, prod.y);
    } else {
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
    uint lane_id = tid & 31u;
    uint warp_id = tid >> 5u;

    for (uint t_start = 0; t_start <= d_deg; t_start += 4) {
        ulong term0 = 0, term1 = 0, term2 = 0, term3 = 0;
        if (gid < half_n) {
            term0 = 1; term1 = 1; term2 = 1; term3 = 1;
        }

        uint t_end = (t_start + 4 <= d_deg + 1) ? 4 : (d_deg + 1 - t_start);

        if (gid < half_n) {
            if (t_end == 4) {
                for (uint i = 0; i < d_deg; i++) {
                    ulong base = (ulong)i << k_log;
                    ulong f0 = f_in[base | gid];
                    ulong f1 = f_in[base | half_n | gid];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    ulong curr = (t_start == 0) ? f1 : mod_add_t<prime_kind>(f0, mod_mul_32_t<prime_kind>(t_start, delta));
                    ulong e0 = (t_start == 0) ? f0 : curr;
                    term0 = mod_mul_t<prime_kind>(term0, e0);
                    
                    ulong e1 = (t_start == 0) ? f1 : mod_add_t<prime_kind>(curr, delta);
                    curr = e1;
                    term1 = mod_mul_t<prime_kind>(term1, e1);
                    
                    ulong e2 = mod_add_t<prime_kind>(curr, delta);
                    curr = e2;
                    term2 = mod_mul_t<prime_kind>(term2, e2);
                    
                    ulong e3 = mod_add_t<prime_kind>(curr, delta);
                    term3 = mod_mul_t<prime_kind>(term3, e3);
                }
            } else if (t_end == 3) {
                for (uint i = 0; i < d_deg; i++) {
                    ulong base = (ulong)i << k_log;
                    ulong f0 = f_in[base | gid];
                    ulong f1 = f_in[base | half_n | gid];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    ulong curr = (t_start == 0) ? f1 : mod_add_t<prime_kind>(f0, mod_mul_32_t<prime_kind>(t_start, delta));
                    ulong e0 = (t_start == 0) ? f0 : curr;
                    term0 = mod_mul_t<prime_kind>(term0, e0);
                    
                    ulong e1 = (t_start == 0) ? f1 : mod_add_t<prime_kind>(curr, delta);
                    curr = e1;
                    term1 = mod_mul_t<prime_kind>(term1, e1);
                    
                    ulong e2 = mod_add_t<prime_kind>(curr, delta);
                    term2 = mod_mul_t<prime_kind>(term2, e2);
                }
            } else if (t_end == 2) {
                for (uint i = 0; i < d_deg; i++) {
                    ulong base = (ulong)i << k_log;
                    ulong f0 = f_in[base | gid];
                    ulong f1 = f_in[base | half_n | gid];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    ulong curr = (t_start == 0) ? f1 : mod_add_t<prime_kind>(f0, mod_mul_32_t<prime_kind>(t_start, delta));
                    ulong e0 = (t_start == 0) ? f0 : curr;
                    term0 = mod_mul_t<prime_kind>(term0, e0);
                    
                    ulong e1 = (t_start == 0) ? f1 : mod_add_t<prime_kind>(curr, delta);
                    term1 = mod_mul_t<prime_kind>(term1, e1);
                }
            } else if (t_end == 1) {
                for (uint i = 0; i < d_deg; i++) {
                    ulong base = (ulong)i << k_log;
                    ulong f0 = f_in[base | gid];
                    ulong e0 = f0;
                    if (t_start > 0) {
                        ulong f1 = f_in[base | half_n | gid];
                        ulong delta = mod_sub_t<prime_kind>(f1, f0);
                        e0 = mod_add_t<prime_kind>(f0, mod_mul_32_t<prime_kind>(t_start, delta));
                    }
                    term0 = mod_mul_t<prime_kind>(term0, e0);
                }
            }
        }

        // Concurrent Intra-Warp SIMD reduction
        if (t_end > 0) term0 = warp_reduce<prime_kind>(term0);
        if (t_end > 1) term1 = warp_reduce<prime_kind>(term1);
        if (t_end > 2) term2 = warp_reduce<prime_kind>(term2);
        if (t_end > 3) term3 = warp_reduce<prime_kind>(term3);

        if (lane_id == 0) {
            scratch[warp_id * 4 + 0] = term0;
            scratch[warp_id * 4 + 1] = term1;
            scratch[warp_id * 4 + 2] = term2;
            scratch[warp_id * 4 + 3] = term3;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Final Warp 0 aggregation
        if (tid < 32) {
            ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            if (tid < 8) {
                s0 = scratch[tid * 4 + 0];
                s1 = scratch[tid * 4 + 1];
                s2 = scratch[tid * 4 + 2];
                s3 = scratch[tid * 4 + 3];
            }
            if (t_end > 0) s0 = warp_reduce<prime_kind>(s0);
            if (t_end > 1) s1 = warp_reduce<prime_kind>(s1);
            if (t_end > 2) s2 = warp_reduce<prime_kind>(s2);
            if (t_end > 3) s3 = warp_reduce<prime_kind>(s3);
            
            if (tid == 0) {
                if (t_end > 0) partial[tgid * (d_deg + 1) + t_start + 0] = s0;
                if (t_end > 1) partial[tgid * (d_deg + 1) + t_start + 1] = s1;
                if (t_end > 2) partial[tgid * (d_deg + 1) + t_start + 2] = s2;
                if (t_end > 3) partial[tgid * (d_deg + 1) + t_start + 3] = s3;
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
    uint total = d_deg << (k_log - 1u);
    if (gid >= total) return;

    uint poly_i = gid >> (k_log - 1u);
    uint j      = gid & (half_n - 1u);

    ulong base_idx = ((ulong)poly_i << k_log) | j;

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
    threadgroup ulong scratch[32]; // 8 warps * 4 terms concurrently
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