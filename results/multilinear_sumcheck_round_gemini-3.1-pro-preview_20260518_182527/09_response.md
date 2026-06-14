To push beyond the incumbent while strictly satisfying the requirement to handle arbitrary runtime `d_deg` without "baking in" bounds, we apply several powerful optimisations:

1. **Dynamic T-Chunking & Loop Peeling**: Instead of unrolling based on static `d_deg` checks (which violates the contract), we evaluate `t` in chunks of 4. We peel the first iteration (`i=0`) to entirely eliminate redundant multiply-by-1 operations. This dynamically matches the optimal instruction count of the statically unrolled incumbent for $d=2$ without hardcoding any paths.
2. **Pointer Arithmetic for Poly Evaluation**: We replace `base | half_n | gid` inside the evaluation loop with a simple pointer advance `f_ptr += stride`. This removes bitwise ALU operations inside the hot loop.
3. **Fast 32x64 Multiplication**: We introduce an optimized `mod_mul_32_t` that avoids a full 64x64->128 multiply. Since `t_start` is 32-bit, we compute only two 32x64 partial products, halving the arithmetic in the `t_start > 0` branches.
4. **Zero-Bit Insertion in Fold**: We optimize `sumcheck_fold` by replacing the shift-and-OR index reconstruction with `gid + (gid & ~(half_n - 1u))`. This exploits two's complement arithmetic to perfectly inject the `0` bit at $k-1$ with just one bitwise AND and an ADD.

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

template<uint prime_kind>
inline ulong mod_mul_32_t(uint a, ulong b) {
    if (prime_kind == 0u) {
        ulong lo = (ulong)a * b;
        ulong p0 = (ulong)a * (uint)b;
        ulong p1 = (ulong)a * (uint)(b >> 32);
        ulong hi = (p1 >> 32) + (((p0 >> 32) + (uint)p1) >> 32);
        return gold_reduce128(lo, hi);
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
    uint stride = 1u << k_log;
    
    uint lane_id = tid & 31u;
    uint warp_id = tid >> 5u;
    
    for (uint t_start = 0; t_start <= d_deg; t_start += 4) {
        ulong term0 = 0, term1 = 0, term2 = 0, term3 = 0;
        uint t_end = (d_deg + 1 >= t_start + 4) ? 4 : (d_deg + 1 - t_start);
        
        if (gid < half_n) {
            term0 = 1; term1 = 1; term2 = 1; term3 = 1;
            device const ulong* f_ptr = f_in + gid;
            
            if (t_start == 0) {
                if (d_deg > 0) {
                    ulong f0 = f_ptr[0];
                    ulong f1 = f_ptr[half_n];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    term0 = f0;
                    term1 = f1;
                    term2 = mod_add_t<prime_kind>(f1, delta);
                    term3 = mod_add_t<prime_kind>(term2, delta);
                    f_ptr += stride;
                }
                for (uint i = 1; i < d_deg; i++) {
                    ulong f0 = f_ptr[0];
                    ulong f1 = f_ptr[half_n];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    ulong e2 = mod_add_t<prime_kind>(f1, delta);
                    ulong e3 = mod_add_t<prime_kind>(e2, delta);

                    term0 = mod_mul_t<prime_kind>(term0, f0);
                    term1 = mod_mul_t<prime_kind>(term1, f1);
                    term2 = mod_mul_t<prime_kind>(term2, e2);
                    term3 = mod_mul_t<prime_kind>(term3, e3);
                    f_ptr += stride;
                }
            } else {
                if (d_deg > 0) {
                    ulong f0 = f_ptr[0];
                    ulong f1 = f_ptr[half_n];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    ulong offset = mod_mul_32_t<prime_kind>(t_start, delta);
                    term0 = mod_add_t<prime_kind>(f0, offset);
                    term1 = mod_add_t<prime_kind>(term0, delta);
                    term2 = mod_add_t<prime_kind>(term1, delta);
                    term3 = mod_add_t<prime_kind>(term2, delta);
                    f_ptr += stride;
                }
                for (uint i = 1; i < d_deg; i++) {
                    ulong f0 = f_ptr[0];
                    ulong f1 = f_ptr[half_n];
                    ulong delta = mod_sub_t<prime_kind>(f1, f0);

                    ulong offset = mod_mul_32_t<prime_kind>(t_start, delta);
                    ulong e0 = mod_add_t<prime_kind>(f0, offset);
                    ulong e1 = mod_add_t<prime_kind>(e0, delta);
                    ulong e2 = mod_add_t<prime_kind>(e1, delta);
                    ulong e3 = mod_add_t<prime_kind>(e2, delta);

                    term0 = mod_mul_t<prime_kind>(term0, e0);
                    term1 = mod_mul_t<prime_kind>(term1, e1);
                    term2 = mod_mul_t<prime_kind>(term2, e2);
                    term3 = mod_mul_t<prime_kind>(term3, e3);
                    f_ptr += stride;
                }
            }
        }
        
        term0 = warp_reduce<prime_kind>(term0);
        term1 = warp_reduce<prime_kind>(term1);
        term2 = warp_reduce<prime_kind>(term2);
        term3 = warp_reduce<prime_kind>(term3);
        
        if (lane_id == 0) {
            scratch[warp_id * 4 + 0] = term0;
            scratch[warp_id * 4 + 1] = term1;
            scratch[warp_id * 4 + 2] = term2;
            scratch[warp_id * 4 + 3] = term3;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (tid < 32) {
            ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            if (tid < 8) {
                s0 = scratch[tid * 4 + 0];
                s1 = scratch[tid * 4 + 1];
                s2 = scratch[tid * 4 + 2];
                s3 = scratch[tid * 4 + 3];
            }
            s0 = warp_reduce<prime_kind>(s0);
            s1 = warp_reduce<prime_kind>(s1);
            s2 = warp_reduce<prime_kind>(s2);
            s3 = warp_reduce<prime_kind>(s3);
            
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

    // Fast injection of the 0 bit at position (k_log - 1)
    uint mask_inv = ~(half_n - 1u);
    uint base_idx = gid + (gid & mask_inv);

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