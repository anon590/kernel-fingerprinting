#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Overflow-safe Montgomery reduction for any q <= 2^32 - 1
inline uint mont_mul(uint a, uint b, uint q, uint q_inv) {
    ulong P = (ulong)a * b;
    uint P_lo = (uint)P;
    uint P_hi = (uint)(P >> 32);
    
    uint Q = P_lo * q_inv;
    
    ulong M = (ulong)Q * q;
    uint M_hi = (uint)(M >> 32);
    
    ulong R_ext = (ulong)P_hi + M_hi + (P_lo != 0u ? 1u : 0u);
    return (uint)(R_ext >= (ulong)q ? R_ext - (ulong)q : R_ext);
}

// 32-bit safe modular addition
inline uint mod_add_safe(uint a, uint b, uint q) {
    uint sum = a + b;
    bool carry = sum < a;
    return (carry || sum >= q) ? sum - q : sum;
}

// 32-bit safe modular subtraction
inline uint mod_sub_safe(uint a, uint b, uint q) {
    uint diff = a - b;
    return diff + ((a < b) ? q : 0u);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    uint q_val = q;

    // Compute q_inv = -q^{-1} mod 2^32 using Newton-Raphson
    uint inv = q_val;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    uint q_inv = 0u - inv;

    // Compute R^2 mod q for converting twiddles to Montgomery form (R = 2^32)
    uint r = (uint)(4294967296ull % q_val);
    uint r2 = (uint)(((ulong)r * r) % q_val);

    threadgroup uint a[N_MAX];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Coalesced loads into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    
    // Cooley-Tukey NTT
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        
        // Only convert the twiddle factor to Montgomery form
        uint zeta       = zetas[k_start + group_idx];
        uint zeta_mont  = mont_mul(zeta, r2, q_val, q_inv);

        uint x = a[j];
        uint y = a[j + length];
        
        // Montgomery multiplication outputs in standard form because y is in standard form
        uint t = mont_mul(y, zeta_mont, q_val, q_inv);

        a[j]          = mod_add_safe(x, t, q_val);
        a[j + length] = mod_sub_safe(x, t, q_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    // Coalesced writeback
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}