#include <metal_stdlib>
using namespace metal;

// Overflow-safe Montgomery reduction for any q <= 2^32 - 1
inline uint mont_mul(uint a, uint b, uint q, uint q_inv) {
    uint P_lo = a * b;
    uint P_hi = mulhi(a, b);
    
    uint Q = P_lo * q_inv;
    uint M_hi = mulhi(Q, q);
    
    ulong T = (ulong)P_hi + M_hi + (P_lo != 0u ? 1u : 0u);
    return (uint)(T >= (ulong)q ? T - q : T);
}

// 32-bit branchless modular addition
inline uint mod_add_safe(uint a, uint b, uint q) {
    uint sum = a + b;
    return sum - ((sum >= q || sum < a) ? q : 0u);
}

// 32-bit branchless modular subtraction
inline uint mod_sub_safe(uint a, uint b, uint q) {
    uint diff = a - b;
    return diff + ((a < b) ? q : 0u);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
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
    uint r = (0xFFFFFFFFu % q_val) + 1u;
    r = (r == q_val) ? 0u : r;
    uint r2 = (uint)(((ulong)r * r) % q_val);

    threadgroup uint a[256];
    threadgroup uint zeta_mont[256];

    uint half_n = n >> 1u;
    uint num_zetas = 1u << n_levels;

    // Collaboratively load and pre-convert zetas to Montgomery form
    for (uint i = ltid; i < num_zetas; i += half_n) {
        zeta_mont[i] = mont_mul(zetas[i], r2, q_val, q_inv);
    }

    device uint *poly = coeffs + (size_t)tgid * n;
    
    // Ensure all twiddles are converted before processing
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Coalesced load & Level 0 (length 128)
    uint x = poly[ltid];
    uint y = poly[ltid + 128u];
    
    if (n_levels > 0u) {
        uint z = zeta_mont[1];
        uint t = mont_mul(y, z, q_val, q_inv);
        uint x_new = mod_add_safe(x, t, q_val);
        uint y_new = mod_sub_safe(x, t, q_val);
        x = x_new;
        y = y_new;
    }
    
    a[ltid] = x;
    a[ltid + 128u] = y;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Level 1 (length 64) via SMEM
    if (n_levels > 1u) {
        uint group_idx = ltid >> 6u;
        uint j_in_group = ltid & 63u;
        uint j = (group_idx << 7u) | j_in_group;
        
        uint z = zeta_mont[2u + group_idx];
        x = a[j];
        y = a[j + 64u];
        
        uint t = mont_mul(y, z, q_val, q_inv);
        a[j] = mod_add_safe(x, t, q_val);
        a[j + 64u] = mod_sub_safe(x, t, q_val);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: Levels 2..7 (lengths 32 down to 1) maintained purely in Registers
    if (n_levels > 2u) {
        uint g = ltid >> 5u;
        uint s = ltid & 31u;
        
        uint A = a[g * 64u + s];
        uint B = a[g * 64u + s + 32u];
        
        uint k_start = 4u;
        
        // Level 2 (length 32) evaluates correctly bound entirely within threads
        uint z = zeta_mont[k_start + g];
        uint t = mont_mul(B, z, q_val, q_inv);
        uint A_new = mod_add_safe(A, t, q_val);
        uint B_new = mod_sub_safe(A, t, q_val);
        A = A_new;
        B = B_new;
        
        k_start <<= 1u;
        
        // Levels 3..7 (lengths 16..1) using completely non-redundant SIMD shuffles
        #pragma unroll
        for (uint log2_L = 4u; log2_L < 5u; log2_L--) {
            uint level = 7u - log2_L;
            if (level >= n_levels) break;
            
            uint L = 1u << log2_L;
            bool is_low = (s & L) == 0u;
            
            // Branchlessly exchange halves
            uint expose = is_low ? B : A;
            uint other  = simd_shuffle_xor(expose, L);
            
            // Assort native pair for precisely ONE unique butterfly execution
            uint x_reg = is_low ? A : other;
            uint y_reg = is_low ? other : B;
            
            // Trace memory index logic mathematically matching graph layout
            uint start_idx = (s & ~L) + (is_low ? 0u : 32u);
            uint j_idx = g * 64u + start_idx;
            uint group_idx = j_idx >> (log2_L + 1u);
            
            uint z_mont = zeta_mont[k_start + group_idx];
            uint t_reg = mont_mul(y_reg, z_mont, q_val, q_inv);
            
            uint x_new_reg = mod_add_safe(x_reg, t_reg, q_val);
            uint y_new_reg = mod_sub_safe(x_reg, t_reg, q_val);
            
            // Scatter evaluated pairs back symmetrically (preserves spatial ownership invariant)
            uint expose_out = is_low ? y_new_reg : x_new_reg;
            uint recv_out   = simd_shuffle_xor(expose_out, L);
            
            A = is_low ? x_new_reg : recv_out;
            B = is_low ? recv_out : y_new_reg;
            
            k_start <<= 1u;
        }
        
        a[g * 64u + s] = A;
        a[g * 64u + s + 32u] = B;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Coalesced flush of finalized coefficients strictly formatted row-major  
    poly[ltid] = a[ltid];
    poly[ltid + 128u] = a[ltid + 128u];
}