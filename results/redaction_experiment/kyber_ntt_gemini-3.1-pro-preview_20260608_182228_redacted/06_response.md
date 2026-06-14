Optimization: Fusion of the initial device memory load with the first NTT level (Level 0), and fusion of the final NTT level with the device memory writeback. This completely eliminates two round-trips through threadgroup memory and removes two `threadgroup_barrier` synchronizations. Memory accesses to device buffers remain coalesced because the threadgroup index mapping perfectly covers the required span with a fixed stride, while ALU operations execute directly from registers during these outer levels.

```metal
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
    bool over = (sum >= q) || (sum < a);
    return over ? (sum - q) : sum;
}

// 32-bit branchless modular subtraction
inline uint mod_sub_safe(uint a, uint b, uint q) {
    uint diff = a - b;
    return (a < b) ? (diff + q) : diff;
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
    uint n_levels_val = n_levels;
    uint half_n = n >> 1u;

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

    uint num_zetas = 1u << n_levels_val;

    // Collaboratively load and pre-convert zetas to Montgomery form
    if (ltid < num_zetas) {
        zeta_mont[ltid] = mont_mul(zetas[ltid], r2, q_val, q_inv);
    }
    if (ltid + half_n < num_zetas) {
        zeta_mont[ltid + half_n] = mont_mul(zetas[ltid + half_n], r2, q_val, q_inv);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Initial coalesced device load
    device uint *poly = coeffs + (size_t)tgid * n;
    uint x = poly[ltid];
    uint y = poly[ltid + half_n];
    
    // Fusion of Level 0 using register values directly
    if (n_levels_val > 0u) {
        uint z_mont = zeta_mont[1];
        uint t = mont_mul(y, z_mont, q_val, q_inv);
        uint orig_x = x;
        x = mod_add_safe(orig_x, t, q_val);
        y = mod_sub_safe(orig_x, t, q_val);
    }
    
    // Early exit if <= 1 levels (Level 0 result maps perfectly to final writeback)
    if (n_levels_val <= 1u) {
        poly[ltid] = x;
        poly[ltid + half_n] = y;
        return;
    }
    
    a[ltid] = x;
    a[ltid + half_n] = y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_len = (31u - clz(half_n)) - 1u;
    uint k_start  = 2u;
    
    // Middle levels: Cooley-Tukey NTT
    #pragma unroll
    for (uint level = 1u; level < 8u; ++level) {
        if (level >= n_levels_val - 1u) break;

        uint length = 1u << log2_len;
        
        // Fast bitwise indexing mapped to thread IDs
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (log2_len + 1u)) | j_in_group;
        
        uint z_mont = zeta_mont[k_start + group_idx];

        uint x_val = a[j];
        uint y_val = a[j + length];
        
        uint t = mont_mul(y_val, z_mont, q_val, q_inv);

        a[j]          = mod_add_safe(x_val, t, q_val);
        a[j + length] = mod_sub_safe(x_val, t, q_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        log2_len--;
        k_start <<= 1u;
    }

    // Fusion of Final Level with device memory writeback
    {
        uint length = 1u << log2_len;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (log2_len + 1u)) | j_in_group;
        
        uint z_mont = zeta_mont[k_start + group_idx];

        uint x_val = a[j];
        uint y_val = a[j + length];
        
        uint t = mont_mul(y_val, z_mont, q_val, q_inv);

        poly[j]          = mod_add_safe(x_val, t, q_val);
        poly[j + length] = mod_sub_safe(x_val, t, q_val);
    }
}
```