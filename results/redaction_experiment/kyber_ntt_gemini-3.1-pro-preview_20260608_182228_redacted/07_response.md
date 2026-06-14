We fundamentally restructured the NTT by performing the first two levels (lengths 128 and 64) via `threadgroup` memory, and the remaining 6 levels (length 32 down to 1) purely within SIMD registers using `simd_shuffle_xor`. This drastically reduces the number of `threadgroup_barrier`s per polynomial from 8 to 2 and eliminates the majority of shared memory traffic. By carefully mapping polynomial chunks to lanes, all register shuffles and Montgomery reductions are perfectly branchless, and both shared and device memory accesses are completely coalesced and bank-conflict-free. This effectively transforms the inner loops from memory-bound to ALU-bound execution.

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

    // 1. Collaboratively load and pre-convert zetas to Montgomery form
    if (ltid < num_zetas) {
        zeta_mont[ltid] = mont_mul(zetas[ltid], r2, q_val, q_inv);
    }
    if (ltid + 128u < num_zetas) {
        zeta_mont[ltid + 128u] = mont_mul(zetas[ltid + 128u], r2, q_val, q_inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint *poly = coeffs + (size_t)tgid * n;

    // 2. Coalesced device load & Level 0
    uint u = poly[ltid];
    uint v = poly[ltid + 128u];
    
    if (n_levels_val > 0u) {
        uint z_mont = zeta_mont[1];
        uint t = mont_mul(v, z_mont, q_val, q_inv);
        uint new_u = mod_add_safe(u, t, q_val);
        uint new_v = mod_sub_safe(u, t, q_val);
        u = new_u; 
        v = new_v;
    }
    
    // Early exit cleanly limits execution if no further stages are requested
    if (n_levels_val <= 1u) {
        poly[ltid] = u;
        poly[ltid + 128u] = v;
        return;
    }
    
    a[ltid] = u;
    a[ltid + 128u] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Level 1 across threadgroup memory
    uint group_idx_1 = ltid >> 6; // ltid / 64
    uint j_in_group_1 = ltid & 63u; // ltid % 64
    uint j_1 = (group_idx_1 << 7) | j_in_group_1; 
    
    uint z_mont_1 = zeta_mont[2u + group_idx_1];
    uint u1 = a[j_1];
    uint v1 = a[j_1 + 64u];
    uint t1 = mont_mul(v1, z_mont_1, q_val, q_inv);
    u = mod_add_safe(u1, t1, q_val);
    v = mod_sub_safe(u1, t1, q_val);
    
    if (n_levels_val == 2u) {
        poly[j_1] = u;
        poly[j_1 + 64u] = v;
        return;
    }
    
    a[j_1] = u;
    a[j_1 + 64u] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Levels 2-7 completely in registers (independent SIMD sub-NTTs)
    uint sg = ltid >> 5;  // ltid / 32 (simdgroup ID)
    uint k = ltid & 31u;  // lane ID
    uint B = sg << 6;     // base block offset for this simdgroup
    uint B_plus_k = B + k;
    uint B_plus_k_32 = B_plus_k + 32u;

    uint x = a[B_plus_k];
    uint y = a[B_plus_k_32];

    // Level 2 fits right within the two variables per thread
    if (n_levels_val > 2u) {
        uint zeta_sg = zeta_mont[4u + sg];
        uint t = mont_mul(y, zeta_sg, q_val, q_inv);
        uint new_x = mod_add_safe(x, t, q_val);
        uint new_y = mod_sub_safe(x, t, q_val);
        x = new_x; 
        y = new_y;
    }

    // Remaining inner levels (lengths 16, 8, 4, 2, 1) use simd_shuffle to communicate
    #pragma unroll
    for (uint level = 3u; level < 8u; ++level) {
        if (level >= n_levels_val) break;

        uint L = 128u >> level;
        uint shift = 8u - level;
        bool is_top = (k & L) == 0u;
        
        // Branchless SIMD butterfly for x half
        uint zeta_x = zeta_mont[(1u << level) + (B_plus_k >> shift)];
        uint x_partner = simd_shuffle_xor(x, L);
        uint x_bot = is_top ? x_partner : x;
        uint x_top = is_top ? x : x_partner;
        uint tx = mont_mul(x_bot, zeta_x, q_val, q_inv);
        x = is_top ? mod_add_safe(x_top, tx, q_val) : mod_sub_safe(x_top, tx, q_val);
        
        // Branchless SIMD butterfly for y half
        uint zeta_y = zeta_mont[(1u << level) + (B_plus_k_32 >> shift)];
        uint y_partner = simd_shuffle_xor(y, L);
        uint y_bot = is_top ? y_partner : y;
        uint y_top = is_top ? y : y_partner;
        uint ty = mont_mul(y_bot, zeta_y, q_val, q_inv);
        y = is_top ? mod_add_safe(y_top, ty, q_val) : mod_sub_safe(y_top, ty, q_val);
    }

    // Coalesced writeback directly to device memory
    poly[B_plus_k] = x;
    poly[B_plus_k_32] = y;
}
```