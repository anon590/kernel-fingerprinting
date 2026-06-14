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

// Shared memory padding to completely eliminate bank conflicts
inline uint pad(uint x) {
    return x + (x >> 5);
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
    uint n_val = n;
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

    uint half_n = n_val >> 1u;
    uint num_zetas = 1u << n_levels_val;

    // Threadgroup allocations (size covers all cases n<=256)
    threadgroup uint a[264];
    threadgroup uint zeta_mont[256];

    // Collaboratively load and pre-convert zetas to Montgomery form
    for (uint i = ltid; i < num_zetas; i += half_n) {
        zeta_mont[i] = mont_mul(zetas[i], r2, q_val, q_inv);
    }

    // Coalesced loads from global memory directly into registers
    device uint *poly = coeffs + (size_t)tgid * n_val;
    uint u = poly[ltid];
    uint v = poly[ltid + half_n];
    
    // Store into bank-conflict-free padded shared memory
    a[pad(ltid)]          = u;
    a[pad(ltid + half_n)] = v;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_len = 31u - clz(half_n);
    uint k_start  = 1u;
    uint level    = 0u;
    
    // 1. Threadgroup stages for length >= 32
    while (level < n_levels_val && log2_len >= 5u) {
        uint L = 1u << log2_len;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (L - 1u);
        uint j          = (group_idx << (log2_len + 1u)) | j_in_group;
        
        uint z_mont = zeta_mont[k_start + group_idx];

        uint p_j  = pad(j);
        uint p_jL = pad(j + L);
        
        uint x = a[p_j];
        uint y = a[p_jL];
        
        uint t = mont_mul(y, z_mont, q_val, q_inv);

        a[p_j]  = mod_add_safe(x, t, q_val);
        a[p_jL] = mod_sub_safe(x, t, q_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        log2_len--;
        k_start <<= 1u;
        level++;
    }

    // Early exit if n_levels was extremely small
    if (level == n_levels_val) {
        poly[ltid]          = a[pad(ltid)];
        poly[ltid + half_n] = a[pad(ltid + half_n)];
        return;
    }

    // 2. Register/SIMD Shuffle stages for length <= 16
    // Load exactly 2 sub-problems (64 elements total) into each SIMD group of 32 threads
    uint sg = ltid >> 5;  
    uint k  = ltid & 31u; 
    
    uint c_u = 2u * sg;
    uint c_v = 2u * sg + 1u;

    u = a[pad(c_u * 32u + k)];
    v = a[pad(c_v * 32u + k)];

    while (level < n_levels_val) {
        uint L = 1u << log2_len;
        uint shift = log2_len + 1u;
        uint group_offset = k >> shift;
        
        uint z_idx_u = k_start + (c_u << (4u - log2_len)) + group_offset;
        uint z_idx_v = k_start + (c_v << (4u - log2_len)) + group_offset;
        
        uint z_u = zeta_mont[z_idx_u];
        uint z_v = zeta_mont[z_idx_v];
        
        bool is_top = (k & L) == 0u;
        
        // Butterfly for `u`
        uint u_partner = simd_shuffle_xor(u, L);
        uint u_bot = is_top ? u_partner : u;
        uint u_top = is_top ? u : u_partner;
        uint t_u = mont_mul(u_bot, z_u, q_val, q_inv);
        u = is_top ? mod_add_safe(u_top, t_u, q_val) : mod_sub_safe(u_top, t_u, q_val);
        
        // Butterfly for `v`
        uint v_partner = simd_shuffle_xor(v, L);
        uint v_bot = is_top ? v_partner : v;
        uint v_top = is_top ? v : v_partner;
        uint t_v = mont_mul(v_bot, z_v, q_val, q_inv);
        v = is_top ? mod_add_safe(v_top, t_v, q_val) : mod_sub_safe(v_top, t_v, q_val);
        
        log2_len--;
        k_start <<= 1u;
        level++;
    }

    // Perfectly coalesced writeback from registers directly to global memory
    poly[c_u * 32u + k] = u;
    poly[c_v * 32u + k] = v;
}