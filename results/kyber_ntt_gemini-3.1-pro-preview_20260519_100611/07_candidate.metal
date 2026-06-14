#include <metal_stdlib>
using namespace metal;

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

    // Fast integer division precomputation
    ulong M = 0xFFFFFFFFFFFFFFFFull / q;
    uint M_lo = (uint)M;
    uint M_hi = (uint)(M >> 32);

    threadgroup uint shared_zetas[256];
    threadgroup uint shared_W[256];
    threadgroup uint a[256];

    uint half_n = n >> 1u;
    
    // Cooperatively load zetas and precompute fractional Barrett multipliers W
    uint num_zetas = 1u << n_levels;
    for (uint i = ltid; i < num_zetas; i += half_n) {
        uint z = zetas[i];
        shared_zetas[i] = z;
        shared_W[i]     = z * M_hi + mulhi(z, M_lo);
    }

    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint shift   = 31u - clz(half_n);
    uint k_start = 1u;
    uint level   = 0u;

    if (n >= 64u) {
        // Phase 1: Shared memory NTT for length >= 32
        // Avoids cross-SIMD shuffles, zero bank conflicts for length >= 32
        while (length >= 32u && level < n_levels) {
            uint group_idx  = ltid >> shift;
            uint j_in_group = ltid & (length - 1u);
            uint j          = (group_idx << (shift + 1u)) | j_in_group;
            
            uint zeta = shared_zetas[k_start + group_idx];
            uint W    = shared_W[k_start + group_idx];

            uint x = a[j];
            uint y = a[j + length];
            
            uint k_barrett = mulhi(y, W);
            uint r = y * zeta - k_barrett * q;
            r = select(r, r - q, r >= q);

            uint sum = x + r;
            a[j]          = select(sum, sum - q, sum >= q);
            
            uint diff = x - r;
            a[j + length] = select(diff + q, diff, x >= r);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            k_start <<= 1u;
            length  >>= 1u;
            shift   -= 1u;
            level++;
        }

        // Phase 2: Register-based NTT for length <= 16
        // Completely eliminates threadgroup barriers and bank conflicts
        uint lane_id = ltid & 31u;
        uint simd_id = ltid >> 5u;
        
        uint base = simd_id * 64u;
        uint j_u = base + lane_id;
        uint j_v = base + 32u + lane_id;
        
        uint u = a[j_u];
        uint v = a[j_v];
        
        #pragma unroll
        for (uint iter = 0; iter < 5u; ++iter) {
            if (level >= n_levels) break;
            
            uint L = n >> (level + 1u);
            uint shift_2L = 31u - clz(n) - level;
            
            bool is_right = (j_u & L) != 0u;
            
            // Process element u
            uint other_u = simd_shuffle_xor(u, L);
            uint x_u = is_right ? other_u : u;
            uint y_u = is_right ? u : other_u;
            
            uint zeta_idx_u  = (1u << level) + (j_u >> shift_2L);
            uint zeta_u      = shared_zetas[zeta_idx_u];
            uint W_u         = shared_W[zeta_idx_u];
            
            uint k_barrett_u = mulhi(y_u, W_u);
            uint r_u         = y_u * zeta_u - k_barrett_u * q;
            r_u              = select(r_u, r_u - q, r_u >= q);
            
            uint sum_u   = x_u + r_u;
            uint x_new_u = select(sum_u, sum_u - q, sum_u >= q);
            uint diff_u  = x_u - r_u;
            uint y_new_u = select(diff_u + q, diff_u, x_u >= r_u);
            
            u = is_right ? y_new_u : x_new_u;

            // Process element v (reuses bounds math from u)
            uint other_v = simd_shuffle_xor(v, L);
            uint x_v = is_right ? other_v : v;
            uint y_v = is_right ? v : other_v;
            
            uint zeta_idx_v  = zeta_idx_u + (32u >> shift_2L);
            uint zeta_v      = shared_zetas[zeta_idx_v];
            uint W_v         = shared_W[zeta_idx_v];
            
            uint k_barrett_v = mulhi(y_v, W_v);
            uint r_v         = y_v * zeta_v - k_barrett_v * q;
            r_v              = select(r_v, r_v - q, r_v >= q);
            
            uint sum_v   = x_v + r_v;
            uint x_new_v = select(sum_v, sum_v - q, sum_v >= q);
            uint diff_v  = x_v - r_v;
            uint y_new_v = select(diff_v + q, diff_v, x_v >= r_v);
            
            v = is_right ? y_new_v : x_new_v;
            
            level++;
        }

        // Direct writeback from registers (perfectly coalesced)
        poly[j_u] = u;
        poly[j_v] = v;

    } else {
        // Fallback Phase 1 for extremely small n (n < 64)
        while (level < n_levels) {
            uint group_idx  = ltid >> shift;
            uint j_in_group = ltid & (length - 1u);
            uint j          = (group_idx << (shift + 1u)) | j_in_group;
            
            uint zeta = shared_zetas[k_start + group_idx];
            uint W    = shared_W[k_start + group_idx];

            uint x = a[j];
            uint y = a[j + length];
            
            uint k_barrett = mulhi(y, W);
            uint r = y * zeta - k_barrett * q;
            r = select(r, r - q, r >= q);

            uint sum = x + r;
            a[j]          = select(sum, sum - q, sum >= q);
            
            uint diff = x - r;
            a[j + length] = select(diff + q, diff, x >= r);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            k_start <<= 1u;
            length  >>= 1u;
            shift   -= 1u;
            level++;
        }
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
    }
}