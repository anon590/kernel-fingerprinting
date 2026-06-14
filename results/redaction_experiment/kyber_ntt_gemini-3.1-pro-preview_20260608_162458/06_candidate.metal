#include <metal_stdlib>
using namespace metal;

template<bool USE_32BIT>
inline uint mod_mul(uint y, uint zeta, uint q_val, uint M_32, uint M_64_lo, uint M_64_hi) {
    if (USE_32BIT) {
        uint prod = y * zeta;
        uint Q_est = mulhi(prod, M_32);
        uint rem = prod - Q_est * q_val;
        return (rem >= q_val) ? rem - q_val : rem;
    } else {
        ulong prod = (ulong)y * zeta;
        uint x0 = (uint)prod;
        uint x1 = (uint)(prod >> 32);
        
        ulong p00 = (ulong)x0 * M_64_lo;
        ulong p01 = (ulong)x0 * M_64_hi;
        ulong p10 = (ulong)x1 * M_64_lo;
        ulong p11 = (ulong)x1 * M_64_hi;
        
        ulong mid = p01 + (uint)(p00 >> 32);
        ulong mid2 = p10 + (uint)mid;
        
        ulong Q_est = p11 + (mid >> 32) + (mid2 >> 32);
        ulong rem = prod - Q_est * q_val;
        return (rem >= q_val) ? (uint)(rem - q_val) : (uint)rem;
    }
}

template<bool USE_32BIT>
inline uint2 mod_mul_vec(uint2 y, uint zeta, uint q_val, uint M_32, uint M_64_lo, uint M_64_hi) {
    if (USE_32BIT) {
        uint2 prod = y * zeta;
        uint2 Q_est = mulhi(prod, uint2(M_32));
        uint2 q_vec = uint2(q_val);
        uint2 rem = prod - Q_est * q_vec;
        return select(rem, rem - q_vec, rem >= q_vec);
    } else {
        uint t0 = mod_mul<false>(y.x, zeta, q_val, M_32, M_64_lo, M_64_hi);
        uint t1 = mod_mul<false>(y.y, zeta, q_val, M_32, M_64_lo, M_64_hi);
        return uint2(t0, t1);
    }
}

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid,
                        uint M_32, uint M_64_lo, uint M_64_hi) 
{
    // Threadgroup levels (for lengths 128 and 64)
    if (0 < n_levels_val) {
        uint length = 128u;
        uint j = ltid;
        uint zeta = zetas[1];
        
        uint x = a[j];
        uint y = a[j + length];
        
        uint t_val = mod_mul<USE_32BIT>(y, zeta, q_val, M_32, M_64_lo, M_64_hi);
        
        a[j]          = (x >= q_val - t_val) ? (x + t_val - q_val) : (x + t_val);
        a[j + length] = (x >= t_val) ? (x - t_val) : (x - t_val + q_val);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (1 < n_levels_val) {
        uint length = 64u;
        uint group_idx = ltid >> 6u;
        uint j_in_group = ltid & 63u;
        uint j = (group_idx << 7u) | j_in_group;
        uint zeta = zetas[2u + group_idx];
        
        uint x = a[j];
        uint y = a[j + length];
        
        uint t_val = mod_mul<USE_32BIT>(y, zeta, q_val, M_32, M_64_lo, M_64_hi);
        
        a[j]          = (x >= q_val - t_val) ? (x + t_val - q_val) : (x + t_val);
        a[j + length] = (x >= t_val) ? (x - t_val) : (x - t_val + q_val);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint2 reg = a2[ltid];

    // Register levels (for length <= 32)
    #pragma unroll
    for (uint lvl = 2; lvl < 8; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7u - lvl; 
            uint length = 1u << length_shift; 
            
            if (length == 1u) {
                uint zeta = zetas[128u + ltid];
                uint t_val = mod_mul<USE_32BIT>(reg[1], zeta, q_val, M_32, M_64_lo, M_64_hi);
                uint x0 = reg[0];
                
                reg[0] = (x0 >= q_val - t_val) ? (x0 + t_val - q_val) : (x0 + t_val);
                reg[1] = (x0 >= t_val) ? (x0 - t_val) : (x0 - t_val + q_val);
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint2 y_vec;
                y_vec.x = simd_shuffle_xor(reg[0], k);
                y_vec.y = simd_shuffle_xor(reg[1], k);
                
                uint2 a_j;
                a_j.x = is_left ? reg[0] : y_vec.x;
                a_j.y = is_left ? reg[1] : y_vec.y;
                
                uint2 a_j_len;
                a_j_len.x = is_left ? y_vec.x : reg[0];
                a_j_len.y = is_left ? y_vec.y : reg[1];
                
                uint2 t = mod_mul_vec<USE_32BIT>(a_j_len, zeta, q_val, M_32, M_64_lo, M_64_hi);
                
                uint2 q_vec = uint2(q_val);
                uint2 sum = a_j + t;
                uint2 diff = a_j - t;
                
                uint2 add_res = select(sum, sum - q_vec, a_j >= q_vec - t);
                uint2 sub_res = select(diff + q_vec, diff, a_j >= t);
                
                reg[0] = is_left ? add_res.x : sub_res.x;
                reg[1] = is_left ? add_res.y : sub_res.y;
            }
        }
    }

    poly2[ltid] = reg;
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

    threadgroup uint a[256];
    
    uint q_val = q;
    uint n_val = n;
    uint n_levels_val = n_levels;

    device uint2 *poly2 = (device uint2 *)(coeffs + (size_t)tgid * n_val);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    // Phase 1: 100% coalesced uint2 load into fast threadgroup memory
    a2[ltid] = poly2[ltid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute fast Barrett multiplier for reduction once per thread
    bool use_32bit = (q_val <= 65535u);
    uint M_32 = 0;
    uint M_64_lo = 0;
    uint M_64_hi = 0;

    if (use_32bit) {
        M_32 = 0xFFFFFFFFu / q_val;
    } else {
        ulong M_64 = 0xFFFFFFFFFFFFFFFFull / q_val;
        M_64_lo = (uint)M_64;
        M_64_hi = (uint)(M_64 >> 32u);
    }

    // Phase 2: Compute NTT
    if (use_32bit) {
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32, M_64_lo, M_64_hi);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32, M_64_lo, M_64_hi);
    }
}