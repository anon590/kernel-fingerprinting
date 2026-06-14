#include <metal_stdlib>
using namespace metal;

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid, uint M_32) 
{
    // Phase 1: Threadgroup memory levels (Lengths 128 and 64)
    #pragma unroll
    for (uint lvl = 0; lvl < 2; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7u - lvl;
            uint length = 1u << length_shift;
            uint group_idx = ltid >> length_shift;
            uint j_in_group = ltid & (length - 1u);
            uint j = (group_idx << (length_shift + 1u)) | j_in_group;
            uint zeta = zetas[(1u << lvl) + group_idx];
            
            uint x = a[j];
            uint y = a[j + length];
            
            uint t_val;
            if (USE_32BIT) {
                uint t = y * zeta;
                uint q_est = mulhi(t, M_32);
                uint t_mod = t - q_est * q_val;
                t_val = t_mod >= q_val ? t_mod - q_val : t_mod;
            } else {
                t_val = (uint)(((ulong)zeta * y) % q_val);
            }
            
            uint sum = x + t_val;
            a[j]          = sum >= q_val ? sum - q_val : sum;
            uint diff = x - t_val;
            a[j + length] = x >= t_val ? diff : diff + q_val;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint2 reg = a2[ltid];

    // Phase 2: Register levels (Lengths 32 down to 1)
    #pragma unroll
    for (uint lvl = 2; lvl < 8; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7u - lvl; 
            uint length = 1u << length_shift; 
            
            if (length == 1u) {
                uint group_idx = ltid;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint t_val;
                if (USE_32BIT) {
                    uint t = reg[1] * zeta;
                    uint q_est = mulhi(t, M_32);
                    uint t_mod = t - q_est * q_val;
                    t_val = t_mod >= q_val ? t_mod - q_val : t_mod;
                } else {
                    t_val = (uint)(((ulong)reg[1] * zeta) % q_val);
                }
                
                uint x0 = reg[0];
                uint sum = x0 + t_val;
                reg[0] = sum >= q_val ? sum - q_val : sum;
                uint diff = x0 - t_val;
                reg[1] = x0 >= t_val ? diff : diff + q_val;
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                // Vectorized shuffle and assignments
                uint2 y = simd_shuffle_xor(reg, k);
                
                uint2 a_j     = is_left ? reg : y;
                uint2 a_j_len = is_left ? y : reg;
                
                uint2 t_val;
                if (USE_32BIT) {
                    uint2 t_raw = a_j_len * zeta;
                    uint2 q_est = mulhi(t_raw, uint2(M_32));
                    uint2 t_mod = t_raw - q_est * q_val;
                    t_val.x = t_mod.x >= q_val ? t_mod.x - q_val : t_mod.x;
                    t_val.y = t_mod.y >= q_val ? t_mod.y - q_val : t_mod.y;
                } else {
                    t_val.x = (uint)(((ulong)a_j_len.x * zeta) % q_val);
                    t_val.y = (uint)(((ulong)a_j_len.y * zeta) % q_val);
                }
                
                uint2 sum = a_j + t_val;
                uint2 add_res;
                add_res.x = sum.x >= q_val ? sum.x - q_val : sum.x;
                add_res.y = sum.y >= q_val ? sum.y - q_val : sum.y;
                
                uint2 diff = a_j - t_val;
                uint2 sub_res;
                sub_res.x = a_j.x >= t_val.x ? diff.x : diff.x + q_val;
                sub_res.y = a_j.y >= t_val.y ? diff.y : diff.y + q_val;
                
                reg = is_left ? add_res : sub_res;
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
    uint n_levels_val = n_levels;

    // Fast 128-bit aligned vector accesses mapped linearly exactly for 256 uints (1024 bytes)
    device uint2 *poly2 = (device uint2 *)(coeffs + (size_t)tgid * 256u);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    // Phase 1: 100% coalesced uint2 load into fast threadgroup memory
    a2[ltid] = poly2[ltid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q_val <= 65535u) {
        uint M_32 = 0xFFFFFFFFu / q_val;
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, 0);
    }
}