#include <metal_stdlib>
using namespace metal;

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid) 
{
    // Threadgroup levels (for length = 128 and 64)
    #pragma unroll
    for (uint lvl = 0; lvl < 2; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7 - lvl;
            uint length = 1u << length_shift;
            uint group_idx = ltid >> length_shift;
            uint j_in_group = ltid & (length - 1u);
            uint j = (group_idx << (length_shift + 1u)) | j_in_group;
            uint zeta = zetas[(1u << lvl) + group_idx];
            
            uint x = a[j];
            uint y = a[j + length];
            
            uint t_val;
            if (USE_32BIT) {
                t_val = (zeta * y) % q_val;
            } else {
                t_val = (uint)(((ulong)zeta * y) % q_val);
            }
            
            a[j]          = (x >= q_val - t_val) ? (x + t_val - q_val) : (x + t_val);
            a[j + length] = (x >= t_val) ? (x - t_val) : (x - t_val + q_val);
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint2 reg = a2[ltid];

    // Register levels (for length <= 32)
    #pragma unroll
    for (uint lvl = 2; lvl < 8; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7 - lvl; 
            uint length = 1u << length_shift; 
            
            if (length == 1) {
                uint group_idx = ltid;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint t_val;
                if (USE_32BIT) {
                    t_val = (reg[1] * zeta) % q_val;
                } else {
                    t_val = (uint)(((ulong)reg[1] * zeta) % q_val);
                }
                uint x0 = reg[0];
                
                reg[0] = (x0 >= q_val - t_val) ? (x0 + t_val - q_val) : (x0 + t_val);
                reg[1] = (x0 >= t_val) ? (x0 - t_val) : (x0 - t_val + q_val);
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint y0 = simd_shuffle_xor(reg[0], k);
                uint y1 = simd_shuffle_xor(reg[1], k);
                
                uint x0 = reg[0];
                uint x1 = reg[1];
                
                uint a_j0     = is_left ? x0 : y0;
                uint a_j_len0 = is_left ? y0 : x0;
                
                uint a_j1     = is_left ? x1 : y1;
                uint a_j_len1 = is_left ? y1 : x1;
                
                uint t0, t1;
                if (USE_32BIT) {
                    t0 = (a_j_len0 * zeta) % q_val;
                    t1 = (a_j_len1 * zeta) % q_val;
                } else {
                    t0 = (uint)(((ulong)a_j_len0 * zeta) % q_val);
                    t1 = (uint)(((ulong)a_j_len1 * zeta) % q_val);
                }
                
                uint add_res0 = (a_j0 >= q_val - t0) ? (a_j0 + t0 - q_val) : (a_j0 + t0);
                uint sub_res0 = (a_j0 >= t0) ? (a_j0 - t0) : (a_j0 - t0 + q_val);
                reg[0] = is_left ? add_res0 : sub_res0;
                
                uint add_res1 = (a_j1 >= q_val - t1) ? (a_j1 + t1 - q_val) : (a_j1 + t1);
                uint sub_res1 = (a_j1 >= t1) ? (a_j1 - t1) : (a_j1 - t1 + q_val);
                reg[1] = is_left ? add_res1 : sub_res1;
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

    device uint2 *poly2 = (device uint2 *)(coeffs + (size_t)tgid * 256u);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    // Phase 1: 100% coalesced uint2 load into fast threadgroup memory
    a2[ltid] = poly2[ltid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Fully unrolled constant-bounded loops.
    if (q_val <= 65535u) {
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid);
    }
}