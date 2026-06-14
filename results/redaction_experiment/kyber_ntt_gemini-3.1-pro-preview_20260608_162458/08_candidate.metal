#include <metal_stdlib>
using namespace metal;

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid, uint M) 
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
                uint q_est = mulhi(t, M);
                uint t_mod = t - q_est * q_val;
                t_val = select(t_mod, t_mod - q_val, t_mod >= q_val);
            } else {
                t_val = (uint)(((ulong)y * zeta) % q_val);
            }
            
            uint sum = x + t_val;
            a[j] = select(sum, sum - q_val, sum >= q_val);
            uint diff = x + q_val - t_val;
            a[j + length] = select(diff, diff - q_val, diff >= q_val);
            
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
                uint zeta = zetas[(1u << lvl) + ltid];
                
                uint t_val;
                if (USE_32BIT) {
                    uint t = reg[1] * zeta;
                    uint q_est = mulhi(t, M);
                    uint t_mod = t - q_est * q_val;
                    t_val = select(t_mod, t_mod - q_val, t_mod >= q_val);
                } else {
                    t_val = (uint)(((ulong)reg[1] * zeta) % q_val);
                }
                
                uint x0 = reg[0];
                uint sum = x0 + t_val;
                reg[0] = select(sum, sum - q_val, sum >= q_val);
                uint diff = x0 + q_val - t_val;
                reg[1] = select(diff, diff - q_val, diff >= q_val);
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                // Step 1: Exchange inputs so each thread holds exact pair for 1 butterfly
                uint send_val = is_left ? reg[1] : reg[0];
                uint recv_val = simd_shuffle_xor(send_val, k);
                
                uint u = is_left ? reg[0] : recv_val;
                uint v = is_left ? recv_val : reg[1];
                
                // Step 2: Compute exactly ONE butterfly per thread
                uint t_val;
                if (USE_32BIT) {
                    uint t = v * zeta;
                    uint q_est = mulhi(t, M);
                    uint t_mod = t - q_est * q_val;
                    t_val = select(t_mod, t_mod - q_val, t_mod >= q_val);
                } else {
                    t_val = (uint)(((ulong)v * zeta) % q_val);
                }
                
                uint sum = u + t_val;
                sum = select(sum, sum - q_val, sum >= q_val);
                uint diff = u + q_val - t_val;
                diff = select(diff, diff - q_val, diff >= q_val);
                
                // Step 3: Route sum to the left thread and diff to the right thread
                uint send_res = is_left ? diff : sum;
                uint recv_res = simd_shuffle_xor(send_res, k);
                
                reg[0] = is_left ? sum : recv_res;
                reg[1] = is_left ? recv_res : diff;
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

    bool use_32bit = (q_val <= 65535u);
    uint M_32 = use_32bit ? (0xFFFFFFFFu / q_val) : 0u;

    if (use_32bit) {
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32);
    }
}