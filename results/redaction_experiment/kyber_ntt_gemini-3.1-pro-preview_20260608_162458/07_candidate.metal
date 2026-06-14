#include <metal_stdlib>
using namespace metal;

template<bool USE_32BIT>
inline uint mul_mod(uint a, uint b, uint q_val, uint M_32) {
    if (USE_32BIT) {
        uint t = a * b;
        uint q_est = mulhi(t, M_32);
        uint t_mod = t - q_est * q_val;
        return select(t_mod, t_mod - q_val, t_mod >= q_val);
    } else {
        return (uint)(((ulong)a * b) % q_val);
    }
}

template<bool USE_32BIT>
inline uint2 mul_mod2(uint2 a, uint zeta, uint q_val, uint M_32) {
    if (USE_32BIT) {
        uint2 t = a * zeta;
        uint2 q_est = mulhi(t, uint2(M_32));
        uint2 q_vec = uint2(q_val);
        uint2 t_mod = t - q_est * q_vec;
        return select(t_mod, t_mod - q_vec, t_mod >= q_vec);
    } else {
        return uint2((uint)(((ulong)a.x * zeta) % q_val), (uint)(((ulong)a.y * zeta) % q_val));
    }
}

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid, uint M_32) 
{
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
            
            uint t_val = mul_mod<USE_32BIT>(y, zeta, q_val, M_32);
            
            uint sum = x + t_val;
            uint diff = x - t_val;
            
            if (USE_32BIT) {
                a[j]          = select(sum, sum - q_val, sum >= q_val);
                a[j + length] = select(diff + q_val, diff, x >= t_val);
            } else {
                a[j]          = select(sum, sum - q_val, x >= q_val - t_val);
                a[j + length] = select(diff + q_val, diff, x >= t_val);
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint2 reg = a2[ltid];

    #pragma unroll
    for (uint lvl = 2; lvl < 8; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7u - lvl; 
            uint length = 1u << length_shift; 
            
            if (length == 1u) {
                uint zeta = zetas[(1u << lvl) + ltid];
                
                uint t_val = mul_mod<USE_32BIT>(reg.y, zeta, q_val, M_32);
                uint x0 = reg.x;
                
                uint sum = x0 + t_val;
                uint diff = x0 - t_val;
                
                if (USE_32BIT) {
                    reg.x = select(sum, sum - q_val, sum >= q_val);
                    reg.y = select(diff + q_val, diff, x0 >= t_val);
                } else {
                    reg.x = select(sum, sum - q_val, x0 >= q_val - t_val);
                    reg.y = select(diff + q_val, diff, x0 >= t_val);
                }
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint2 y_vec;
                y_vec.x = simd_shuffle_xor(reg.x, k);
                y_vec.y = simd_shuffle_xor(reg.y, k);
                
                bool2 is_left2 = bool2(is_left);
                uint2 a_j     = select(y_vec, reg, is_left2);
                uint2 a_j_len = select(reg, y_vec, is_left2);
                
                uint2 t_vec = mul_mod2<USE_32BIT>(a_j_len, zeta, q_val, M_32);
                
                uint2 q_vec = uint2(q_val);
                uint2 sum = a_j + t_vec;
                uint2 diff = a_j - t_vec;
                
                uint2 add_res, sub_res;
                if (USE_32BIT) {
                    add_res = select(sum, sum - q_vec, sum >= q_vec);
                    sub_res = select(diff + q_vec, diff, a_j >= t_vec);
                } else {
                    add_res = select(sum, sum - q_vec, a_j >= q_vec - t_vec);
                    sub_res = select(diff + q_vec, diff, a_j >= t_vec);
                }
                
                reg = select(sub_res, add_res, is_left2);
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

    bool use_32bit = (q_val <= 65535u);
    uint M_32 = use_32bit ? (0xFFFFFFFFu / q_val) : 0u;

    if (use_32bit) {
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32);
    }
}