#include <metal_stdlib>
using namespace metal;

template<bool USE_32>
inline void process_levels(threadgroup uint* a, device const uint* zetas, 
                           uint q_val, uint n_val, uint n_levels_val, uint ltid, 
                           uint M_32, uint M_64_lo, uint M_64_hi) 
{
    #pragma unroll
    for (uint lvl = 0; lvl < 8; ++lvl) {
        if (lvl >= n_levels_val) break;
        
        uint length = n_val >> (lvl + 1);
        uint length_shift = ctz(length);
        uint group_idx = ltid >> length_shift;
        uint j_in_group = ltid & (length - 1u);
        uint j = (group_idx << (length_shift + 1u)) | j_in_group;
        uint zeta = zetas[(1u << lvl) + group_idx];
        
        uint x = a[j];
        uint y = a[j + length];
        
        uint t_val;
        if (USE_32) {
            uint prod = y * zeta;
            uint Q_est = mulhi(prod, M_32);
            uint rem = prod - Q_est * q_val;
            t_val = (rem >= q_val) ? (rem - q_val) : rem;
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
            t_val = (rem >= q_val) ? (uint)(rem - q_val) : (uint)rem;
        }
        
        a[j]          = (x >= q_val - t_val) ? (x + t_val - q_val) : (x + t_val);
        a[j + length] = (x >= t_val) ? (x - t_val) : (x - t_val + q_val);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
    
    uint n_val = n;
    uint q_val = q;
    uint n_levels_val = n_levels;

    threadgroup uint shared_M_32;
    threadgroup ulong shared_M_64;
    threadgroup bool shared_use_32;
    
    if (ltid == 0) {
        bool use_32 = (q_val <= 65535u);
        shared_use_32 = use_32;
        if (use_32) {
            shared_M_32 = 0xFFFFFFFFu / q_val;
        } else {
            shared_M_64 = 0xFFFFFFFFFFFFFFFFull / q_val;
        }
    }

    device uint2 *poly2 = (device uint2 *)(coeffs + tgid * n_val);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    a2[ltid] = poly2[ltid];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    bool use_32 = shared_use_32;
    uint M_32 = shared_M_32;
    ulong M_64 = shared_M_64;
    uint M_64_lo = (uint)M_64;
    uint M_64_hi = (uint)(M_64 >> 32);

    if (use_32) {
        process_levels<true>(a, zetas, q_val, n_val, n_levels_val, ltid, M_32, M_64_lo, M_64_hi);
    } else {
        process_levels<false>(a, zetas, q_val, n_val, n_levels_val, ltid, M_32, M_64_lo, M_64_hi);
    }

    poly2[ltid] = a2[ltid];
}