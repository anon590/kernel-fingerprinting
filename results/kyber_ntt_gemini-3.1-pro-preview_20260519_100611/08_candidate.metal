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

    threadgroup uint a[256];
    threadgroup uint shared_zetas[256];
    threadgroup uint shared_W[256];
    threadgroup uint shared_M[2];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Compute expensive 64-bit reciprocal for Barrett reduction once
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        shared_M[0] = (uint)M;
        shared_M[1] = (uint)(M >> 32);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint M_lo = shared_M[0];
    uint M_hi = shared_M[1];
    uint num_zetas = 1u << n_levels;

    // Cooperatively precompute W multipliers for all zetas
    for (uint i = ltid; i < num_zetas; i += half_n) {
        uint z = zetas[i];
        shared_zetas[i] = z;
        shared_W[i]     = z * M_hi + mulhi(z, M_lo);
    }

    // Synchronize after precomputations
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint shift   = ctz(half_n);
    uint length  = half_n;
    uint k_start = 1u;
    
    #pragma unroll 8
    for (uint level = 0u; level < 8u; ++level) {
        if (level >= n_levels) break;
        
        // Fast bitwise index logic
        uint mask      = length - 1u;
        uint j         = ((ltid & ~mask) << 1u) | (ltid & mask);
        uint group_idx = ltid >> shift;
        
        uint zeta_idx = k_start + group_idx;
        uint zeta     = shared_zetas[zeta_idx];
        uint W        = shared_W[zeta_idx];

        uint j_len = j + length;
        uint x = a[j];
        uint y = a[j_len];
        
        // 32-bit fractional Barrett reduction
        uint k_barrett = mulhi(y, W);
        uint r = y * zeta - k_barrett * q;
        r = select(r, r - q, r >= q);

        // Canonical butterfly
        uint sum = x + r;
        a[j]     = select(sum, sum - q, sum >= q);
        
        uint diff = x - r;
        a[j_len]  = select(diff + q, diff, x >= r);

        // Threadgroup barrier is strictly only needed when data crosses SIMD boundaries
        if (length > 32u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Ensure all butterfly writes from final iteration are visible before writeback
    threadgroup_barrier(mem_flags::mem_threadgroup);

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}