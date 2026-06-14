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
    threadgroup uint shared_M[2];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    uint q_val        = q;
    uint n_levels_val = n_levels;

    // Thread 0 evaluates the expensive 64-bit division once per threadgroup
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q_val;
        shared_M[0] = (uint)M;
        shared_M[1] = (uint)(M >> 32);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint M_lo = shared_M[0];
    uint M_hi = shared_M[1];

    uint length = half_n;
    uint shift  = 31u - clz(half_n);
    uint level  = 0u;

    // Phase 1: Cross-SIMD levels (length >= 64)
    #pragma unroll 3
    for (uint i = 0; i < 3; ++i) {
        if (level >= n_levels_val || length < 64u) break;

        uint mask       = length - 1u;
        uint group_idx  = ltid >> shift;
        uint j          = ((ltid & ~mask) << 1u) | (ltid & mask);
        
        uint zeta = zetas[(1u << level) | group_idx];
        uint W    = zeta * M_hi + mulhi(zeta, M_lo);

        uint x = a[j];
        uint y = a[j + length];
        
        // Fractional Barrett reduction
        uint k_barrett = mulhi(y, W);
        uint r = y * zeta - k_barrett * q_val;
        r = select(r, r - q_val, r >= q_val);

        uint sum = x + r;
        a[j]          = select(sum, sum - q_val, sum >= q_val);
        
        uint diff = x - r;
        a[j + length] = select(diff + q_val, diff, x >= r);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        length >>= 1u;
        shift  -= 1u;
        level  += 1u;
    }

    // Phase 2: Intra-SIMD levels (length < 64)
    #pragma unroll 8
    for (uint i = 0; i < 8; ++i) {
        if (level >= n_levels_val) break;

        uint mask       = length - 1u;
        uint group_idx  = ltid >> shift;
        uint j          = ((ltid & ~mask) << 1u) | (ltid & mask);
        
        uint zeta = zetas[(1u << level) | group_idx];
        uint W    = zeta * M_hi + mulhi(zeta, M_lo);

        uint x = a[j];
        uint y = a[j + length];
        
        // Fractional Barrett reduction
        uint k_barrett = mulhi(y, W);
        uint r = y * zeta - k_barrett * q_val;
        r = select(r, r - q_val, r >= q_val);

        uint sum = x + r;
        a[j]          = select(sum, sum - q_val, sum >= q_val);
        
        uint diff = x - r;
        a[j + length] = select(diff + q_val, diff, x >= r);

        // Sub-64 lengths naturally partition completely into 32-thread SIMD groups
        simdgroup_barrier(mem_flags::mem_threadgroup);

        length >>= 1u;
        shift  -= 1u;
        level  += 1u;
    }

    // Ensure all butterfly writes from the final intra-SIMD iteration are globally visible
    // before SIMDs cooperatively writeback to device memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Writeback canonical values perfectly coalesced
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}