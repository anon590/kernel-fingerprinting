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

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Compute the 64-bit reciprocal for the runtime modulus.
    // The Metal compiler scalarizes this since q is constant across the grid.
    ulong M = 0xFFFFFFFFFFFFFFFFull / q;
    uint M_lo = (uint)M;
    uint M_hi = (uint)(M >> 32);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        // Compute indices using bitwise operations
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        
        // Load zeta from device cache and compute its fractional Barrett multiplier (W)
        // using purely 32-bit math. This eliminates threadgroup memory reads for W.
        uint zeta = zetas[k_start + group_idx];
        uint W    = zeta * M_hi + mulhi(zeta, M_lo);

        uint x = a[j];
        uint y = a[j + length];
        
        // Fast fractional Barrett reduction avoiding 64-bit math entirely.
        // Even if y * zeta overflows 32-bits, modulo 2^32 arithmetic natively yields the exact remainder.
        uint k = mulhi(y, W);
        uint r = y * zeta - k * q;
        
        // The fractional error bounds mathematically guarantee the remainder is strictly < 3q
        r = select(r, r - q, r >= q);
        r = select(r, r - q, r >= q);

        // Canonical butterfly
        uint sum = x + r;
        a[j]          = select(sum, sum - q, sum >= q);
        
        uint diff = x - r;
        a[j + length] = select(diff + q, diff, x >= r);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Writeback canonical values perfectly coalesced
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}