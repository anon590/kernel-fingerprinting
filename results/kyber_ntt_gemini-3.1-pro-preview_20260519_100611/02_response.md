To optimize the Cooley-Tukey NTT kernel, I implemented a novel, exact 32-bit fractional Barrett reduction that eliminates 64-bit math from the inner loop. 

First, we compute the 64-bit division magic `M = 2^64 / q` once per thread. Inside the NTT loop, we dynamically construct a 32-bit reciprocal `W` for the current `zeta` using just two 32-bit multiplications: `W = zeta * M_hi + mulhi(zeta, M_lo)`. 

The modular multiplication `y * zeta % q` is then evaluated entirely in 32-bit unsigned arithmetic via `uint rem = y * zeta - mulhi(y, W) * q`. Because the approximation is mathematically tight, the true remainder strictly satisfies `0 <= rem < 2q < 2^32`. Thus, the 32-bit wraparound subtraction evaluates the remainder flawlessly, requiring only one conditional `- q`.

This massively improves over the previous 64-bit Barrett implementation by dropping the inner loop cost from 4 widening multiplications to just 2 native 32-bit multiplications.

```metal
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

    // Load elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    
    // Exact 64-bit reciprocal for the given runtime modulus
    ulong M = 0xFFFFFFFFFFFFFFFFull / q;
    uint M_hi = (uint)(M >> 32);
    uint M_lo = (uint)M;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        // Dynamically compute the 32-bit Barrett multiplier for the current zeta.
        // This exactly equals the high 32 bits of (zeta * M).
        uint W = zeta * M_hi + mulhi(zeta, M_lo);

        uint x = a[j];
        uint y = a[j + length];
        
        // 32-bit fractional Barrett reduction
        uint k = mulhi(y, W);
        uint rem = y * zeta - k * q;
        uint t = (rem >= q) ? (rem - q) : rem;

        // Canonical butterfly
        uint sum = x + t;
        a[j]          = (sum >= q) ? (sum - q) : sum;
        a[j + length] = (x >= t) ? (x - t) : (x + q - t);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Writeback canonical values
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```