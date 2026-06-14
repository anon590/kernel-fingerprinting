### Optimization Description

1. **Optimization**: Replaced the heavy 128-bit arithmetic from the incumbent's Barrett reduction with a fast **32-bit fractional Barrett multiplier**. The expensive 64-bit constant division `(2^64 - 1) / q` that caused performance regressions in the previous attempt is now hoisted and evaluated **exactly once** by thread 0, safely broadcast via threadgroup memory.
2. **Expected Improvement**: By computing the multiplier `W` on-the-fly using only 32-bit `mulhi` and `*` operations, we drastically reduce the ALU pressure per butterfly (saving multiple 64-bit multiplications and additions). The bounded error mathematically guarantees the exact remainder modulo `q` requires only a single conditional subtraction, yielding peak GPU execution throughput while ensuring bit-exact canonical outputs.

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
    threadgroup uint shared_M[2];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Thread 0 evaluates the expensive 64-bit division once per threadgroup
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        shared_M[0] = (uint)M;
        shared_M[1] = (uint)(M >> 32);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint M_lo = shared_M[0];
    uint M_hi = shared_M[1];

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        // Compute indices bitwise to avoid division/modulo inside the loop
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        
        // Compute fractional Barrett multiplier W for the given zeta
        uint zeta = zetas[k_start + group_idx];
        uint W    = zeta * M_hi + mulhi(zeta, M_lo);

        uint x = a[j];
        uint y = a[j + length];
        
        // 32-bit fractional Barrett reduction.
        // It is mathematically guaranteed that the error in the quotient is <= 1,
        // bounding the remainder strictly < 2q, completely eliminating 64-bit math.
        uint k_barrett = mulhi(y, W);
        uint r = y * zeta - k_barrett * q;
        r = select(r, r - q, r >= q);

        // Canonical butterfly operations
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
```