### Optimization Description

This kernel replaces the incumbent's full 64-bit exact Barrett reduction with an optimized **fractional Barrett reduction**. Using the globally computed 64-bit reciprocal `M = 2^64 / q`, we precompute exact 32-bit Barrett multipliers `W = (zeta * M) >> 32` for all twiddle factors in threadgroup memory during the initial load phase. 

Inside the inner Cooley-Tukey loop, computing the quotient `k` now requires only a single 32-bit `mulhi(y, W)`, bypassing the need for a full 64x64 multiply. We then compute the remainder `y * zeta - k * q` exactly in 64-bit and conditionally reduce it. This cuts the heavy multiplication (`umull` / `mulhi`) instruction count in the inner loop from **9** (incumbent) down to **3** per butterfly, drastically alleviating ALU pressure while maintaining mathematically bulletproof bit-exactness for any runtime $q < 2^{32}$.

### Metal Source Code

```metal
#include <metal_stdlib>
using namespace metal;

struct ZetaW {
    uint zeta;
    uint W;
};

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
    threadgroup ZetaW tg_ZW[256];
    threadgroup uint tg_M_lo;
    threadgroup uint tg_M_hi;

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Thread 0 computes the 64-bit reciprocal for the runtime modulus
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        tg_M_lo = (uint)M;
        tg_M_hi = (uint)(M >> 32);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint M_lo = tg_M_lo;
    uint M_hi = tg_M_hi;

    // Precompute Barrett fractional multipliers (W) for all utilized zetas
    uint num_zetas = 1u << n_levels;
    for (uint i = ltid; i < num_zetas; i += half_n) {
        uint z = zetas[i];
        tg_ZW[i].zeta = z;
        // Exactly computes the upper 32 bits of (z * M)
        tg_ZW[i].W    = z * M_hi + mulhi(z, M_lo);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        
        ZetaW zw  = tg_ZW[k_start + group_idx];
        uint zeta = zw.zeta;
        uint W    = zw.W;

        uint x = a[j];
        uint y = a[j + length];
        
        // Fast fractional Barrett reduction
        uint k = mulhi(y, W);
        ulong rem64 = (ulong)y * zeta - (ulong)k * q;
        
        // The fractional error bounds guarantee the remainder is strictly < 3q.
        // Two conditional subtractions reduce it exactly to [0, q - 1].
        rem64 = select(rem64, rem64 - q, rem64 >= q);
        rem64 = select(rem64, rem64 - q, rem64 >= q);
        uint rem = (uint)rem64;

        // Canonical butterfly
        uint sum = x + rem;
        a[j]          = select(sum, sum - q, sum >= q || sum < x);
        a[j + length] = select(q - (rem - x), x - rem, x >= rem);

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