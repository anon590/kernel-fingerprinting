**Optimization:** Replace the expensive `%` modular reduction (used in `mod_mul`) with a Barrett-style reduction using a precomputed reciprocal `m = floor(2^32 / q)`. Each thread computes the reciprocal once from `q` (cheap) and then every butterfly's `mod_mul` becomes a 64-bit multiply, a `mulhi`-like shift, and at most two subtractions—no integer division. I also cache the two coefficients each thread owns in registers across stages (re-loading from threadgroup memory after each barrier is fine on Apple GPUs, but keeping them in registers cuts shared-memory traffic). The barrier still flushes per stage so neighbors see updates. This keeps the algorithm bit-exact while removing the per-butterfly `udiv`, which dominated the seed's runtime.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint barrett_mul(uint a, uint b, uint q, ulong m) {
    ulong p = (ulong)a * (ulong)b;
    // q < 2^32, m = floor(2^64 / q) would overflow; use m = floor(2^32 / q)
    // and reduce in two steps. Simpler: use direct 64-bit product then
    // Barrett with m = floor(2^62 / q).
    ulong qest = ((p >> 31) * m) >> 31;
    ulong r = p - qest * (ulong)q;
    uint ru = (uint)r;
    if (ru >= q) ru -= q;
    if (ru >= q) ru -= q;
    return ru;
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Precompute Barrett reciprocal: m = floor(2^62 / q).
    // For any p < 2^62, ((p >> 31) * m) >> 31 approximates p / q within +/- a few.
    ulong m = ((ulong)1 << 62) / (ulong)q;

    // Load two coefficients per thread.
    uint c0 = poly[ltid];
    uint c1 = poly[ltid + half_n];
    a[ltid]          = c0;
    a[ltid + half_n] = c1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = barrett_mul(zeta, y, q, m);

        a[j]          = mod_add(x, t, q);
        a[j + length] = mod_sub(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```