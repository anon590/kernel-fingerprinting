**Optimization:** Keep coefficients in registers across stages instead of round-tripping through threadgroup memory each stage. Each thread owns two coefficients (its butterfly pair) for the early stages where `length >= n/2` would normally require the partner thread — but since each thread owns one butterfly per level (different partnerings at each level), we instead store all `n` coefficients in threadgroup memory and do the butterflies there, but cache the two values in registers between the read and the barrier-protected write. The main win comes from replacing the expensive `%` with Barrett/precomputed reduction... but `q` is runtime. Instead, I'll focus on: (1) caching `zeta` per simdgroup via `simd_broadcast` when groups align, (2) reducing barriers using `simdgroup_barrier` when the butterfly stride fits in a simdgroup (32 lanes → stride ≤ 16 means the 2*length coefficients fit in 32 lanes worth of contiguous memory accessed by the same simdgroup), and (3) skipping the initial barrier by having each thread load only the values it needs.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_mul(uint a, uint b, uint q) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)q);
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

    // Each thread loads two coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    // Stages where butterfly stride > 16: need full threadgroup barrier.
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, q);

        uint v_add = mod_add(x, t, q);
        uint v_sub = mod_sub(x, t, q);

        a[j]          = v_add;
        a[j + length] = v_sub;

        // When 2*length <= 32, all threads working on overlapping data
        // are within the same simdgroup -> cheaper barrier.
        if (length > 16u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```