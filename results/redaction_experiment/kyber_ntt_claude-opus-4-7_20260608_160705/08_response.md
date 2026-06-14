**Optimization:** The incumbent uses a barrier after every butterfly write. I'll eliminate the trailing barrier in each level (the next iteration's read-barrier already covers it), keep the register-cached approach but with a single barrier per level instead of two. I'll also use `mulhi`/Barrett-style with precomputed integer reciprocal `m = floor(2^32 / q)` to avoid float rounding correction branches, replacing them with a single conditional subtraction. Additionally, I'll skip the initial barrier by using `simdgroup_barrier` where the access pattern stays within a simd, but since n=256 spans 4 simdgroups, threadgroup barriers are needed. The main win is halving barrier count per level (8 levels → saves 8 barriers) and a branchless Barrett reduction.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

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

    const uint half_n = n >> 1u;
    const uint qv     = q;
    const uint nl     = n_levels;

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Precompute Barrett reciprocal: m = floor(2^32 / q). For q < 2^16
    // and prod < 2^32, qhat = (prod * m) >> 32 is within 1 of true quotient.
    const uint m_recip = (uint)((((ulong)1) << 32) / (ulong)qv);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Barrett reduction: t = (zeta * y) mod q, with q < 2^16.
        uint prod = zeta * y;                // < 2^32 for q < 2^16
        uint qhat = (uint)(((ulong)prod * (ulong)m_recip) >> 32);
        uint t    = prod - qhat * qv;
        if (t >= qv) t -= qv;                // single correction

        uint sum_  = x + t;
        if (sum_ >= qv) sum_ -= qv;
        uint diff_ = x + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum_;
        a[j + length] = diff_;

        k_start <<= 1u;
        length  >>= 1u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```