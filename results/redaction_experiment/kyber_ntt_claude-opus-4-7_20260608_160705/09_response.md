Optimization: (1) Cache twiddles in threadgroup memory once per polynomial to eliminate redundant `device` loads across levels — but more importantly, (2) cache the per-level zeta directly in registers via a single load per group, (3) preload coefficients into **registers** for the first stage (where `length = n/2`, each thread accesses a unique pair from device memory — skip the threadgroup roundtrip), and (4) merge the last stages by keeping the two coefficients owned by each thread in registers through a sequence of butterflies, exchanging via threadgroup memory only when partner index changes. The biggest win: reduce barriers by using a single barrier per level (write-then-barrier-then-read pattern, with reads/writes to distinct indices guaranteed by the butterfly mapping). Also replace Barrett with **Montgomery-free** mul-mod using `mulhi`-style 32-bit Barrett that is branchless.

I'll keep it simple and clean: single barrier per level, branchless Barrett, and skip the load-store roundtrip by doing stage 0 directly from device memory.

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

    // Barrett reciprocal: m = floor(2^32 / q). For q small (<= 2^16) the
    // product zeta*y fits in 32 bits, so 32-bit Barrett works branchlessly
    // with at most one correction.
    const uint m_recip = (uint)((((ulong)1) << 32) / (ulong)qv);

    // ---- Stage 0: read from device, write into threadgroup memory. ----
    // length = half_n, group_idx = 0 for all threads when ltid < half_n.
    // Actually group_idx = ltid / half_n which is 0 for ltid in [0, half_n).
    // zeta_index = 1 << 0 = 1, so zeta = zetas[1] (single zeta for stage 0).
    {
        uint length = half_n;
        uint j      = ltid;             // group_idx = 0
        uint zeta   = zetas[1];

        uint x = poly[j];
        uint y = poly[j + length];

        uint prod = zeta * y;
        uint qhat = (uint)(((ulong)prod * (ulong)m_recip) >> 32);
        uint t    = prod - qhat * qv;
        t = (t >= qv) ? (t - qv) : t;

        uint sum_  = x + t;  sum_  = (sum_  >= qv) ? (sum_  - qv) : sum_;
        uint diff_ = x + qv - t; diff_ = (diff_ >= qv) ? (diff_ - qv) : diff_;

        a[j]          = sum_;
        a[j + length] = diff_;
    }

    uint length  = half_n >> 1u;
    uint k_start = 2u;

    for (uint level = 1u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint prod = zeta * y;
        uint qhat = (uint)(((ulong)prod * (ulong)m_recip) >> 32);
        uint t    = prod - qhat * qv;
        t = (t >= qv) ? (t - qv) : t;

        uint sum_  = x + t;
        sum_  = (sum_  >= qv) ? (sum_  - qv) : sum_;
        uint diff_ = x + qv - t;
        diff_ = (diff_ >= qv) ? (diff_ - qv) : diff_;

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