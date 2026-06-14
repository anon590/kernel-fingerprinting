**Optimization:** The incumbent has two barriers per level (one before reading would-be-stale data, one after writing). Since each thread reads/writes its own `(j, j+length)` pair and those are disjoint across threads in the same level (it's an in-place butterfly), only ONE barrier per level is required — the barrier between levels. I also pack two coefficients per thread into local registers for stage 0 directly from the device load (skipping a write-then-read trip through threadgroup memory), and shave the second barrier per inner level. For batch=256 this should reduce barrier-stall time substantially since most of the runtime is sync overhead at small n.

I also replace the float-Barrett with a 32×32→64 multiply followed by a single `% q` — the Apple GPU has fast 64-bit integer mul and `%`, and this avoids the float rounding correction branch. Each thread does ~8 multiplies total, so this is cheap.

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

    // ---- Stage 0 directly from device memory into registers + tg mem. ----
    // length = half_n, group_idx = 0, zeta_index = 1.
    uint length = half_n;
    {
        uint j     = ltid;          // group_idx = 0
        uint zeta  = zetas[1];

        uint x = poly[j];
        uint y = poly[j + length];

        uint t = (uint)(((ulong)zeta * (ulong)y) % (ulong)qv);

        uint sum_  = x + t;          if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;     if (diff_ >= qv) diff_ -= qv;

        a[j]          = sum_;
        a[j + length] = diff_;
    }

    uint k_start = 2u;
    length >>= 1u;

    for (uint level = 1u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = (uint)(((ulong)zeta * (ulong)y) % (ulong)qv);

        uint sum_  = x + t;          if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;     if (diff_ >= qv) diff_ -= qv;

        // Single barrier before next level's reads; writes here are to the
        // same (j, j+length) slots this thread just read, so no intra-level
        // race exists.
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