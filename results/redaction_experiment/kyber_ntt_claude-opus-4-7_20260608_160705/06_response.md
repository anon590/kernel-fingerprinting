**Optimization:** The dominant cost at batch=256 is global memory traffic and barrier overhead. I'll: (1) cache each thread's two coefficients in registers between levels — for the first `n_levels - 1` levels the butterfly stays inside one thread (length >= 1 with the same thread owning (j, j+length) when length >= half-region). Actually, with the standard mapping each thread always owns its (j, j+length) pair within a level, but across levels the partner changes. So I keep values in registers when possible and only stage to threadgroup memory when partners must be exchanged. (2) Skip the initial threadgroup load — work directly from registers for level 0. (3) Use Barrett with a 32-bit reciprocal so it's robust for any q<2^16 (the multiplication zeta*y fits in 32 bits). (4) Reduce barriers: one barrier per level suffices since each thread writes the exact two slots it just read. (5) Pre-load zetas indices via simple addressing.

Actually re-reading: previously two barriers per level. We can drop the post-write barrier and keep only pre-read for next level → one barrier per level.

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

    // Load coefficients directly into registers; we own ltid and ltid+half_n.
    uint x = poly[ltid];
    uint y = poly[ltid + half_n];

    // Barrett reciprocal: floor(2^32 / q). q < 2^16 in practice but this works for q < 2^31.
    // We'll use floating-point reciprocal which is exact enough for q*y product < 2^32.
    float inv_q_f = 1.0f / (float)qv;

    uint length  = half_n;
    uint k_start = 1u;

    // ---------- Level 0 ----------
    // group_idx = ltid / half_n = 0 (since ltid < half_n)
    // j = ltid, partner = ltid + half_n, both already in registers.
    {
        uint zeta = zetas[k_start];  // zetas[1]
        uint prod = zeta * y;
        float pf  = (float)prod;
        uint qhat = (uint)(pf * inv_q_f);
        uint t    = prod - qhat * qv;
        if ((int)t < 0) t += qv;
        if (t >= qv)    t -= qv;

        uint sum_  = x + t;            if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;       if (diff_ >= qv) diff_ -= qv;

        // After level 0: x corresponds to slot ltid (in [0,half_n)),
        // y corresponds to slot ltid+half_n. The "length" for next level
        // is half_n/2, so a thread's new partner is in the OTHER half of
        // its current half. We must stage through threadgroup memory.
        a[ltid]          = sum_;
        a[ltid + half_n] = diff_;

        k_start <<= 1u;
        length  >>= 1u;
    }

    // For subsequent levels, each thread re-derives (j, j+length) and
    // loads/stores via threadgroup memory.
    for (uint level = 1u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint xv = a[j];
        uint yv = a[j + length];

        uint prod = zeta * yv;
        float pf  = (float)prod;
        uint qhat = (uint)(pf * inv_q_f);
        uint t    = prod - qhat * qv;
        if ((int)t < 0) t += qv;
        if (t >= qv)    t -= qv;

        uint sum_  = xv + t;          if (sum_  >= qv) sum_  -= qv;
        uint diff_ = xv + qv - t;     if (diff_ >= qv) diff_ -= qv;

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