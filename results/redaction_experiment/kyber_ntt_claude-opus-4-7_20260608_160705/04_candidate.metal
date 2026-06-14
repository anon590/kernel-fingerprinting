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

    uint half_n = n >> 1u;
    uint qv     = q;
    uint nl     = n_levels;

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load: each of the half_n threads loads two coefficients.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Precompute float reciprocal of q for Barrett-style reduction.
    // q < 2^23 (Kyber uses 3329), zeta,y < q so product < 2^46, well within
    // float64-ish range but we exploit that q fits in ~12 bits so product
    // fits in ~24 bits which is exact in float. Use double for safety on
    // larger q values that still fit the spec (q < 2^32; but pairs (zeta,y)
    // can yield products up to 2^64 in general). Here q is small so float
    // works, but we use float with a safety correction loop.
    float inv_q_f = 1.0f / (float)qv;

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nl; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t = (zeta * y) mod q via Barrett-style float reciprocal.
        // zeta, y < q <= ~3329 so prod < ~1.1e7, exact in float.
        uint prod_lo = zeta * y;                       // exact for small q
        float pf = (float)prod_lo;
        uint qhat = (uint)(pf * inv_q_f);
        uint t = prod_lo - qhat * qv;
        // Correct for rounding: t may be in [-q, 2q).
        if ((int)t < 0) t += qv;
        if (t >= qv)    t -= qv;

        uint sum_  = x + t;
        if (sum_ >= qv) sum_ -= qv;
        uint diff_ = x + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum_;
        a[j + length] = diff_;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}