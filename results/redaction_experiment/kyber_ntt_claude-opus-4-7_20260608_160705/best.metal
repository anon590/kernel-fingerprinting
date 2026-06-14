#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;
constexpr constant uint Z_MAX = 256u;

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
    threadgroup uint zt[Z_MAX];
    threadgroup uint qbarrett;  // floor(2^32 / q)

    const uint half_n = n >> 1u;
    const uint qv     = q;
    const uint nl     = n_levels;
    const uint zlen   = 1u << nl;

    // Precompute Barrett magic: m = floor(2^32 / q).
    if (ltid == 0u) {
        qbarrett = (uint)(0xFFFFFFFFu / qv);
        // Adjust: we want floor(2^32 / q). 0xFFFFFFFF / q == floor((2^32 - 1)/q)
        // which equals floor(2^32/q) unless q divides 2^32 (it doesn't for odd q).
        // For odd q this is exact.
    }

    // Cooperative load of zetas into threadgroup memory.
    // zlen <= 256, threads = n/2 = 128 typically, so each thread loads up to 2.
    for (uint i = ltid; i < zlen; i += half_n) {
        zt[i] = zetas[i];
    }

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load of polynomial.
    uint x0 = poly[ltid];
    uint y0 = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint m_barrett = qbarrett;

    // Stage 0 fused with load.
    {
        uint zeta = zt[1];
        uint prod = zeta * y0;
        // Barrett: t = prod - q * ((prod * m) >> 32)
        uint hi = mulhi(prod, m_barrett);
        uint t  = prod - hi * qv;
        if (t >= qv) t -= qv;

        uint sum_  = x0 + t;
        if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x0 + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        a[ltid]          = sum_;
        a[ltid + half_n] = diff_;
    }

    uint log2_len = (uint)(31 - (int)clz(half_n)) - 1u;
    uint k_start  = 2u;

    for (uint level = 1u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint length     = 1u << log2_len;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zt[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint prod = zeta * y;
        uint hi   = mulhi(prod, m_barrett);
        uint t    = prod - hi * qv;
        if (t >= qv) t -= qv;

        uint sum_  = x + t;
        if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        a[j]          = sum_;
        a[j + length] = diff_;

        k_start <<= 1u;
        if (log2_len > 0u) log2_len -= 1u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}