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
    uint x0 = poly[ltid];
    uint y0 = poly[ltid + half_n];

    // Stage 0 fused with the load: zeta_index = 1, length = half_n,
    // j = ltid, j+length = ltid+half_n.
    {
        uint zeta = zetas[1];
        // q small (Kyber q=3329), so zeta*y0 fits well in 32 bits.
        uint prod = zeta * y0;
        uint t = prod % qv;

        uint sum_  = x0 + t;
        if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x0 + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        a[ltid]          = sum_;
        a[ltid + half_n] = diff_;
    }

    // log2(length) starts at log2(half_n) - 1, since stage 0 already done.
    // length for level L is n >> (L+1). We track log2_len.
    // Compute log2(half_n): nl-1 because n = 1 << nl is the typical setup,
    // but n could be smaller; derive via clz.
    uint log2_len = (uint)(31 - clz(half_n)) - 1u; // log2(half_n/2)
    uint k_start  = 2u;

    for (uint level = 1u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint length     = 1u << log2_len;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint prod = zeta * y;
        uint t = prod % qv;

        uint sum_  = x + t;
        if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        // Each thread's writes target the same (j, j+length) it just read.
        // The next level's reads happen after the barrier at the top.
        a[j]          = sum_;
        a[j + length] = diff_;

        k_start <<= 1u;
        if (log2_len > 0u) log2_len -= 1u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}