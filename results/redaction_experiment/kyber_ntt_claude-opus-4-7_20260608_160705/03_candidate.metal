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
    threadgroup uint zs[N_MAX];

    uint half_n = n >> 1u;
    uint qv = q;
    uint nl = n_levels;

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load of coefficients (each thread loads 2 slots; covers n<=256)
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperative load of zetas table (at most 256 entries, half_n>=1)
    uint z_count = 1u << nl;
    if (ltid < z_count) {
        zs[ltid] = zetas[ltid];
    }
    uint idx2 = ltid + half_n;
    if (idx2 < z_count) {
        zs[idx2] = zetas[idx2];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    ulong qL = (ulong)qv;

    for (uint level = 0u; level < nl; ++level) {
        // Compute butterfly indices
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zs[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t = (zeta * y) mod q.  zeta, y < q < 2^23, so product < 2^46.
        ulong prod = (ulong)zeta * (ulong)y;
        uint t = (uint)(prod % qL);

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