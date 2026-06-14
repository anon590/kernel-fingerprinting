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
    threadgroup uint a[N_MAX];
    threadgroup uint zcache[N_MAX];
    threadgroup uint zshoup[N_MAX];

    const uint qv     = q;
    const uint nv     = n;
    const uint nlv    = n_levels;
    const uint half_n = nv >> 1u;
    const uint zlen   = 1u << nlv;

    device uint *poly = coeffs + (size_t)tgid * (size_t)nv;

    // Load polynomial coefficients into threadgroup memory.
    if (ltid < half_n) {
        a[ltid]          = poly[ltid];
        a[ltid + half_n] = poly[ltid + half_n];
    }

    // Load zetas and compute Shoup constants: zsh = floor(zeta * 2^32 / q).
    // zlen <= 256, half_n <= 128, so each of the half_n threads handles up to
    // ceil(zlen / half_n) entries. We loop with stride half_n to cover all of zlen.
    for (uint idx = ltid; idx < zlen; idx += half_n) {
        uint zv = zetas[idx];
        zcache[idx] = zv;
        // zsh = floor((zv << 32) / q). Since zv < q <= 2^32-1, the numerator fits.
        zshoup[idx] = (uint)(((ulong)zv << 32) / (ulong)qv);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length   = half_n;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));
    uint k_start  = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;

        uint zeta = zcache[k_start + group_idx];
        uint zsh  = zshoup[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Shoup multiplication: returns t = zeta*y mod q in [0, 2q), then reduce.
        // q_hat = mulhi(zsh, y);  r = zeta*y - q_hat*q  (both mod 2^32)
        uint q_hat = mulhi(zsh, y);
        uint r     = zeta * y - q_hat * qv;
        if (r >= qv) r -= qv;

        // Canonical butterfly.
        uint sum = x + r;
        sum = (sum >= qv) ? (sum - qv) : sum;
        uint dif = x + qv - r;
        dif = (dif >= qv) ? (dif - qv) : dif;

        a[j]          = sum;
        a[j + length] = dif;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        log2_len = (log2_len == 0u) ? 0u : (log2_len - 1u);
    }

    if (ltid < half_n) {
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
    }
}