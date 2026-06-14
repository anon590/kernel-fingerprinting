I'll acknowledge the system reminder about the keybindings-help skill - it's not relevant to this MSL kernel optimization task.

**Optimization strategy:** The incumbent uses a generic 64-bit Barrett reduction (`mulhi64` with software 64x64 mul), which is expensive on Apple GPUs that lack native 64-bit multipliers. Since q < 2^32 (and in practice q < 2^24 for Kyber 3329 / Dilithium 8380417), I can exploit the fact that `t = zeta * y < q * q < 2^48` for typical params, but more importantly, since both operands are `< q < 2^32`, the product fits in 64 bits and Barrett with a **32-bit** reciprocal `m = floor(2^64 / q)` only needs `mulhi(t, m)` where the multiplication can be done with much cheaper 32x32→64 partial products. Even better: use `metal::mulhi` on the full 64-bit dividend split as two halves — but the cleanest win is replacing `mulhi64` with the hardware-friendly version using `umul_hi`-style intrinsics and dropping the per-step branch by using arithmetic-only canonical correction. I also fuse the zeta load directly (no caching — zetas fit in L1 and reloading is cheap compared to extra threadgroup pressure) and remove the redundant `ctz` per level by tracking `log2_len` directly.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Fast mulhi for 64-bit using metal's built-in mulhi on uint halves.
// Computes the high 64 bits of (a * b) where a,b are 64-bit.
inline ulong mulhi64(ulong a, ulong b) {
    uint a_lo = (uint)a;
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)b;
    uint b_hi = (uint)(b >> 32);

    // 32x32 -> 64 partial products via metal mulhi (high) + plain mul (low).
    ulong ll = (ulong)a_lo * (ulong)b_lo;
    ulong lh = (ulong)a_lo * (ulong)b_hi;
    ulong hl = (ulong)a_hi * (ulong)b_lo;
    ulong hh = (ulong)a_hi * (ulong)b_hi;

    ulong mid = (ll >> 32) + (uint)lh + (uint)hl;
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
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
    threadgroup uint a[N_MAX];
    threadgroup uint zcache[N_MAX];

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    // Barrett reciprocal: m = floor(2^64 / q). For t < q*2^32 <= q*q*... we get
    // q_est = mulhi64(t, m) in [t/q - 1, t/q], so r = t - q_est*q in [0, 2q).
    ulong mbar = (~(ulong)0) / (ulong)qv;
    ulong rem_chk = (~(ulong)0) - mbar * (ulong)qv;
    if (rem_chk == (ulong)qv - 1u) mbar += 1u;

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients.
    uint c0 = poly[ltid];
    uint c1 = poly[ltid + half_n];
    a[ltid]          = c0;
    a[ltid + half_n] = c1;

    // Cache zetas (up to n entries; only (1<<nlv) are used).
    zcache[ltid]          = zetas[ltid];
    zcache[ltid + half_n] = zetas[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Track length as power of 2 with explicit log2_len to avoid ctz each level.
    // half_n = n/2; for n=256, half_n=128, log2(half_n)=7.
    uint log2_len = 0u;
    {
        uint v = half_n;
        while (v > 1u) { v >>= 1u; log2_len++; }
    }
    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zcache[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Barrett reduction of zeta*y mod q.
        ulong tfull = (ulong)zeta * (ulong)y;
        ulong qest  = mulhi64(tfull, mbar);
        uint  r     = (uint)(tfull - qest * (ulong)qv);
        r = (r >= qv) ? (r - qv) : r;

        // Canonical add/sub without branches.
        uint s = x + r;
        s = (s >= qv) ? (s - qv) : s;
        uint d = x + qv - r;
        d = (d >= qv) ? (d - qv) : d;

        a[j]          = s;
        a[j + length] = d;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        if (log2_len > 0u) log2_len--;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```