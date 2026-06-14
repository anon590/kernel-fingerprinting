**Optimization:** Replace the slow `% q` integer modulo with Barrett-style reduction using a precomputed reciprocal of `q` (computed once per threadgroup via `mulhi`). Since `q < 2^32` and the product `zeta * y < 2^64`, we can reduce a 64-bit value `t` by computing `q_hi = mulhi(t_hi, m) + ...` — but simpler and equally effective: use a 64-bit Barrett reduction with a precomputed 64-bit reciprocal `m = floor(2^64 / q)`. Then `t mod q = t - ((t * m) >> 64) * q`, with at most one correction subtraction. This removes the expensive hardware division (which dominates the seed's runtime) and is bit-exact. Also unroll the load/store and keep additions/subtractions branchless-ish.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Compute floor(2^64 / q) using Metal's 64-bit unsigned division.
// Called once per threadgroup; q is small so this is cheap relative
// to the savings during the butterfly inner loop.
inline ulong barrett_m(uint q) {
    // 2^64 / q  =  ((2^64 - 1) / q) with a possible +1 correction.
    // Use: m = (~0ULL) / q; if ((~0ULL) - m*q == q-1) m += 1; but
    // simpler: compute via two halves. We just use ulong division.
    ulong qq = (ulong)q;
    ulong m  = (~(ulong)0) / qq;
    // Check if 2^64 is exactly divisible; if (~0ULL) % q == q-1 then add 1.
    ulong r  = (~(ulong)0) - m * qq;
    if (r == qq - 1) m += 1;
    return m;
}

// Reduce a 64-bit t into [0, q) given m = floor(2^64 / q).
// h = floor(t * m / 2^64); r = t - h*q; r in [0, 2q); subtract q if needed.
inline uint barrett_reduce(ulong t, uint q, ulong m) {
    // Compute high 64 bits of t * m using 32-bit halves.
    ulong t_lo = t & 0xFFFFFFFFULL;
    ulong t_hi = t >> 32;
    ulong m_lo = m & 0xFFFFFFFFULL;
    ulong m_hi = m >> 32;

    ulong ll = t_lo * m_lo;
    ulong lh = t_lo * m_hi;
    ulong hl = t_hi * m_lo;
    ulong hh = t_hi * m_hi;

    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    ulong h   = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);

    ulong r = t - h * (ulong)q;
    // r is in [0, 2q) (typically), correct with up to 2 subtractions to be safe.
    uint ru = (uint)r;
    if (ru >= q) ru -= q;
    if (ru >= q) ru -= q;
    return ru;
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
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
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    uint qv      = q;
    ulong m      = barrett_m(qv);
    uint half_n  = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load two coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    uint nlv     = n_levels;
    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = barrett_reduce((ulong)zeta * (ulong)y, qv, m);

        a[j]          = mod_add(x, t, qv);
        a[j + length] = mod_sub(x, t, qv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```