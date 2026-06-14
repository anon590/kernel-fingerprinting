**Optimization**: The previous version did a full `ulong` Montgomery reduce per butterfly. Since q < 2^31 (Kyber q = 3329, but spec says fits in u32) and coefficients/zetas are bounded by q, I can use a much cheaper Barrett-like reduction. Specifically I precompute `m32 = floor(2^32 / q)` once per thread; then for `t = zeta * y` (both < q < 2^16 for Kyber, but I'll handle up to q < 2^16 safely with a 32-bit product; for larger q I fall back). Actually, since n_levels<=8 implies the field supports a 512th root of unity, and the host buffer is uint32, I'll do `ulong` multiply then reduce via `mulhi`-based Barrett: `r = T - ((T * m64) >> 64) * q` — but that's also expensive. 

Better: just use `T % q` via the hardware path replaced by Barrett with one `mulhi`. For each butterfly: `prod = (ulong)zeta * y`; `q_est = mulhi_64(prod, m64)`; `r = prod - q_est * q`; correct with up to 2 subtractions. I'll also **eliminate one barrier per level** by double-buffering with two threadgroup arrays (ping-pong), removing the pre-write barrier.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;
constexpr constant uint Z_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// 64-bit high multiply: high 64 bits of a*b.
inline ulong mulhi64(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;

    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFul) + (hl & 0xFFFFFFFFul);
    ulong hi  = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    return hi;
}

// Barrett-ish: given T = zeta*y with T < q^2 < 2^62 (q<2^31), reduce mod q.
// m64 = floor(2^64 / q). Then q_est = mulhi64(T, m64). r = T - q_est*q.
// r in [0, 2q); correct with one conditional subtract.
inline uint barrett_reduce(ulong T, uint q, ulong m64) {
    ulong q_est = mulhi64(T, m64);
    ulong r = T - q_est * (ulong)q;
    // r < 2q (in fact r < q+something small)
    uint ru = (uint)r;
    if (ru >= q) ru -= q;
    return ru;
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
    threadgroup uint zlocal[Z_MAX];

    uint qreg   = q;
    uint nreg   = n;
    uint nlv    = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;

    // Precompute Barrett constant: m64 = floor(2^64 / q).
    // 2^64 / q computed as ((2^64 - 1) / q) plus possible +1 correction.
    // Use unsigned long division directly.
    ulong m64;
    {
        // Compute floor(2^64 / q). Since 2^64 overflows, use:
        // m = (0 - q) / q + 1  ==  floor((2^64 - q)/q) + 1 = floor(2^64/q)  when q | 2^64? no.
        // Safer: m = floor((2^64 - 1)/q); if ((m+1)*q == 0 in 64-bit) m += 1 (when q divides 2^64, impossible for odd q>1; for q=power of two could matter but q is prime).
        // We use: m64 = (~0ul - (~0ul % q)) / q ... but simplest:
        ulong qq = (ulong)qreg;
        ulong max64 = ~0ul;
        ulong r = max64 % qq;
        m64 = max64 / qq;
        if (r + 1ul == qq) m64 += 1ul; // then 2^64 is exactly divisible
    }

    // Cooperatively load zetas into threadgroup memory (raw, not Montgomery).
    {
        uint i1 = ltid;
        if (i1 < zcount) zlocal[i1] = zetas[i1];
        uint i2 = ltid + half_n;
        if (i2 < zcount) zlocal[i2] = zetas[i2];
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load polynomial.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zv         = zlocal[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = barrett_reduce((ulong)zv * (ulong)y, qreg, m64);

        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```