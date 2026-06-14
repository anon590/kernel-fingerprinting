#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;
constexpr constant uint Z_MAX = 256u;

// Barrett-style reduction: given t < q^2 < 2^46, compute t mod q
// using precomputed m = floor(2^k / q). We use k = 46 so m fits in 32 bits
// for q up to ~2^23. Actually use a generic 64-bit approach.
// Since q < 2^23, t = a*b < 2^46. We compute:
//   approx = (t * m) >> shift; r = t - approx * q; if (r >= q) r -= q;
// where m = floor(2^shift / q) and shift = 46.
// But simpler and correct: just use (t - (t/q)*q). The division by a
// runtime q is slow. Use Barrett with shift=46.

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Barrett reduction. m_barrett = floor(2^48 / q). Then for t < 2^46:
//   q_est = (t * m) >> 48;  r = t - q_est * q;
//   r may be in [0, 2*q); subtract q if needed.
inline uint mod_mul_barrett(uint a, uint b, uint q, ulong m_barrett) {
    ulong t = (ulong)a * (ulong)b;
    // multiply t (up to 2^46) by m (up to 2^25 for q ~ 2^23) -> up to 2^71. Overflow!
    // Use mulhi style: compute high 64 bits of t * m.
    // Easier: shift t down first. Since t < 2^46 and we want (t*m) >> 48,
    // and m = floor(2^48/q), use the high part.
    // Actually we can use: q_est = mulhi(t, m) >> (48 - 64) ... messy.
    // Simplest correct path: fall back to t % q. The host's q is small;
    // compiler may optimize. But that's what we want to avoid.
    //
    // Use 128-bit emulation via two 32-bit pieces:
    // t = t_hi * 2^32 + t_lo
    // t * m = t_hi*m * 2^32 + t_lo*m
    uint t_lo = (uint)t;
    uint t_hi = (uint)(t >> 32);
    ulong prod_lo = (ulong)t_lo * m_barrett;
    ulong prod_hi = (ulong)t_hi * m_barrett;
    // (t * m) >> 48 = ((prod_hi << 32) + prod_lo) >> 48
    //               = (prod_hi >> 16) + (prod_lo >> 48)  [with carry from low bits]
    // Combine: total = prod_hi << 32 | ... but we just need >>48.
    // result = (prod_hi >> 16) + (prod_lo >> 48)
    // We need to include carry from (prod_hi_low_16 << 32 + prod_lo) overflow into bit 48.
    // top 16 bits of prod_lo combine with low 16 bits of prod_hi:
    ulong mid = (prod_hi << 16) + (prod_lo >> 32); // careful: this can overflow
    // safer: do it carefully
    ulong q_est = (prod_hi >> 16) + ((prod_hi & 0xFFFFul) << 16) + (prod_lo >> 32);
    q_est >>= 16;
    // Hmm, let me redo cleanly.
    // total_high64 = prod_hi + (prod_lo >> 32) + carry_from_low_part
    // Actually: t*m as 128-bit = prod_hi * 2^32 + prod_lo (treating both as 64-bit).
    // Let H = prod_hi, L = prod_lo.
    // Then total = H * 2^32 + L. >> 48 = (H >> 16) + (L >> 48) provided we
    // account for the low 16 bits of H lining up with bits 32..47 of total.
    // bits [48..) of total = bits [16..) of H + carry from (bits [0..16) of H << 32 + L) into bit 48.
    ulong low_part = ((H_placeholder()) << 32); // placeholder removed below
    return 0u;
}

// Given the complexity, use a cleaner approach: precompute m = 2^32 / q approx
// using a different strategy. Or just trust the compiler on uint % uint.

inline uint mod_mul(uint a, uint b, uint q) {
    ulong t = (ulong)a * (ulong)b;
    return (uint)(t % (ulong)q);
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
    threadgroup uint zs[Z_MAX];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load of coefficients
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperative load of zetas into threadgroup memory
    uint z_count = 1u << n_levels;
    if (ltid < z_count) {
        zs[ltid] = zetas[ltid];
    }
    if (ltid + half_n < z_count) {
        zs[ltid + half_n] = zetas[ltid + half_n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zs[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, q);

        uint sum  = mod_add(x, t, q);
        uint diff = mod_sub(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum;
        a[j + length] = diff;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}