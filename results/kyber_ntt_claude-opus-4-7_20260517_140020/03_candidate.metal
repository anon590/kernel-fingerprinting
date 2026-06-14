#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Compute -q^{-1} mod 2^32 via Newton iteration (5 steps suffice for 32-bit).
inline uint neg_qinv32(uint q) {
    // x = q^{-1} mod 2^k, doubling k each iter starting from k=3 (q is odd).
    uint x = q;                       // q^{-1} mod 8 (since q odd)
    x = x * (2u - q * x);             // mod 2^6
    x = x * (2u - q * x);             // mod 2^12
    x = x * (2u - q * x);             // mod 2^24
    x = x * (2u - q * x);             // mod 2^48 -> low 32 bits valid mod 2^32
    // Now x == q^{-1} mod 2^32. We want -q^{-1} mod 2^32.
    return 0u - x;
}

// Montgomery reduce: input t < q * 2^32, output in [0, q) (assuming q < 2^31).
// r = (t + ((uint32)(t) * qinv_neg) * q) >> 32 ; if r >= q, r -= q.
// Using qinv_neg = -q^{-1} mod 2^32, the classic form:
//   m = (uint32)t * qinv_neg ;  r = (t + m*q) >> 32
inline uint mont_reduce(ulong t, uint q, uint qinv_neg) {
    uint tlo = (uint)t;
    uint m   = tlo * qinv_neg;          // mod 2^32
    ulong mq = (ulong)m * (ulong)q;
    ulong s  = t + mq;                  // low 32 bits become zero
    uint r   = (uint)(s >> 32);
    // r < 2q ; canonicalize.
    if (r >= q) r -= q;
    return r;
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}
inline uint mod_sub(uint a, uint b, uint q) {
    uint t = a + q - b;
    return (t >= q) ? (t - q) : t;
}

// One-shot Barrett-ish reduce of (a * b) mod q, where a*b < q * 2^32.
// Used only once per zeta during Montgomery-form conversion at kernel start.
inline uint mulmod_barrett(uint a, uint b, uint q, ulong mbar) {
    ulong t = (ulong)a * (ulong)b;
    // q_est = mulhi64(t, mbar)
    ulong a_lo = t & 0xFFFFFFFFULL, a_hi = t >> 32;
    ulong b_lo = mbar & 0xFFFFFFFFULL, b_hi = mbar >> 32;
    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;
    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    ulong qest = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    uint r = (uint)(t - qest * (ulong)q);
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
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
    threadgroup uint zmont[N_MAX]; // zetas in Montgomery form

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    uint qinv_neg = neg_qinv32(qv);

    // R = 2^32 mod q.  Compute via: R = (2^32) - floor(2^32 / q) * q  using 64-bit.
    // 2^32 = (1ULL << 32). r32 = (uint)((1ULL<<32) % q).
    uint R = (uint)(((ulong)1 << 32) % (ulong)qv);

    // Barrett reciprocal mbar = floor(2^64 / q) for converting zetas to Montgomery form.
    // Since we need it only ~n times once per polynomial, this is fine.
    ulong mbar = (~(ulong)0) / (ulong)qv;
    ulong rem  = (~(ulong)0) - mbar * (ulong)qv;
    if (rem == (ulong)qv - 1) mbar += 1;

    device uint *poly = coeffs + (size_t)tgid * (ulong)nv;

    // Load coefficients.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Convert zetas to Montgomery form: zmont[k] = zetas[k] * R mod q.
    // We only need indices [1, (1<<nlv)-1], but loading all n is fine; extras unused.
    {
        uint z0 = zetas[ltid];
        uint z1 = zetas[ltid + half_n];
        zmont[ltid]          = mulmod_barrett(z0, R, qv, mbar);
        zmont[ltid + half_n] = mulmod_barrett(z1, R, qv, mbar);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length    = half_n;
    uint log2_len  = 0u;
    // log2(half_n): half_n <= 128 = 2^7
    {
        uint v = half_n;
        while (v > 1u) { v >>= 1u; log2_len++; }
    }
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = (log2_len == 0u) ? ltid : (ltid >> log2_len);
        uint j_in_group = (length == 0u) ? 0u : (ltid & (length - 1u));
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta_m     = zmont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Montgomery multiply: t = mont_reduce(zeta_m * y) in [0, q).
        ulong prod = (ulong)zeta_m * (ulong)y;
        uint  t    = mont_reduce(prod, qv, qinv_neg);

        a[j]          = mod_add(x, t, qv);
        a[j + length] = mod_sub(x, t, qv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        if (log2_len > 0u) log2_len--;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}