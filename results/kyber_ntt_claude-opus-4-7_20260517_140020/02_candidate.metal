#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Reduce a 64-bit value t (with t < q * 2^32, i.e. zeta*y where zeta,y < q < 2^32)
// to canonical [0, q) using Barrett with a 32-bit reciprocal.
//
// We need m and shift such that for all t in [0, q*2^32):
//   floor(t / q) == floor(t * m / 2^(32+shift))   approximately, with <=2 correction.
//
// Use: shift = ceil(log2(q)); m = floor(2^(32+shift) / q).
// Then q_est = mulhi(t_lo_or_full * m), corrected.
//
// Simpler: since q < 2^24 in all real ZK params (Kyber 3329, Dilithium 8380417),
// just use 64-bit mul by m where m = floor(2^k / q), and shift right by k.
// We pick k = 32 + ceil(log2(q)) so quotient fits.

// Compute mulhi for 64-bit: high 64 bits of a*b, where a,b are 64-bit.
inline ulong mulhi64(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFULL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFULL;
    ulong b_hi = b >> 32;
    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;
    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    int d = (int)a - (int)b;
    return (d < 0) ? (uint)(d + (int)q) : (uint)d;
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
    threadgroup uint zcache[N_MAX]; // cache zetas[1..n-1]

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    // Precompute Barrett constants for 64-bit dividend:
    //   shift = bit_width(q)  (smallest s with 2^s >= q, but we use ceil-log2 robustly)
    //   m = floor(2^(64) / q)
    // Then for t < q * 2^32 <= 2^(32+shift):
    //   q_est = mulhi64(t, m); remainder = t - q_est * q in [0, 2q); correct.
    ulong mbar = (~(ulong)0) / (ulong)qv;
    // possible +1 correction so that mbar = floor(2^64/q) exactly when 2^64 % q == 0
    ulong rem  = (~(ulong)0) - mbar * (ulong)qv;
    if (rem == (ulong)qv - 1) mbar += 1;

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cache zetas[0..n-1] (at most 256 entries). Two loads per thread covers n/2 threads.
    // Total zetas used: (1 << nlv) entries, which is <= n. We just load n entries; extras unused.
    zcache[ltid]          = zetas[ltid];
    zcache[ltid + half_n] = zetas[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = (length == 0u) ? 0u : (ltid >> (uint)ctz(length));
        // ltid / length where length is power of two
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zcache[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t_full = zeta * y, fits in 64 bits since both < 2^32.
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett: q_est = mulhi64(tfull, mbar); r = tfull - q_est * q
        ulong qest = mulhi64(tfull, mbar);
        uint  r    = (uint)(tfull - qest * (ulong)qv);
        // r in [0, 2q); single correction.
        if (r >= qv) r -= qv;

        a[j]          = mod_add(x, r, qv);
        a[j + length] = mod_sub(x, r, qv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}