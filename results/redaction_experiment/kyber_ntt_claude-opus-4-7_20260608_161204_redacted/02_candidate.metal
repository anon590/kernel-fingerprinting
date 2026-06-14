#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Fast modular reduction of t = zeta * y where zeta, y < q < 2^32.
// Uses a precomputed Barrett constant m = floor(2^k / q), with k chosen
// so that the high-bits multiplication yields a quotient within 1-2 of
// the true quotient. We pick k = 62 so that m fits in 64 bits and
// t * m fits before shift via emulated 128-bit upper word.
//
// Simpler & sufficient path for q up to ~2^23 (Kyber 3329, Dilithium ~2^23):
// t fits in < 2^46, so (t * m) >> 62 with m = floor(2^62 / q) fits in ulong.
inline uint barrett_mul_mod(uint a, uint b, uint q, ulong m62) {
    ulong t = (ulong)a * (ulong)b;        // < 2^46 for typical ZK primes
    // q_hat = (t * m) >> 62; both t and m fit in 64 bits and (t*m) may overflow
    // for general q, but for q < 2^23 we have t < 2^46 and m < 2^40 so
    // t*m < 2^86 -- overflow. So we instead use mulhi-style: compute
    // (t * m) >> 62 carefully.
    // Decompose: t * m = (t_hi * 2^32 + t_lo) * m
    ulong t_hi = t >> 32;
    ulong t_lo = t & 0xffffffffUL;
    // (t_hi * m) << 32 + (t_lo * m)
    // We only need >> 62 of the full product.
    ulong hi_part = t_hi * m62;           // contributes >> 30 after the <<32 - 62 = -30 shift, i.e. << 30 of low bits, but we want >> 62 of full
    ulong lo_part = t_lo * m62;           // >> 62
    // Full product P = hi_part * 2^32 + lo_part.
    // P >> 62 = (hi_part << 32 | lo_part) >> 62
    //        = (hi_part >> 30) + carry-bit from lo_part top
    ulong q_hat = (hi_part >> 30) + ((lo_part >> 62) + ((hi_part & 0x3fffffffUL) << 2 >> 0));
    // The above is fiddly; let's just do it cleanly:
    // P_high64 = hi_part + (lo_part >> 32); P_low32 = lo_part & 0xffffffff
    // Then P >> 62 = P_high64 >> 30 (since we shift past the low 32 + 30 more bits)
    ulong P_high = hi_part + (lo_part >> 32);
    q_hat = P_high >> 30;

    ulong r = t - q_hat * (ulong)q;
    uint ri = (uint)r;
    if (ri >= q) ri -= q;
    if (ri >= q) ri -= q;
    return ri;
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

    const uint q_local = q;
    const uint half_n  = n >> 1u;

    // Barrett constant: m = floor(2^62 / q). For q < 2^23, m has ~39 bits.
    const ulong m62 = ((ulong)1 << 62) / (ulong)q_local;

    device uint *poly = coeffs + (size_t)tgid * (ulong)n;

    // Coalesced load.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = barrett_mul_mod(zeta, y, q_local, m62);

        a[j]          = mod_add(x, t, q_local);
        a[j + length] = mod_sub(x, t, q_local);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}