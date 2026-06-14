#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

inline void umul128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    lo = (p00 & EPSILON) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Reduce x = lo + hi * 2^64 (mod p).
// Requires hi < 2^63 so that (hi_lo << 32) - hi_lo doesn't have issues and
// subsequent additions stay representable. Sum of 4 canonical products has
// hi < 4 * (p-1)^2 / 2^64 < 4 * 2^64, so we may have hi up to ~2^66; handle
// by iterative reduction.
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong r = t0 + t1;
    if (r < t0) r += EPSILON;
    if (r >= P_GOLD) r -= P_GOLD;
    return r;
}

inline ulong gmul(ulong a, ulong b) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    return gold_reduce128(lo, hi);
}

// Two-limb MAC: (acc_lo, acc_hi) += a * b. Caller must guarantee no overflow
// out of the 128-bit accumulator. Sum of up to 4 canonical products: each
// product < (p-1)^2 < 2^128 - 2^97 + ..., so 4 products sum < 4*(2^128) which
// CAN overflow. However each product < 2^128 - 2^64*(something), and in
// practice for Goldilocks canonical values (p-1)^2 ≈ 2^128 - 2^97; 4 of those
// is ~2^130 which DOES overflow. So we still need 3-limb. BUT: since canonical
// values are < p < 2^64, and p < 2^64, product hi part < p (because hi = floor(a*b/2^64) < a < 2^64,
// but more tightly hi < p when both a,b < p). Sum of 4 hi parts < 4*p < 2^66.
// We can carry overflow of the lo-sum into hi without losing it, as long as
// final hi < 2^64. 4*p ≈ 2^66 — overflows 64 bits. So we DO need a 3-limb.
// Instead: reduce per-pair. Reduce after each pair to keep accumulator bounded.

inline void mac128(ulong a, ulong b,
                   thread ulong &acc_lo, thread ulong &acc_hi) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    ulong new_lo = acc_lo + lo;
    ulong carry = (ulong)(new_lo < acc_lo);
    acc_lo = new_lo;
    acc_hi = acc_hi + hi + carry;
}

inline ulong sbox7(ulong x) {
    ulong x2 = gmul(x, x);
    ulong x4 = gmul(x2, x2);
    ulong x3 = gmul(x2, x);
    return gmul(x4, x3);
}

// 3x3 row dot: reduce intermediate after 2 macs to keep hi bounded.
inline ulong row3(ulong m0, ulong m1, ulong m2,
                  ulong s0, ulong s1, ulong s2) {
    // First two macs: hi can be up to 2*p < 2^65 -> overflows 64 bits possible.
    // To be safe, reduce after first 2, then add the third product, then reduce.
    ulong lo = 0, hi = 0;
    mac128(m0, s0, lo, hi);
    mac128(m1, s1, lo, hi);
    ulong r01 = gold_reduce128(lo, hi);
    // Now r01 < p < 2^64; one more product fits.
    ulong lo2, hi2;
    umul128(m2, s2, lo2, hi2);
    ulong nl = lo2 + r01;
    ulong c  = (ulong)(nl < lo2);
    ulong nh = hi2 + c;
    return gold_reduce128(nl, nh);
}

inline ulong row4(ulong m0, ulong m1, ulong m2, ulong m3,
                  ulong s0, ulong s1, ulong s2, ulong s3) {
    ulong lo = 0, hi = 0;
    mac128(m0, s0, lo, hi);
    mac128(m1, s1, lo, hi);
    ulong r01 = gold_reduce128(lo, hi);
    ulong lo2 = 0, hi2 = 0;
    mac128(m2, s2, lo2, hi2);
    mac128(m3, s3, lo2, hi2);
    ulong r23 = gold_reduce128(lo2, hi2);
    return gadd(r01, r23);
}

inline void mds3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                 const thread ulong M[9]) {
    ulong n0 = row3(M[0], M[1], M[2], s0, s1, s2);
    ulong n1 = row3(M[3], M[4], M[5], s0, s1, s2);
    ulong n2 = row3(M[6], M[7], M[8], s0, s1, s2);
    s0 = n0; s1 = n1; s2 = n2;
}

inline void mds4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                 const thread ulong M[16]) {
    ulong n0 = row4(M[0],  M[1],  M[2],  M[3],  s0, s1, s2, s3);
    ulong n1 = row4(M[4],  M[5],  M[6],  M[7],  s0, s1, s2, s3);
    ulong n2 = row4(M[8],  M[9],  M[10], M[11], s0, s1, s2, s3);
    ulong n3 = row4(M[12], M[13], M[14], M[15], s0, s1, s2, s3);
    s0 = n0; s1 = n1; s2 = n2; s3 = n3;
}

inline void poseidon2_t3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[9],
                         const thread ulong D[3],
                         uint r_f, uint r_p)
{
    mds3(s0, s1, s2, M);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        mds3(s0, s1, s2, M);
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0, s1), s2);
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        mds3(s0, s1, s2, M);
    }
}

inline void poseidon2_t4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[16],
                         const thread ulong D[4],
                         uint r_f, uint r_p)
{
    mds4(s0, s1, s2, s3, M);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        mds4(s0, s1, s2, s3, M);
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0, s1), gadd(s2, s3));
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        ulong n3 = gadd(sum, gmul(D[3], s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        mds4(s0, s1, s2, s3, M);
    }
}

kernel void merkle_build_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &arity        [[buffer(5)]],
    constant uint      &t            [[buffer(6)]],
    constant uint      &r_f          [[buffer(7)]],
    constant uint      &r_p          [[buffer(8)]],
    constant uint      &in_offset    [[buffer(9)]],
    constant uint      &out_offset   [[buffer(10)]],
    constant uint      &child_count  [[buffer(11)]],
    uint p [[thread_position_in_grid]])
{
    uint parent_count = (child_count + arity - 1u) / arity;
    if (p >= parent_count) return;

    uint base = p * arity;

    if (t == 3u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];

        ulong M[9];
        for (uint k = 0u; k < 9u; ++k) M[k] = ext_mds[k];
        ulong D[3];
        D[0] = int_diag[0]; D[1] = int_diag[1]; D[2] = int_diag[2];

        poseidon2_t3(s0, s1, s2, rc_ext, rc_int, M, D, r_f, r_p);

        tree[out_offset + p] = s0;
        return;
    }

    if (t == 4u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul, s3 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];
        if (arity >= 4u && base + 3u < child_count) s3 = tree[in_offset + base + 3u];

        ulong M[16];
        for (uint k = 0u; k < 16u; ++k) M[k] = ext_mds[k];
        ulong D[4];
        D[0] = int_diag[0]; D[1] = int_diag[1]; D[2] = int_diag[2]; D[3] = int_diag[3];

        poseidon2_t4(s0, s1, s2, s3, rc_ext, rc_int, M, D, r_f, r_p);

        tree[out_offset + p] = s0;
        return;
    }

    // Generic fallback
    ulong state[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint i = 0u; i < arity; ++i) {
        uint src = base + i;
        if (src < child_count && i < t) state[i] = tree[in_offset + src];
    }
    ulong M[16];
    for (uint k = 0u; k < t * t; ++k) M[k] = ext_mds[k];
    ulong D[4];
    for (uint k = 0u; k < t; ++k) D[k] = int_diag[k];

    // initial mds (generic): use repeated reduction per pair
    {
        ulong tmp[4];
        for (uint i = 0u; i < t; ++i) {
            ulong acc = gmul(M[i * t + 0u], state[0]);
            for (uint j = 1u; j < t; ++j) acc = gadd(acc, gmul(M[i * t + j], state[j]));
            tmp[i] = acc;
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    uint half_f = r_f >> 1u;
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        ulong tmp[4];
        for (uint i = 0u; i < t; ++i) {
            ulong acc = gmul(M[i * t + 0u], state[0]);
            for (uint j = 1u; j < t; ++j) acc = gadd(acc, gmul(M[i * t + j], state[j]));
            tmp[i] = acc;
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gadd(state[0], rc_int[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < t; ++i) s = gadd(s, state[i]);
        ulong tmp[4];
        for (uint i = 0u; i < t; ++i) tmp[i] = gadd(s, gmul(D[i], state[i]));
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    for (uint r = half_f; r < r_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        ulong tmp[4];
        for (uint i = 0u; i < t; ++i) {
            ulong acc = gmul(M[i * t + 0u], state[0]);
            for (uint j = 1u; j < t; ++j) acc = gadd(acc, gmul(M[i * t + j], state[j]));
            tmp[i] = acc;
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    tree[out_offset + p] = state[0];
}