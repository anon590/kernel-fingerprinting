#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline void mul128(ulong a, ulong b, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00 = (ulong)a0 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (ulong)(uint)p01 + (ulong)(uint)p10;
    lo = (uint)p00 | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

inline void sqr128(ulong a, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);

    ulong p01 = (ulong)a0 * a1;
    ulong p00 = (ulong)a0 * a0;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = (p00 >> 32) + ((ulong)(uint)p01 << 1);
    lo = (uint)p00 | (mid << 32);
    hi = p11 + ((p01 >> 32) << 1) + (mid >> 32);
}

inline void mac128(ulong a, ulong b, thread ulong &hi, thread ulong &lo) {
    ulong h, l;
    mul128(a, b, h, l);
    lo += l;
    hi += h + (lo < l ? 1ul : 0ul);
}

inline void mac128_small(ulong a, uint b0, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    ulong p00 = (ulong)a0 * b0;
    ulong p10 = (ulong)a1 * b0;
    ulong mid = (p00 >> 32) + (ulong)(uint)p10;
    ulong l = (uint)p00 | (mid << 32);
    ulong h = (p10 >> 32) + (mid >> 32);
    lo += l;
    hi += h + (lo < l ? 1ul : 0ul);
}

inline ulong reduce128(ulong hi, ulong lo) {
    ulong hi_hi = hi >> 32;
    ulong hi_lo = (uint)hi;
    
    ulong sub = hi_hi + hi_lo;
    ulong t0 = lo - sub;
    t0 -= (lo < sub) ? 0xFFFFFFFFul : 0ul;
    
    ulong t1 = hi_lo << 32;
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;
    
    return (t2 >= P_GOLD) ? t2 - P_GOLD : t2;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? t - P_GOLD : t;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong hi, lo;
    mul128(a, b, hi, lo);
    return reduce128(hi, lo);
}

inline ulong gold_sqr(ulong a) {
    ulong hi, lo;
    sqr128(a, hi, lo);
    return reduce128(hi, lo);
}

inline ulong gold_mac(ulong a, ulong b, ulong c) {
    ulong hi = 0, lo = a;
    mac128(b, c, hi, lo);
    return reduce128(hi, lo);
}

inline ulong gold_mac_small(ulong a, ulong b, uint c) {
    ulong hi = 0, lo = a;
    mac128_small(b, c, hi, lo);
    return reduce128(hi, lo);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void mds_multiply_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                           ulong m00, ulong m01, ulong m02,
                           ulong m10, ulong m11, ulong m12,
                           ulong m20, ulong m21, ulong m22,
                           bool mds_small)
{
    if (mds_small) {
        ulong hi0 = 0, lo0 = 0;
        mac128_small(s0, (uint)m00, hi0, lo0);
        mac128_small(s1, (uint)m01, hi0, lo0);
        mac128_small(s2, (uint)m02, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);

        ulong hi1 = 0, lo1 = 0;
        mac128_small(s0, (uint)m10, hi1, lo1);
        mac128_small(s1, (uint)m11, hi1, lo1);
        mac128_small(s2, (uint)m12, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);

        ulong hi2 = 0, lo2 = 0;
        mac128_small(s0, (uint)m20, hi2, lo2);
        mac128_small(s1, (uint)m21, hi2, lo2);
        mac128_small(s2, (uint)m22, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);

        s0 = n0; s1 = n1; s2 = n2;
    } else {
        ulong hi0 = 0, lo0 = 0;
        mac128(s0, m00, hi0, lo0);
        mac128(s1, m01, hi0, lo0);
        mac128(s2, m02, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);

        ulong hi1 = 0, lo1 = 0;
        mac128(s0, m10, hi1, lo1);
        mac128(s1, m11, hi1, lo1);
        mac128(s2, m12, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);

        ulong hi2 = 0, lo2 = 0;
        mac128(s0, m20, hi2, lo2);
        mac128(s1, m21, hi2, lo2);
        mac128(s2, m22, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);

        s0 = n0; s1 = n1; s2 = n2;
    }
}

inline void poseidon2_permute_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];
    bool mds_small = ((m00 | m01 | m02 | m10 | m11 | m12 | m20 | m21 | m22) >> 32) == 0;

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];
    bool diag_small = ((d0 | d1 | d2) >> 32) == 0;

    mds_multiply_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22, mds_small);

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        mds_multiply_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22, mds_small);
    }

    if (diag_small) {
        uint d0_s = (uint)d0; uint d1_s = (uint)d1; uint d2_s = (uint)d2;
        #pragma unroll 4
        for (uint r = 0; r < 32; ++r) {
            if (r >= r_p) break;
            s0 = sbox7(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            s0 = gold_mac_small(sum, s0, d0_s);
            s1 = gold_mac_small(sum, s1, d1_s);
            s2 = gold_mac_small(sum, s2, d2_s);
        }
    } else {
        #pragma unroll 4
        for (uint r = 0; r < 32; ++r) {
            if (r >= r_p) break;
            s0 = sbox7(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            s0 = gold_mac(sum, s0, d0);
            s1 = gold_mac(sum, s1, d1);
            s2 = gold_mac(sum, s2, d2);
        }
    }

    uint rem_f = r_f - half_f;
    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= rem_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        mds_multiply_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22, mds_small);
    }
}

inline void mds_multiply_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                           ulong m00, ulong m01, ulong m02, ulong m03,
                           ulong m10, ulong m11, ulong m12, ulong m13,
                           ulong m20, ulong m21, ulong m22, ulong m23,
                           ulong m30, ulong m31, ulong m32, ulong m33,
                           bool mds_small)
{
    if (mds_small) {
        ulong hi0 = 0, lo0 = 0;
        mac128_small(s0, (uint)m00, hi0, lo0); mac128_small(s1, (uint)m01, hi0, lo0);
        mac128_small(s2, (uint)m02, hi0, lo0); mac128_small(s3, (uint)m03, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);

        ulong hi1 = 0, lo1 = 0;
        mac128_small(s0, (uint)m10, hi1, lo1); mac128_small(s1, (uint)m11, hi1, lo1);
        mac128_small(s2, (uint)m12, hi1, lo1); mac128_small(s3, (uint)m13, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);

        ulong hi2 = 0, lo2 = 0;
        mac128_small(s0, (uint)m20, hi2, lo2); mac128_small(s1, (uint)m21, hi2, lo2);
        mac128_small(s2, (uint)m22, hi2, lo2); mac128_small(s3, (uint)m23, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);

        ulong hi3 = 0, lo3 = 0;
        mac128_small(s0, (uint)m30, hi3, lo3); mac128_small(s1, (uint)m31, hi3, lo3);
        mac128_small(s2, (uint)m32, hi3, lo3); mac128_small(s3, (uint)m33, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);

        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    } else {
        ulong hi0 = 0, lo0 = 0;
        mac128(s0, m00, hi0, lo0); mac128(s1, m01, hi0, lo0);
        mac128(s2, m02, hi0, lo0); mac128(s3, m03, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);

        ulong hi1 = 0, lo1 = 0;
        mac128(s0, m10, hi1, lo1); mac128(s1, m11, hi1, lo1);
        mac128(s2, m12, hi1, lo1); mac128(s3, m13, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);

        ulong hi2 = 0, lo2 = 0;
        mac128(s0, m20, hi2, lo2); mac128(s1, m21, hi2, lo2);
        mac128(s2, m22, hi2, lo2); mac128(s3, m23, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);

        ulong hi3 = 0, lo3 = 0;
        mac128(s0, m30, hi3, lo3); mac128(s1, m31, hi3, lo3);
        mac128(s2, m32, hi3, lo3); mac128(s3, m33, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);

        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }
}

inline void poseidon2_permute_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2], m03 = ext_mds[3];
    ulong m10 = ext_mds[4], m11 = ext_mds[5], m12 = ext_mds[6], m13 = ext_mds[7];
    ulong m20 = ext_mds[8], m21 = ext_mds[9], m22 = ext_mds[10], m23 = ext_mds[11];
    ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];

    ulong m_or = m00 | m01 | m02 | m03 | m10 | m11 | m12 | m13 | m20 | m21 | m22 | m23 | m30 | m31 | m32 | m33;
    bool mds_small = (m_or >> 32) == 0;

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];
    bool diag_small = ((d0 | d1 | d2 | d3) >> 32) == 0;

    mds_multiply_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33, mds_small);

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        mds_multiply_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33, mds_small);
    }

    if (diag_small) {
        uint d0_s = (uint)d0; uint d1_s = (uint)d1; uint d2_s = (uint)d2; uint d3_s = (uint)d3;
        #pragma unroll 4
        for (uint r = 0; r < 32; ++r) {
            if (r >= r_p) break;
            s0 = sbox7(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
            s0 = gold_mac_small(sum, s0, d0_s);
            s1 = gold_mac_small(sum, s1, d1_s);
            s2 = gold_mac_small(sum, s2, d2_s);
            s3 = gold_mac_small(sum, s3, d3_s);
        }
    } else {
        #pragma unroll 4
        for (uint r = 0; r < 32; ++r) {
            if (r >= r_p) break;
            s0 = sbox7(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
            s0 = gold_mac(sum, s0, d0);
            s1 = gold_mac(sum, s1, d1);
            s2 = gold_mac(sum, s2, d2);
            s3 = gold_mac(sum, s3, d3);
        }
    }

    uint rem_f = r_f - half_f;
    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= rem_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        mds_multiply_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33, mds_small);
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

    if (t <= 3) {
        ulong s0 = 0, s1 = 0, s2 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        
        poseidon2_permute_3(s0, s1, s2, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    } else {
        ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        if (arity > 3 && base + 3 < child_count) s3 = tree[in_offset + base + 3];
        
        poseidon2_permute_4(s0, s1, s2, s3, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    }
}