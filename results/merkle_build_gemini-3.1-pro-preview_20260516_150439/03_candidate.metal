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

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
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
    ulong new_lo = lo + l;
    hi += h + (new_lo < lo ? 1ul : 0ul);
    lo = new_lo;
}

inline ulong reduce128(ulong hi, ulong lo) {
    ulong hi_hi = hi >> 32;
    ulong hi_lo = (uint)hi;
    
    ulong sub = hi_hi + hi_lo;
    ulong t0 = lo - sub;
    ulong borrow = (lo < sub) ? 0xFFFFFFFFul : 0ul;
    t0 -= borrow;
    
    ulong t1 = hi_lo << 32;
    ulong t2 = t0 + t1;
    ulong carry = (t2 < t0) ? 0xFFFFFFFFul : 0ul;
    t2 += carry;
    
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

inline ulong mac_gold(ulong a, ulong b, ulong c) {
    ulong hi, lo;
    mul128(b, c, hi, lo);
    ulong new_lo = lo + a;
    hi += (new_lo < lo ? 1ul : 0ul);
    return reduce128(hi, new_lo);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
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

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    {
        ulong hi0, lo0;
        mul128(m00, s0, hi0, lo0); mac128(m01, s1, hi0, lo0); mac128(m02, s2, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1, lo1;
        mul128(m10, s0, hi1, lo1); mac128(m11, s1, hi1, lo1); mac128(m12, s2, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2, lo2;
        mul128(m20, s0, hi2, lo2); mac128(m21, s1, hi2, lo2); mac128(m22, s2, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    uint half_f = r_f >> 1;
    device const ulong *rc = rc_ext;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, *rc++));
        s1 = sbox7(gold_add(s1, *rc++));
        s2 = sbox7(gold_add(s2, *rc++));
        
        ulong hi0, lo0;
        mul128(m00, s0, hi0, lo0); mac128(m01, s1, hi0, lo0); mac128(m02, s2, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1, lo1;
        mul128(m10, s0, hi1, lo1); mac128(m11, s1, hi1, lo1); mac128(m12, s2, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2, lo2;
        mul128(m20, s0, hi2, lo2); mac128(m21, s1, hi2, lo2); mac128(m22, s2, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    device const ulong *rci = rc_int;
    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, *rci++));
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        s0 = mac_gold(sum, d0, s0);
        s1 = mac_gold(sum, d1, s1);
        s2 = mac_gold(sum, d2, s2);
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, *rc++));
        s1 = sbox7(gold_add(s1, *rc++));
        s2 = sbox7(gold_add(s2, *rc++));
        
        ulong hi0, lo0;
        mul128(m00, s0, hi0, lo0); mac128(m01, s1, hi0, lo0); mac128(m02, s2, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1, lo1;
        mul128(m10, s0, hi1, lo1); mac128(m11, s1, hi1, lo1); mac128(m12, s2, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2, lo2;
        mul128(m20, s0, hi2, lo2); mac128(m21, s1, hi2, lo2); mac128(m22, s2, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        s0 = n0; s1 = n1; s2 = n2;
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

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    {
        ulong hi0, lo0;
        mul128(m00, s0, hi0, lo0); mac128(m01, s1, hi0, lo0); mac128(m02, s2, hi0, lo0); mac128(m03, s3, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1, lo1;
        mul128(m10, s0, hi1, lo1); mac128(m11, s1, hi1, lo1); mac128(m12, s2, hi1, lo1); mac128(m13, s3, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2, lo2;
        mul128(m20, s0, hi2, lo2); mac128(m21, s1, hi2, lo2); mac128(m22, s2, hi2, lo2); mac128(m23, s3, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        ulong hi3, lo3;
        mul128(m30, s0, hi3, lo3); mac128(m31, s1, hi3, lo3); mac128(m32, s2, hi3, lo3); mac128(m33, s3, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    uint half_f = r_f >> 1;
    device const ulong *rc = rc_ext;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, *rc++));
        s1 = sbox7(gold_add(s1, *rc++));
        s2 = sbox7(gold_add(s2, *rc++));
        s3 = sbox7(gold_add(s3, *rc++));
        
        ulong hi0, lo0;
        mul128(m00, s0, hi0, lo0); mac128(m01, s1, hi0, lo0); mac128(m02, s2, hi0, lo0); mac128(m03, s3, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1, lo1;
        mul128(m10, s0, hi1, lo1); mac128(m11, s1, hi1, lo1); mac128(m12, s2, hi1, lo1); mac128(m13, s3, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2, lo2;
        mul128(m20, s0, hi2, lo2); mac128(m21, s1, hi2, lo2); mac128(m22, s2, hi2, lo2); mac128(m23, s3, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        ulong hi3, lo3;
        mul128(m30, s0, hi3, lo3); mac128(m31, s1, hi3, lo3); mac128(m32, s2, hi3, lo3); mac128(m33, s3, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    device const ulong *rci = rc_int;
    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, *rci++));
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        s0 = mac_gold(sum, d0, s0);
        s1 = mac_gold(sum, d1, s1);
        s2 = mac_gold(sum, d2, s2);
        s3 = mac_gold(sum, d3, s3);
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, *rc++));
        s1 = sbox7(gold_add(s1, *rc++));
        s2 = sbox7(gold_add(s2, *rc++));
        s3 = sbox7(gold_add(s3, *rc++));
        
        ulong hi0, lo0;
        mul128(m00, s0, hi0, lo0); mac128(m01, s1, hi0, lo0); mac128(m02, s2, hi0, lo0); mac128(m03, s3, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1, lo1;
        mul128(m10, s0, hi1, lo1); mac128(m11, s1, hi1, lo1); mac128(m12, s2, hi1, lo1); mac128(m13, s3, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2, lo2;
        mul128(m20, s0, hi2, lo2); mac128(m21, s1, hi2, lo2); mac128(m22, s2, hi2, lo2); mac128(m23, s3, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        ulong hi3, lo3;
        mul128(m30, s0, hi3, lo3); mac128(m31, s1, hi3, lo3); mac128(m32, s2, hi3, lo3); mac128(m33, s3, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
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