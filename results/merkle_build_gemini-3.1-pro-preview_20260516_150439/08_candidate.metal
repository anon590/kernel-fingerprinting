#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

struct U64Split {
    uint lo;
    uint hi;
};

inline U64Split split(ulong a) {
    return { (uint)a, (uint)(a >> 32) };
}

inline void mul128_split(U64Split a, U64Split b, thread ulong &hi, thread ulong &lo) {
    ulong p01 = (ulong)a.lo * b.hi;
    ulong p10 = (ulong)a.hi * b.lo;
    ulong p00 = (ulong)a.lo * b.lo;
    ulong p11 = (ulong)a.hi * b.hi;

    ulong mid = (p00 >> 32) + (ulong)(uint)p01 + (ulong)(uint)p10;
    lo = (uint)p00 | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

inline void mac128_split(U64Split a, U64Split b, thread ulong &hi, thread ulong &lo) {
    ulong h, l;
    mul128_split(a, b, h, l);
    lo += l;
    hi += h + (ulong)(lo < l);
}

inline void sqr128(ulong a, thread ulong &hi, thread ulong &lo) {
    U64Split sa = split(a);
    ulong p01 = (ulong)sa.lo * sa.hi;
    ulong p00 = (ulong)sa.lo * sa.lo;
    ulong p11 = (ulong)sa.hi * sa.hi;

    ulong mid = (p00 >> 32) + ((ulong)(uint)p01 << 1);
    lo = (uint)p00 | (mid << 32);
    hi = p11 + ((p01 >> 32) << 1) + (mid >> 32);
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

inline ulong gold_mul_split(U64Split a, U64Split b) {
    ulong hi, lo;
    mul128_split(a, b, hi, lo);
    return reduce128(hi, lo);
}

inline ulong gold_mul(ulong a, ulong b) {
    return gold_mul_split(split(a), split(b));
}

inline ulong gold_sqr(ulong a) {
    ulong hi, lo;
    sqr128(a, hi, lo);
    return reduce128(hi, lo);
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
    U64Split m00 = split(ext_mds[0]), m01 = split(ext_mds[1]), m02 = split(ext_mds[2]);
    U64Split m10 = split(ext_mds[3]), m11 = split(ext_mds[4]), m12 = split(ext_mds[5]);
    U64Split m20 = split(ext_mds[6]), m21 = split(ext_mds[7]), m22 = split(ext_mds[8]);

    U64Split d0 = split(int_diag[0]), d1 = split(int_diag[1]), d2 = split(int_diag[2]);

    {
        U64Split sp0 = split(s0);
        U64Split sp1 = split(s1);
        U64Split sp2 = split(s2);

        ulong hi0 = 0, lo0 = 0;
        mac128_split(m00, sp0, hi0, lo0);
        mac128_split(m01, sp1, hi0, lo0);
        mac128_split(m02, sp2, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1 = 0, lo1 = 0;
        mac128_split(m10, sp0, hi1, lo1);
        mac128_split(m11, sp1, hi1, lo1);
        mac128_split(m12, sp2, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2 = 0, lo2 = 0;
        mac128_split(m20, sp0, hi2, lo2);
        mac128_split(m21, sp1, hi2, lo2);
        mac128_split(m22, sp2, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        
        U64Split sp0 = split(s0);
        U64Split sp1 = split(s1);
        U64Split sp2 = split(s2);

        ulong hi0 = 0, lo0 = 0;
        mac128_split(m00, sp0, hi0, lo0);
        mac128_split(m01, sp1, hi0, lo0);
        mac128_split(m02, sp2, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1 = 0, lo1 = 0;
        mac128_split(m10, sp0, hi1, lo1);
        mac128_split(m11, sp1, hi1, lo1);
        mac128_split(m12, sp2, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2 = 0, lo2 = 0;
        mac128_split(m20, sp0, hi2, lo2);
        mac128_split(m21, sp1, hi2, lo2);
        mac128_split(m22, sp2, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        ulong n0 = gold_add(sum, gold_mul_split(d0, split(s0)));
        ulong n1 = gold_add(sum, gold_mul_split(d1, split(s1)));
        ulong n2 = gold_add(sum, gold_mul_split(d2, split(s2)));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        
        U64Split sp0 = split(s0);
        U64Split sp1 = split(s1);
        U64Split sp2 = split(s2);

        ulong hi0 = 0, lo0 = 0;
        mac128_split(m00, sp0, hi0, lo0);
        mac128_split(m01, sp1, hi0, lo0);
        mac128_split(m02, sp2, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1 = 0, lo1 = 0;
        mac128_split(m10, sp0, hi1, lo1);
        mac128_split(m11, sp1, hi1, lo1);
        mac128_split(m12, sp2, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2 = 0, lo2 = 0;
        mac128_split(m20, sp0, hi2, lo2);
        mac128_split(m21, sp1, hi2, lo2);
        mac128_split(m22, sp2, hi2, lo2);
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
    U64Split m00 = split(ext_mds[0]), m01 = split(ext_mds[1]), m02 = split(ext_mds[2]), m03 = split(ext_mds[3]);
    U64Split m10 = split(ext_mds[4]), m11 = split(ext_mds[5]), m12 = split(ext_mds[6]), m13 = split(ext_mds[7]);
    U64Split m20 = split(ext_mds[8]), m21 = split(ext_mds[9]), m22 = split(ext_mds[10]), m23 = split(ext_mds[11]);
    U64Split m30 = split(ext_mds[12]), m31 = split(ext_mds[13]), m32 = split(ext_mds[14]), m33 = split(ext_mds[15]);

    U64Split d0 = split(int_diag[0]), d1 = split(int_diag[1]), d2 = split(int_diag[2]), d3 = split(int_diag[3]);

    {
        U64Split sp0 = split(s0);
        U64Split sp1 = split(s1);
        U64Split sp2 = split(s2);
        U64Split sp3 = split(s3);

        ulong hi0 = 0, lo0 = 0;
        mac128_split(m00, sp0, hi0, lo0);
        mac128_split(m01, sp1, hi0, lo0);
        mac128_split(m02, sp2, hi0, lo0);
        mac128_split(m03, sp3, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1 = 0, lo1 = 0;
        mac128_split(m10, sp0, hi1, lo1);
        mac128_split(m11, sp1, hi1, lo1);
        mac128_split(m12, sp2, hi1, lo1);
        mac128_split(m13, sp3, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2 = 0, lo2 = 0;
        mac128_split(m20, sp0, hi2, lo2);
        mac128_split(m21, sp1, hi2, lo2);
        mac128_split(m22, sp2, hi2, lo2);
        mac128_split(m23, sp3, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        ulong hi3 = 0, lo3 = 0;
        mac128_split(m30, sp0, hi3, lo3);
        mac128_split(m31, sp1, hi3, lo3);
        mac128_split(m32, sp2, hi3, lo3);
        mac128_split(m33, sp3, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        
        U64Split sp0 = split(s0);
        U64Split sp1 = split(s1);
        U64Split sp2 = split(s2);
        U64Split sp3 = split(s3);

        ulong hi0 = 0, lo0 = 0;
        mac128_split(m00, sp0, hi0, lo0);
        mac128_split(m01, sp1, hi0, lo0);
        mac128_split(m02, sp2, hi0, lo0);
        mac128_split(m03, sp3, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1 = 0, lo1 = 0;
        mac128_split(m10, sp0, hi1, lo1);
        mac128_split(m11, sp1, hi1, lo1);
        mac128_split(m12, sp2, hi1, lo1);
        mac128_split(m13, sp3, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2 = 0, lo2 = 0;
        mac128_split(m20, sp0, hi2, lo2);
        mac128_split(m21, sp1, hi2, lo2);
        mac128_split(m22, sp2, hi2, lo2);
        mac128_split(m23, sp3, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        ulong hi3 = 0, lo3 = 0;
        mac128_split(m30, sp0, hi3, lo3);
        mac128_split(m31, sp1, hi3, lo3);
        mac128_split(m32, sp2, hi3, lo3);
        mac128_split(m33, sp3, hi3, lo3);
        ulong n3 = reduce128(hi3, lo3);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        ulong n0 = gold_add(sum, gold_mul_split(d0, split(s0)));
        ulong n1 = gold_add(sum, gold_mul_split(d1, split(s1)));
        ulong n2 = gold_add(sum, gold_mul_split(d2, split(s2)));
        ulong n3 = gold_add(sum, gold_mul_split(d3, split(s3)));
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        
        U64Split sp0 = split(s0);
        U64Split sp1 = split(s1);
        U64Split sp2 = split(s2);
        U64Split sp3 = split(s3);

        ulong hi0 = 0, lo0 = 0;
        mac128_split(m00, sp0, hi0, lo0);
        mac128_split(m01, sp1, hi0, lo0);
        mac128_split(m02, sp2, hi0, lo0);
        mac128_split(m03, sp3, hi0, lo0);
        ulong n0 = reduce128(hi0, lo0);
        
        ulong hi1 = 0, lo1 = 0;
        mac128_split(m10, sp0, hi1, lo1);
        mac128_split(m11, sp1, hi1, lo1);
        mac128_split(m12, sp2, hi1, lo1);
        mac128_split(m13, sp3, hi1, lo1);
        ulong n1 = reduce128(hi1, lo1);
        
        ulong hi2 = 0, lo2 = 0;
        mac128_split(m20, sp0, hi2, lo2);
        mac128_split(m21, sp1, hi2, lo2);
        mac128_split(m22, sp2, hi2, lo2);
        mac128_split(m23, sp3, hi2, lo2);
        ulong n2 = reduce128(hi2, lo2);
        
        ulong hi3 = 0, lo3 = 0;
        mac128_split(m30, sp0, hi3, lo3);
        mac128_split(m31, sp1, hi3, lo3);
        mac128_split(m32, sp2, hi3, lo3);
        mac128_split(m33, sp3, hi3, lo3);
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