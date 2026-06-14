I've optimized the kernel in several key ways:
1. **Parallel 128-bit Additions**: Replaced sequential `mac128` chains in the MDS matrix multiplication with parallel reduction trees (`mul_add3` and `mul_add4`). This breaks loop-carried dependencies and maps perfectly to ARM64 `adds` and `adcs` instructions.
2. **Fused MAC-Sum**: In the internal partial rounds, the formula `n_i = sum + d_i * s_i` previously required a full Goldilocks multiplication followed by a modular addition. I fused this into a single `mac_sum` function that accumulates the product and the sum into a 128-bit register *before* performing a single, unified 128-bit reduction. This saves ~100 heavy reductions per permutation!
3. **Faster Squaring**: Optimized `sqr128` by extracting the low and high 32-bits of the cross-product and using shifts instead of multiple 64-bit additions.
4. **Pointer Arithmetic**: Used pointer increments for round constants instead of recalculating index offsets in the loops.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline void mul128(ulong a, ulong b, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    lo = (uint)p00 | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

inline void sqr128(ulong a, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    ulong p00 = (ulong)a0 * a0;
    ulong p11 = (ulong)a1 * a1;
    ulong p01 = (ulong)a0 * a1;

    ulong p01_lo = (uint)p01;
    ulong p01_hi = p01 >> 32;
    
    ulong mid = (p00 >> 32) + (p01_lo << 1);
    lo = (uint)p00 | (mid << 32);
    hi = p11 + (p01_hi << 1) + (mid >> 32);
}

inline ulong reduce128(ulong hi, ulong lo) {
    ulong hi_lo = (uint)hi;
    ulong hi_hi = hi >> 32;
    
    ulong sub = hi_hi + hi_lo;
    ulong t0 = lo - sub;
    t0 -= (lo < sub) ? 0xFFFFFFFFul : 0ul;
    
    ulong t1 = hi_lo << 32;
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;
    
    return (t2 >= P_GOLD) ? t2 - P_GOLD : t2;
}

inline void mul_add3(ulong m0, ulong s0, ulong m1, ulong s1, ulong m2, ulong s2, thread ulong &hi, thread ulong &lo) {
    ulong h0, l0, h1, l1, h2, l2;
    mul128(m0, s0, h0, l0);
    mul128(m1, s1, h1, l1);
    mul128(m2, s2, h2, l2);
    
    ulong l_01 = l0 + l1;
    ulong h_01 = h0 + h1 + (l_01 < l0 ? 1ul : 0ul);
    
    lo = l_01 + l2;
    hi = h_01 + h2 + (lo < l_01 ? 1ul : 0ul);
}

inline void mul_add4(ulong m0, ulong s0, ulong m1, ulong s1, ulong m2, ulong s2, ulong m3, ulong s3, thread ulong &hi, thread ulong &lo) {
    ulong h0, l0, h1, l1, h2, l2, h3, l3;
    mul128(m0, s0, h0, l0);
    mul128(m1, s1, h1, l1);
    mul128(m2, s2, h2, l2);
    mul128(m3, s3, h3, l3);
    
    ulong l_01 = l0 + l1;
    ulong h_01 = h0 + h1 + (l_01 < l0 ? 1ul : 0ul);
    
    ulong l_23 = l2 + l3;
    ulong h_23 = h2 + h3 + (l_23 < l2 ? 1ul : 0ul);
    
    lo = l_01 + l_23;
    hi = h_01 + h_23 + (lo < l_01 ? 1ul : 0ul);
}

inline ulong mac_sum(ulong sum, ulong d, ulong s) {
    ulong h, l;
    mul128(d, s, h, l);
    ulong lo = sum + l;
    ulong hi = h + (lo < l ? 1ul : 0ul);
    return reduce128(hi, lo);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? t - P_GOLD : t;
}

inline ulong sbox7(ulong x) {
    ulong hi2, lo2;
    sqr128(x, hi2, lo2);
    ulong x2 = reduce128(hi2, lo2);
    
    ulong hi4, lo4;
    sqr128(x2, hi4, lo4);
    ulong x4 = reduce128(hi4, lo4);
    
    ulong hi6, lo6;
    mul128(x4, x2, hi6, lo6);
    ulong x6 = reduce128(hi6, lo6);
    
    ulong hi7, lo7;
    mul128(x6, x, hi7, lo7);
    return reduce128(hi7, lo7);
}

inline void apply_mds_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                        ulong m00, ulong m01, ulong m02,
                        ulong m10, ulong m11, ulong m12,
                        ulong m20, ulong m21, ulong m22) {
    ulong h0, l0;
    mul_add3(m00, s0, m01, s1, m02, s2, h0, l0);
    ulong n0 = reduce128(h0, l0);
    
    ulong h1, l1;
    mul_add3(m10, s0, m11, s1, m12, s2, h1, l1);
    ulong n1 = reduce128(h1, l1);
    
    ulong h2, l2;
    mul_add3(m20, s0, m21, s1, m22, s2, h2, l2);
    ulong n2 = reduce128(h2, l2);
    
    s0 = n0; s1 = n1; s2 = n2;
}

inline void apply_mds_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                        ulong m00, ulong m01, ulong m02, ulong m03,
                        ulong m10, ulong m11, ulong m12, ulong m13,
                        ulong m20, ulong m21, ulong m22, ulong m23,
                        ulong m30, ulong m31, ulong m32, ulong m33) {
    ulong h0, l0;
    mul_add4(m00, s0, m01, s1, m02, s2, m03, s3, h0, l0);
    ulong n0 = reduce128(h0, l0);
    
    ulong h1, l1;
    mul_add4(m10, s0, m11, s1, m12, s2, m13, s3, h1, l1);
    ulong n1 = reduce128(h1, l1);
    
    ulong h2, l2;
    mul_add4(m20, s0, m21, s1, m22, s2, m23, s3, h2, l2);
    ulong n2 = reduce128(h2, l2);
    
    ulong h3, l3;
    mul_add4(m30, s0, m31, s1, m32, s2, m33, s3, h3, l3);
    ulong n3 = reduce128(h3, l3);
    
    s0 = n0; s1 = n1; s2 = n2; s3 = n3;
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

    apply_mds_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22);

    uint half_f = r_f >> 1;
    device const ulong *rc = rc_ext;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc[0]));
        s1 = sbox7(gold_add(s1, rc[1]));
        s2 = sbox7(gold_add(s2, rc[2]));
        rc += 3;
        apply_mds_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22);
    }

    device const ulong *rc_i = rc_int;
    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_i[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        ulong n0 = mac_sum(sum, d0, s0);
        ulong n1 = mac_sum(sum, d1, s1);
        ulong n2 = mac_sum(sum, d2, s2);
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc[0]));
        s1 = sbox7(gold_add(s1, rc[1]));
        s2 = sbox7(gold_add(s2, rc[2]));
        rc += 3;
        apply_mds_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22);
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

    apply_mds_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);

    uint half_f = r_f >> 1;
    device const ulong *rc = rc_ext;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc[0]));
        s1 = sbox7(gold_add(s1, rc[1]));
        s2 = sbox7(gold_add(s2, rc[2]));
        s3 = sbox7(gold_add(s3, rc[3]));
        rc += 4;
        apply_mds_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);
    }

    device const ulong *rc_i = rc_int;
    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_i[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        ulong n0 = mac_sum(sum, d0, s0);
        ulong n1 = mac_sum(sum, d1, s1);
        ulong n2 = mac_sum(sum, d2, s2);
        ulong n3 = mac_sum(sum, d3, s3);
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc[0]));
        s1 = sbox7(gold_add(s1, rc[1]));
        s2 = sbox7(gold_add(s2, rc[2]));
        s3 = sbox7(gold_add(s3, rc[3]));
        rc += 4;
        apply_mds_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);
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
        uint n_children = min(arity, child_count - base);
        
        if (n_children > 0) s0 = tree[in_offset + base + 0];
        if (n_children > 1) s1 = tree[in_offset + base + 1];
        if (n_children > 2) s2 = tree[in_offset + base + 2];
        
        poseidon2_permute_3(s0, s1, s2, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    } else {
        ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint base = p * arity;
        uint n_children = min(arity, child_count - base);
        
        if (n_children > 0) s0 = tree[in_offset + base + 0];
        if (n_children > 1) s1 = tree[in_offset + base + 1];
        if (n_children > 2) s2 = tree[in_offset + base + 2];
        if (n_children > 3) s3 = tree[in_offset + base + 3];
        
        poseidon2_permute_4(s0, s1, s2, s3, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    }
}
```