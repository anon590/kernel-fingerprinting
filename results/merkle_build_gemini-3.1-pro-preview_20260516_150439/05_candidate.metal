#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline void mul128(ulong a, ulong b, thread ulong &hi, thread ulong &lo) {
    lo = a * b;
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);
    
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00_hi = ((ulong)a0 * b0) >> 32;
    
    ulong mid_hi = (p00_hi + (ulong)(uint)p01 + (ulong)(uint)p10) >> 32;
    hi = (ulong)a1 * b1 + (p01 >> 32) + (p10 >> 32) + mid_hi;
}

inline void sqr128(ulong a, thread ulong &hi, thread ulong &lo) {
    lo = a * a;
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    
    ulong p01 = (ulong)a0 * a1;
    ulong p00_hi = ((ulong)a0 * a0) >> 32;
    
    ulong mid_hi = (p00_hi + ((ulong)(uint)p01 << 1)) >> 32;
    hi = (ulong)a1 * a1 + ((p01 >> 32) << 1) + mid_hi;
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

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline ulong mac_reduce128(ulong sum, ulong a, ulong b) {
    ulong h, l;
    mul128(a, b, h, l);
    ulong ll = l + sum;
    ulong hh = h + (ll < l ? 1ul : 0ul);
    return reduce128(hh, ll);
}

inline ulong dot3(ulong m0, ulong s0, ulong m1, ulong s1, ulong m2, ulong s2) {
    ulong h0, l0, h1, l1, h2, l2;
    mul128(m0, s0, h0, l0);
    mul128(m1, s1, h1, l1);
    mul128(m2, s2, h2, l2);

    ulong l_01 = l0 + l1;
    ulong c_01 = (l_01 < l0) ? 1ul : 0ul;
    ulong l_012 = l_01 + l2;
    ulong c_012 = (l_012 < l_01) ? 1ul : 0ul;
    
    ulong h = h0 + h1 + h2 + c_01 + c_012;
    return reduce128(h, l_012);
}

inline ulong dot4(ulong m0, ulong s0, ulong m1, ulong s1, ulong m2, ulong s2, ulong m3, ulong s3) {
    ulong h0, l0, h1, l1, h2, l2, h3, l3;
    mul128(m0, s0, h0, l0);
    mul128(m1, s1, h1, l1);
    mul128(m2, s2, h2, l2);
    mul128(m3, s3, h3, l3);

    ulong l_01 = l0 + l1;
    ulong c_01 = (l_01 < l0) ? 1ul : 0ul;
    ulong l_23 = l2 + l3;
    ulong c_23 = (l_23 < l2) ? 1ul : 0ul;
    
    ulong l = l_01 + l_23;
    ulong c = (l < l_01) ? 1ul : 0ul;
    
    ulong h = h0 + h1 + h2 + h3 + c_01 + c_23 + c;
    return reduce128(h, l);
}

inline void apply_mds_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                        ulong m00, ulong m01, ulong m02,
                        ulong m10, ulong m11, ulong m12,
                        ulong m20, ulong m21, ulong m22) {
    ulong n0 = dot3(m00, s0, m01, s1, m02, s2);
    ulong n1 = dot3(m10, s0, m11, s1, m12, s2);
    ulong n2 = dot3(m20, s0, m21, s1, m22, s2);
    s0 = n0; s1 = n1; s2 = n2;
}

inline void apply_mds_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                        ulong m00, ulong m01, ulong m02, ulong m03,
                        ulong m10, ulong m11, ulong m12, ulong m13,
                        ulong m20, ulong m21, ulong m22, ulong m23,
                        ulong m30, ulong m31, ulong m32, ulong m33) {
    ulong n0 = dot4(m00, s0, m01, s1, m02, s2, m03, s3);
    ulong n1 = dot4(m10, s0, m11, s1, m12, s2, m13, s3);
    ulong n2 = dot4(m20, s0, m21, s1, m22, s2, m23, s3);
    ulong n3 = dot4(m30, s0, m31, s1, m32, s2, m33, s3);
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
    uint rc_idx = 0;

    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        apply_mds_3(s0, s1, s2, m00, m01, m02, m10, m11, m12, m20, m21, m22);
    }

    #pragma unroll 4
    for (uint r = 0; r < 32; ++r) {
        if (r >= r_p) break;
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        s0 = mac_reduce128(sum, d0, s0);
        s1 = mac_reduce128(sum, d1, s1);
        s2 = mac_reduce128(sum, d2, s2);
    }

    uint rem_f = r_f - half_f;
    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= rem_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
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
    uint rc_idx = 0;

    #pragma unroll 4
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        apply_mds_4(s0, s1, s2, s3, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);
    }

    #pragma unroll 4
    for (uint r = 0; r < 32; ++r) {
        if (r >= r_p) break;
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        s0 = mac_reduce128(sum, d0, s0);
        s1 = mac_reduce128(sum, d1, s1);
        s2 = mac_reduce128(sum, d2, s2);
        s3 = mac_reduce128(sum, d3, s3);
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