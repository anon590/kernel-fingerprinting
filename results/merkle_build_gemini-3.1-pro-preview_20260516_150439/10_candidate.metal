#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline void sqr128(ulong a, thread ulong &hi, thread ulong &lo) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);

    ulong p01 = (ulong)a0 * a1;
    ulong p00 = (ulong)a0 * a0;
    ulong p11 = (ulong)a1 * a1;

    ulong p01_lo = (uint)p01;
    ulong p01_hi = (p01 >> 32);

    ulong mid = (p00 >> 32) + (p01_lo << 1);
    lo = (uint)p00 | (mid << 32);
    hi = p11 + (p01_hi << 1) + (mid >> 32);
}

inline void mac128_pre(uint a0, uint a1, uint b0, uint b1, thread ulong &hi, thread ulong &lo) {
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00 = (ulong)a0 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong l = (uint)p00 | (mid << 32);
    ulong h = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    ulong new_lo = lo + l;
    hi += h + (new_lo < lo ? 1ul : 0ul);
    lo = new_lo;
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

inline ulong gold_mul_pre(uint a0, uint a1, ulong b) {
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00 = (ulong)a0 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo = (uint)p00 | (mid << 32);
    ulong hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    return reduce128(hi, lo);
}

inline ulong gold_sqr(ulong a) {
    ulong hi, lo;
    sqr128(a, hi, lo);
    return reduce128(hi, lo);
}

inline ulong gold_mul(ulong a, ulong b) {
    return gold_mul_pre((uint)a, (uint)(a >> 32), b);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matmul_3(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread const uint2 *m_halves) {
    uint s0_0 = (uint)s0, s0_1 = (uint)(s0 >> 32);
    uint s1_0 = (uint)s1, s1_1 = (uint)(s1 >> 32);
    uint s2_0 = (uint)s2, s2_1 = (uint)(s2 >> 32);

    ulong hi0 = 0, lo0 = 0;
    mac128_pre(m_halves[0].x, m_halves[0].y, s0_0, s0_1, hi0, lo0);
    mac128_pre(m_halves[1].x, m_halves[1].y, s1_0, s1_1, hi0, lo0);
    mac128_pre(m_halves[2].x, m_halves[2].y, s2_0, s2_1, hi0, lo0);
    ulong n0 = reduce128(hi0, lo0);

    ulong hi1 = 0, lo1 = 0;
    mac128_pre(m_halves[3].x, m_halves[3].y, s0_0, s0_1, hi1, lo1);
    mac128_pre(m_halves[4].x, m_halves[4].y, s1_0, s1_1, hi1, lo1);
    mac128_pre(m_halves[5].x, m_halves[5].y, s2_0, s2_1, hi1, lo1);
    ulong n1 = reduce128(hi1, lo1);

    ulong hi2 = 0, lo2 = 0;
    mac128_pre(m_halves[6].x, m_halves[6].y, s0_0, s0_1, hi2, lo2);
    mac128_pre(m_halves[7].x, m_halves[7].y, s1_0, s1_1, hi2, lo2);
    mac128_pre(m_halves[8].x, m_halves[8].y, s2_0, s2_1, hi2, lo2);
    ulong n2 = reduce128(hi2, lo2);

    s0 = n0; s1 = n1; s2 = n2;
}

inline void poseidon2_permute_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    uint2 m_halves[9];
    for (int i = 0; i < 9; ++i) {
        ulong m = ext_mds[i];
        m_halves[i] = uint2((uint)m, (uint)(m >> 32));
    }
    
    uint2 d_halves[3];
    for (int i = 0; i < 3; ++i) {
        ulong d = int_diag[i];
        d_halves[i] = uint2((uint)d, (uint)(d >> 32));
    }

    matmul_3(s0, s1, s2, m_halves);

    uint half_f = r_f >> 1;
    uint second_half_f = r_f - half_f;

    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        uint idx = r * 3;
        s0 = sbox7(gold_add(s0, rc_ext[idx]));
        s1 = sbox7(gold_add(s1, rc_ext[idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[idx + 2]));
        matmul_3(s0, s1, s2, m_halves);
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        ulong n0 = gold_add(sum, gold_mul_pre(d_halves[0].x, d_halves[0].y, s0));
        ulong n1 = gold_add(sum, gold_mul_pre(d_halves[1].x, d_halves[1].y, s1));
        ulong n2 = gold_add(sum, gold_mul_pre(d_halves[2].x, d_halves[2].y, s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        if (r >= second_half_f) break;
        uint idx = (half_f + r) * 3;
        s0 = sbox7(gold_add(s0, rc_ext[idx]));
        s1 = sbox7(gold_add(s1, rc_ext[idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[idx + 2]));
        matmul_3(s0, s1, s2, m_halves);
    }
}

inline void matmul_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3, thread const uint2 *m_halves) {
    uint s0_0 = (uint)s0, s0_1 = (uint)(s0 >> 32);
    uint s1_0 = (uint)s1, s1_1 = (uint)(s1 >> 32);
    uint s2_0 = (uint)s2, s2_1 = (uint)(s2 >> 32);
    uint s3_0 = (uint)s3, s3_1 = (uint)(s3 >> 32);

    ulong hi0 = 0, lo0 = 0;
    mac128_pre(m_halves[0].x,  m_halves[0].y,  s0_0, s0_1, hi0, lo0);
    mac128_pre(m_halves[1].x,  m_halves[1].y,  s1_0, s1_1, hi0, lo0);
    mac128_pre(m_halves[2].x,  m_halves[2].y,  s2_0, s2_1, hi0, lo0);
    mac128_pre(m_halves[3].x,  m_halves[3].y,  s3_0, s3_1, hi0, lo0);
    ulong n0 = reduce128(hi0, lo0);

    ulong hi1 = 0, lo1 = 0;
    mac128_pre(m_halves[4].x,  m_halves[4].y,  s0_0, s0_1, hi1, lo1);
    mac128_pre(m_halves[5].x,  m_halves[5].y,  s1_0, s1_1, hi1, lo1);
    mac128_pre(m_halves[6].x,  m_halves[6].y,  s2_0, s2_1, hi1, lo1);
    mac128_pre(m_halves[7].x,  m_halves[7].y,  s3_0, s3_1, hi1, lo1);
    ulong n1 = reduce128(hi1, lo1);

    ulong hi2 = 0, lo2 = 0;
    mac128_pre(m_halves[8].x,  m_halves[8].y,  s0_0, s0_1, hi2, lo2);
    mac128_pre(m_halves[9].x,  m_halves[9].y,  s1_0, s1_1, hi2, lo2);
    mac128_pre(m_halves[10].x, m_halves[10].y, s2_0, s2_1, hi2, lo2);
    mac128_pre(m_halves[11].x, m_halves[11].y, s3_0, s3_1, hi2, lo2);
    ulong n2 = reduce128(hi2, lo2);

    ulong hi3 = 0, lo3 = 0;
    mac128_pre(m_halves[12].x, m_halves[12].y, s0_0, s0_1, hi3, lo3);
    mac128_pre(m_halves[13].x, m_halves[13].y, s1_0, s1_1, hi3, lo3);
    mac128_pre(m_halves[14].x, m_halves[14].y, s2_0, s2_1, hi3, lo3);
    mac128_pre(m_halves[15].x, m_halves[15].y, s3_0, s3_1, hi3, lo3);
    ulong n3 = reduce128(hi3, lo3);

    s0 = n0; s1 = n1; s2 = n2; s3 = n3;
}

inline void poseidon2_permute_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    uint2 m_halves[16];
    for (int i = 0; i < 16; ++i) {
        ulong m = ext_mds[i];
        m_halves[i] = uint2((uint)m, (uint)(m >> 32));
    }
    
    uint2 d_halves[4];
    for (int i = 0; i < 4; ++i) {
        ulong d = int_diag[i];
        d_halves[i] = uint2((uint)d, (uint)(d >> 32));
    }

    matmul_4(s0, s1, s2, s3, m_halves);

    uint half_f = r_f >> 1;
    uint second_half_f = r_f - half_f;

    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        if (r >= half_f) break;
        uint idx = r * 4;
        s0 = sbox7(gold_add(s0, rc_ext[idx]));
        s1 = sbox7(gold_add(s1, rc_ext[idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[idx + 3]));
        matmul_4(s0, s1, s2, s3, m_halves);
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        ulong n0 = gold_add(sum, gold_mul_pre(d_halves[0].x, d_halves[0].y, s0));
        ulong n1 = gold_add(sum, gold_mul_pre(d_halves[1].x, d_halves[1].y, s1));
        ulong n2 = gold_add(sum, gold_mul_pre(d_halves[2].x, d_halves[2].y, s2));
        ulong n3 = gold_add(sum, gold_mul_pre(d_halves[3].x, d_halves[3].y, s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        if (r >= second_half_f) break;
        uint idx = (half_f + r) * 4;
        s0 = sbox7(gold_add(s0, rc_ext[idx]));
        s1 = sbox7(gold_add(s1, rc_ext[idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[idx + 3]));
        matmul_4(s0, s1, s2, s3, m_halves);
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