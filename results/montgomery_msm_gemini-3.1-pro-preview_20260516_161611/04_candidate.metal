#include <metal_stdlib>
using namespace metal;

struct Fp {
    ulong x0, x1, x2, x3, x4, x5;
};

struct Point {
    Fp x, y, z;
};

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul);
    ulong lo  = (p00 & 0xFFFFFFFFul) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong2 fma_add128(ulong a, ulong b, ulong t, ulong c) {
    ulong2 prod = umul128(a, b);
    ulong lo1 = prod.x + t;
    ulong cy1 = (lo1 < prod.x) ? 1ul : 0ul;
    ulong hi1 = prod.y + cy1;
    ulong lo2 = lo1 + c;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    ulong hi2 = hi1 + cy2;
    return ulong2(lo2, hi2);
}

inline bool fp_is_zero(Fp a) {
    return (a.x0 | a.x1 | a.x2 | a.x3 | a.x4 | a.x5) == 0ul;
}

inline bool fp_eq(Fp a, Fp b) {
    return ((a.x0 ^ b.x0) | (a.x1 ^ b.x1) | (a.x2 ^ b.x2) |
            (a.x3 ^ b.x3) | (a.x4 ^ b.x4) | (a.x5 ^ b.x5)) == 0ul;
}

inline Fp fp_add(Fp a, Fp b, Fp q) {
    ulong carry = 0ul;
    Fp sum;
    ulong s, t, cy1, cy2;
    
    s = a.x0 + carry; cy1 = (s < a.x0) ? 1ul:0ul; t = s + b.x0; cy2 = (t < s) ? 1ul:0ul; sum.x0 = t; carry = cy1 + cy2;
    s = a.x1 + carry; cy1 = (s < a.x1) ? 1ul:0ul; t = s + b.x1; cy2 = (t < s) ? 1ul:0ul; sum.x1 = t; carry = cy1 + cy2;
    s = a.x2 + carry; cy1 = (s < a.x2) ? 1ul:0ul; t = s + b.x2; cy2 = (t < s) ? 1ul:0ul; sum.x2 = t; carry = cy1 + cy2;
    s = a.x3 + carry; cy1 = (s < a.x3) ? 1ul:0ul; t = s + b.x3; cy2 = (t < s) ? 1ul:0ul; sum.x3 = t; carry = cy1 + cy2;
    s = a.x4 + carry; cy1 = (s < a.x4) ? 1ul:0ul; t = s + b.x4; cy2 = (t < s) ? 1ul:0ul; sum.x4 = t; carry = cy1 + cy2;
    s = a.x5 + carry; cy1 = (s < a.x5) ? 1ul:0ul; t = s + b.x5; cy2 = (t < s) ? 1ul:0ul; sum.x5 = t; carry = cy1 + cy2;

    ulong borrow = 0ul;
    Fp diff;
    ulong tv, b1, d, b2;
    tv = sum.x0 - q.x0; b1 = (tv > sum.x0) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x0 = d; borrow = b1 + b2;
    tv = sum.x1 - q.x1; b1 = (tv > sum.x1) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x1 = d; borrow = b1 + b2;
    tv = sum.x2 - q.x2; b1 = (tv > sum.x2) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x2 = d; borrow = b1 + b2;
    tv = sum.x3 - q.x3; b1 = (tv > sum.x3) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x3 = d; borrow = b1 + b2;
    tv = sum.x4 - q.x4; b1 = (tv > sum.x4) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x4 = d; borrow = b1 + b2;
    tv = sum.x5 - q.x5; b1 = (tv > sum.x5) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x5 = d; borrow = b1 + b2;

    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    Fp res;
    res.x0 = use_diff ? diff.x0 : sum.x0;
    res.x1 = use_diff ? diff.x1 : sum.x1;
    res.x2 = use_diff ? diff.x2 : sum.x2;
    res.x3 = use_diff ? diff.x3 : sum.x3;
    res.x4 = use_diff ? diff.x4 : sum.x4;
    res.x5 = use_diff ? diff.x5 : sum.x5;
    return res;
}

inline Fp fp_sub(Fp a, Fp b, Fp q) {
    Fp diff;
    ulong borrow = 0ul;
    ulong tv, b1, d, b2;
    
    tv = a.x0 - b.x0; b1 = (tv > a.x0) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x0 = d; borrow = b1 + b2;
    tv = a.x1 - b.x1; b1 = (tv > a.x1) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x1 = d; borrow = b1 + b2;
    tv = a.x2 - b.x2; b1 = (tv > a.x2) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x2 = d; borrow = b1 + b2;
    tv = a.x3 - b.x3; b1 = (tv > a.x3) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x3 = d; borrow = b1 + b2;
    tv = a.x4 - b.x4; b1 = (tv > a.x4) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x4 = d; borrow = b1 + b2;
    tv = a.x5 - b.x5; b1 = (tv > a.x5) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x5 = d; borrow = b1 + b2;

    if (borrow != 0ul) {
        Fp sum;
        ulong carry = 0ul;
        ulong s, cy1, t, cy2;
        s = diff.x0 + carry; cy1 = (s < diff.x0) ? 1ul:0ul; t = s + q.x0; cy2 = (t < s) ? 1ul:0ul; sum.x0 = t; carry = cy1 + cy2;
        s = diff.x1 + carry; cy1 = (s < diff.x1) ? 1ul:0ul; t = s + q.x1; cy2 = (t < s) ? 1ul:0ul; sum.x1 = t; carry = cy1 + cy2;
        s = diff.x2 + carry; cy1 = (s < diff.x2) ? 1ul:0ul; t = s + q.x2; cy2 = (t < s) ? 1ul:0ul; sum.x2 = t; carry = cy1 + cy2;
        s = diff.x3 + carry; cy1 = (s < diff.x3) ? 1ul:0ul; t = s + q.x3; cy2 = (t < s) ? 1ul:0ul; sum.x3 = t; carry = cy1 + cy2;
        s = diff.x4 + carry; cy1 = (s < diff.x4) ? 1ul:0ul; t = s + q.x4; cy2 = (t < s) ? 1ul:0ul; sum.x4 = t; carry = cy1 + cy2;
        s = diff.x5 + carry; cy1 = (s < diff.x5) ? 1ul:0ul; t = s + q.x5; cy2 = (t < s) ? 1ul:0ul; sum.x5 = t; carry = cy1 + cy2;
        return sum;
    }
    return diff;
}

#define MONT_STEP(b_val) \
    C = 0ul; \
    r = fma_add128(a.x0, b_val, t0, C); t0 = r.x; C = r.y; \
    r = fma_add128(a.x1, b_val, t1, C); t1 = r.x; C = r.y; \
    r = fma_add128(a.x2, b_val, t2, C); t2 = r.x; C = r.y; \
    r = fma_add128(a.x3, b_val, t3, C); t3 = r.x; C = r.y; \
    r = fma_add128(a.x4, b_val, t4, C); t4 = r.x; C = r.y; \
    r = fma_add128(a.x5, b_val, t5, C); t5 = r.x; C = r.y; \
    s = t6 + C; t6 = s; t7 += (s < t6 ? 1ul : 0ul); \
    \
    m = t0 * q_inv_neg; \
    C = 0ul; \
    r = fma_add128(m, q.x0, t0, C); t0 = r.x; C = r.y; \
    r = fma_add128(m, q.x1, t1, C); t1 = r.x; C = r.y; \
    r = fma_add128(m, q.x2, t2, C); t2 = r.x; C = r.y; \
    r = fma_add128(m, q.x3, t3, C); t3 = r.x; C = r.y; \
    r = fma_add128(m, q.x4, t4, C); t4 = r.x; C = r.y; \
    r = fma_add128(m, q.x5, t5, C); t5 = r.x; C = r.y; \
    s = t6 + C; t6 = s; t7 += (s < t6 ? 1ul : 0ul); \
    \
    t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7; t7 = 0ul;

inline Fp fp_mul(Fp a, Fp b, Fp q, ulong q_inv_neg) {
    ulong t0=0, t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;
    ulong2 r; ulong C; ulong s; ulong m;
    
    MONT_STEP(b.x0)
    MONT_STEP(b.x1)
    MONT_STEP(b.x2)
    MONT_STEP(b.x3)
    MONT_STEP(b.x4)
    MONT_STEP(b.x5)

    Fp diff;
    ulong borrow = 0ul;
    ulong tv, b1, d, b2;
    tv = t0 - q.x0; b1 = (tv > t0) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x0 = d; borrow = b1 + b2;
    tv = t1 - q.x1; b1 = (tv > t1) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x1 = d; borrow = b1 + b2;
    tv = t2 - q.x2; b1 = (tv > t2) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x2 = d; borrow = b1 + b2;
    tv = t3 - q.x3; b1 = (tv > t3) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x3 = d; borrow = b1 + b2;
    tv = t4 - q.x4; b1 = (tv > t4) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x4 = d; borrow = b1 + b2;
    tv = t5 - q.x5; b1 = (tv > t5) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff.x5 = d; borrow = b1 + b2;

    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    Fp res;
    res.x0 = use_diff ? diff.x0 : t0;
    res.x1 = use_diff ? diff.x1 : t1;
    res.x2 = use_diff ? diff.x2 : t2;
    res.x3 = use_diff ? diff.x3 : t3;
    res.x4 = use_diff ? diff.x4 : t4;
    res.x5 = use_diff ? diff.x5 : t5;
    return res;
}

inline Point point_double(Point p, Fp q, ulong q_inv_neg) {
    if (fp_is_zero(p.z) || fp_is_zero(p.y)) {
        Fp zero = {0,0,0,0,0,0};
        return {zero, zero, zero};
    }
    Fp A = fp_mul(p.x, p.x, q, q_inv_neg);
    Fp B = fp_mul(p.y, p.y, q, q_inv_neg);
    Fp C = fp_mul(B, B, q, q_inv_neg);

    Fp tmp = fp_add(p.x, B, q);
    Fp D = fp_mul(tmp, tmp, q, q_inv_neg);
    D = fp_sub(D, A, q);
    D = fp_sub(D, C, q);
    D = fp_add(D, D, q);

    Fp E = fp_add(A, A, q);
    E = fp_add(E, A, q);

    Fp F = fp_mul(E, E, q, q_inv_neg);

    Point out;
    tmp = fp_add(D, D, q);
    out.x = fp_sub(F, tmp, q);

    tmp = fp_sub(D, out.x, q);
    tmp = fp_mul(E, tmp, q, q_inv_neg);
    Fp tmp2 = fp_add(C, C, q);
    tmp2 = fp_add(tmp2, tmp2, q);
    tmp2 = fp_add(tmp2, tmp2, q);
    out.y = fp_sub(tmp, tmp2, q);

    tmp = fp_mul(p.y, p.z, q, q_inv_neg);
    out.z = fp_add(tmp, tmp, q);

    return out;
}

inline Point point_add(Point p1, Point p2, Fp q, ulong q_inv_neg) {
    if (fp_is_zero(p1.z)) return p2;
    if (fp_is_zero(p2.z)) return p1;

    Fp Z1Z1 = fp_mul(p1.z, p1.z, q, q_inv_neg);
    Fp Z2Z2 = fp_mul(p2.z, p2.z, q, q_inv_neg);
    Fp U1 = fp_mul(p1.x, Z2Z2, q, q_inv_neg);
    Fp U2 = fp_mul(p2.x, Z1Z1, q, q_inv_neg);
    Fp tmp = fp_mul(p1.y, p2.z, q, q_inv_neg);
    Fp S1 = fp_mul(tmp, Z2Z2, q, q_inv_neg);
    tmp = fp_mul(p2.y, p1.z, q, q_inv_neg);
    Fp S2 = fp_mul(tmp, Z1Z1, q, q_inv_neg);

    if (fp_eq(U1, U2)) {
        if (fp_eq(S1, S2)) return point_double(p1, q, q_inv_neg);
        Fp zero = {0,0,0,0,0,0};
        return {zero, zero, zero};
    }

    Fp H = fp_sub(U2, U1, q);
    Fp R = fp_sub(S2, S1, q);
    Fp HH = fp_mul(H, H, q, q_inv_neg);
    Fp HHH = fp_mul(H, HH, q, q_inv_neg);
    Fp V = fp_mul(U1, HH, q, q_inv_neg);

    Point out;
    out.x = fp_mul(R, R, q, q_inv_neg);
    out.x = fp_sub(out.x, HHH, q);
    tmp = fp_add(V, V, q);
    out.x = fp_sub(out.x, tmp, q);

    tmp = fp_sub(V, out.x, q);
    tmp = fp_mul(R, tmp, q, q_inv_neg);
    Fp tmp2 = fp_mul(S1, HHH, q, q_inv_neg);
    out.y = fp_sub(tmp, tmp2, q);

    tmp = fp_mul(p1.z, p2.z, q, q_inv_neg);
    out.z = fp_mul(tmp, H, q, q_inv_neg);

    return out;
}

inline Point load_point(device const ulong *src) {
    Point p;
    p.x.x0 = src[0]; p.x.x1 = src[1]; p.x.x2 = src[2]; p.x.x3 = src[3]; p.x.x4 = src[4]; p.x.x5 = src[5];
    p.y.x0 = src[6]; p.y.x1 = src[7]; p.y.x2 = src[8]; p.y.x3 = src[9]; p.y.x4 = src[10]; p.y.x5 = src[11];
    p.z.x0 = src[12]; p.z.x1 = src[13]; p.z.x2 = src[14]; p.z.x3 = src[15]; p.z.x4 = src[16]; p.z.x5 = src[17];
    return p;
}

inline void store_point(device ulong *dst, Point p) {
    dst[0] = p.x.x0; dst[1] = p.x.x1; dst[2] = p.x.x2; dst[3] = p.x.x3; dst[4] = p.x.x4; dst[5] = p.x.x5;
    dst[6] = p.y.x0; dst[7] = p.y.x1; dst[8] = p.y.x2; dst[9] = p.y.x3; dst[10] = p.y.x4; dst[11] = p.y.x5;
    dst[12] = p.z.x0; dst[13] = p.z.x1; dst[14] = p.z.x2; dst[15] = p.z.x3; dst[16] = p.z.x4; dst[17] = p.z.x5;
}

kernel void montgomery_msm_pair(
    device const ulong *scalars      [[buffer(0)]],
    device const ulong *points_in    [[buffer(1)]],
    device       ulong *scratch      [[buffer(2)]],
    device const ulong *q            [[buffer(3)]],
    constant ulong     &q_inv_neg    [[buffer(4)]],
    constant uint      &n_pairs      [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_pairs) return;

    Fp q_local;
    q_local.x0 = q[0]; q_local.x1 = q[1]; q_local.x2 = q[2]; 
    q_local.x3 = q[3]; q_local.x4 = q[4]; q_local.x5 = q[5];

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    ulong s_sh0 = (s0 << 1);
    ulong s_sh1 = (s1 << 1) | (s0 >> 63);
    ulong s_sh2 = (s2 << 1) | (s1 >> 63);
    ulong s_sh3 = (s3 << 1) | (s2 >> 63);
    ulong s_sh4 = (s3 >> 63);

    ulong t0 = s0 + s_sh0; ulong c0 = (t0 < s0) ? 1ul : 0ul; ulong D0 = t0;
    ulong t1 = s1 + s_sh1; ulong c1 = (t1 < s1) ? 1ul : 0ul; ulong D1 = t1 + c0; ulong c2 = (D1 < t1) ? 1ul : 0ul; ulong cy1 = c1 + c2;
    ulong t2 = s2 + s_sh2; ulong c3 = (t2 < s2) ? 1ul : 0ul; ulong D2 = t2 + cy1; ulong c4 = (D2 < t2) ? 1ul : 0ul; ulong cy2 = c3 + c4;
    ulong t3 = s3 + s_sh3; ulong c5 = (t3 < s3) ? 1ul : 0ul; ulong D3 = t3 + cy2; ulong c6 = (D3 < t3) ? 1ul : 0ul; ulong cy3 = c5 + c6;
    ulong D4 = s_sh4 + cy3;

    ulong nd0=0, nd1=0, nd2=0, nd3=0, nd4=0, nd5=0, nd6=0, nd7=0, nd8=0;
    for (int i = 0; i <= 256; ++i) {
        int k = i + 1;
        int d_bit = (k < 64) ? ((D0 >> k) & 1) :
                    (k < 128) ? ((D1 >> (k - 64)) & 1) :
                    (k < 192) ? ((D2 >> (k - 128)) & 1) :
                    (k < 256) ? ((D3 >> (k - 192)) & 1) : ((D4 >> (k - 256)) & 1);
        int s_bit = (k < 64) ? ((s0 >> k) & 1) :
                    (k < 128) ? ((s1 >> (k - 64)) & 1) :
                    (k < 192) ? ((s2 >> (k - 128)) & 1) :
                    (k < 256) ? ((s3 >> (k - 192)) & 1) : 0;
        int digit = d_bit - s_bit;
        ulong enc = (digit == 1) ? 1ul : (digit == -1) ? 2ul : 0ul;
        int word = i / 32;
        int shift = (i % 32) * 2;
        if (word == 0) nd0 |= (enc << shift);
        else if (word == 1) nd1 |= (enc << shift);
        else if (word == 2) nd2 |= (enc << shift);
        else if (word == 3) nd3 |= (enc << shift);
        else if (word == 4) nd4 |= (enc << shift);
        else if (word == 5) nd5 |= (enc << shift);
        else if (word == 6) nd6 |= (enc << shift);
        else if (word == 7) nd7 |= (enc << shift);
        else nd8 |= (enc << shift);
    }

    Point P = load_point(points_in + idx * 18u);
    Fp zero = {0,0,0,0,0,0};
    Point nP = P;
    nP.y = fp_sub(zero, P.y, q_local);

    Point A = {zero, zero, zero};
    bool found_one = false;

    for (int bit = 256; bit >= 0; --bit) {
        int word = bit / 32;
        int shift = (bit % 32) * 2;
        ulong enc = (word == 0) ? nd0 :
                    (word == 1) ? nd1 :
                    (word == 2) ? nd2 :
                    (word == 3) ? nd3 :
                    (word == 4) ? nd4 :
                    (word == 5) ? nd5 :
                    (word == 6) ? nd6 :
                    (word == 7) ? nd7 : nd8;
        enc = (enc >> shift) & 3ul;
        int naf_digit = (enc == 1ul) ? 1 : (enc == 2ul) ? -1 : 0;
        
        if (!found_one) {
            if (naf_digit == 0) continue;
            found_one = true;
            if (naf_digit == 1) A = P;
            else A = nP;
            continue;
        }
        
        A = point_double(A, q_local, q_inv_neg);
        
        if (naf_digit == 1) {
            A = point_add(A, P, q_local, q_inv_neg);
        } else if (naf_digit == -1) {
            A = point_add(A, nP, q_local, q_inv_neg);
        }
    }

    store_point(scratch + idx * 18u, A);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    Fp q_local;
    q_local.x0 = q[0]; q_local.x1 = q[1]; q_local.x2 = q[2]; 
    q_local.x3 = q[3]; q_local.x4 = q[4]; q_local.x5 = q[5];

    Point A = load_point(scratch + idx * 18u);
    Point B = load_point(scratch + (idx + half_count) * 18u);

    Point R = point_add(A, B, q_local, q_inv_neg);
    store_point(scratch + idx * 18u, R);
}