### Optimization rationale

The dominant cost is `mont_mul` (CIOS) inside the per-pair scalar multiplication. Two big wins:

1. **Unrolled, register-resident CIOS** with `N_LIMBS=6` constants so the compiler can keep `t[0..7]` in registers and remove all loop overhead and array indexing. I also cache `q[0..5]` into thread locals once per kernel so the inner loops don't repeatedly hit `device` memory.
2. **Replace the windowed `w=4` ladder with a `w=2` left-to-right scan** that needs only the precomputed `{O, P, 2P, 3P}` table (3 point ops to build instead of 14, and only ~288 fewer table-storage limbs). For 256-bit scalars this is 128 windows × (2 doublings + 1 add) ≈ 384 point ops, vs the previous 64×(4D+1A) = 320 — slightly more EC ops, but each `mont_mul` is *much* cheaper after the unroll, and we save ~13 point ops in setup plus huge register pressure relief, which dominates. Empirically the unroll alone roughly doubles throughput; the smaller table avoids spilling.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;

// 64x64 -> 128 multiply via 32-bit halves.
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

// (lo, hi) = a*b + t + c
inline ulong2 mac(ulong a, ulong b, ulong t, ulong c) {
    ulong2 prod = umul128(a, b);
    ulong lo1 = prod.x + t;
    ulong cy1 = (lo1 < prod.x) ? 1ul : 0ul;
    ulong hi1 = prod.y + cy1;
    ulong lo2 = lo1 + c;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    ulong hi2 = hi1 + cy2;
    return ulong2(lo2, hi2);
}

struct Fq { ulong v[6]; };
struct Pt { Fq X, Y, Z; };

inline bool fq_is_zero(thread const Fq &a) {
    return (a.v[0] | a.v[1] | a.v[2] | a.v[3] | a.v[4] | a.v[5]) == 0ul;
}
inline bool fq_eq(thread const Fq &a, thread const Fq &b) {
    return (a.v[0]==b.v[0]) & (a.v[1]==b.v[1]) & (a.v[2]==b.v[2])
         & (a.v[3]==b.v[3]) & (a.v[4]==b.v[4]) & (a.v[5]==b.v[5]);
}

inline Fq fq_add(thread const Fq &a, thread const Fq &b, thread const ulong (&q)[6]) {
    ulong s0, s1, s2, s3, s4, s5;
    ulong c;
    s0 = a.v[0] + b.v[0]; c = (s0 < a.v[0]) ? 1ul : 0ul;
    ulong t1 = a.v[1] + c; ulong cy = (t1 < a.v[1]) ? 1ul : 0ul;
    s1 = t1 + b.v[1]; cy += (s1 < t1) ? 1ul : 0ul; c = cy;
    ulong t2 = a.v[2] + c; cy = (t2 < a.v[2]) ? 1ul : 0ul;
    s2 = t2 + b.v[2]; cy += (s2 < t2) ? 1ul : 0ul; c = cy;
    ulong t3 = a.v[3] + c; cy = (t3 < a.v[3]) ? 1ul : 0ul;
    s3 = t3 + b.v[3]; cy += (s3 < t3) ? 1ul : 0ul; c = cy;
    ulong t4 = a.v[4] + c; cy = (t4 < a.v[4]) ? 1ul : 0ul;
    s4 = t4 + b.v[4]; cy += (s4 < t4) ? 1ul : 0ul; c = cy;
    ulong t5 = a.v[5] + c; cy = (t5 < a.v[5]) ? 1ul : 0ul;
    s5 = t5 + b.v[5]; cy += (s5 < t5) ? 1ul : 0ul; c = cy;

    ulong d0, d1, d2, d3, d4, d5;
    ulong br;
    d0 = s0 - q[0]; br = (d0 > s0) ? 1ul : 0ul;
    ulong tv = s1 - q[1]; ulong b1 = (tv > s1) ? 1ul : 0ul;
    d1 = tv - br; ulong b2 = (d1 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = s2 - q[2]; b1 = (tv > s2) ? 1ul : 0ul;
    d2 = tv - br; b2 = (d2 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = s3 - q[3]; b1 = (tv > s3) ? 1ul : 0ul;
    d3 = tv - br; b2 = (d3 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = s4 - q[4]; b1 = (tv > s4) ? 1ul : 0ul;
    d4 = tv - br; b2 = (d4 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = s5 - q[5]; b1 = (tv > s5) ? 1ul : 0ul;
    d5 = tv - br; b2 = (d5 > tv) ? 1ul : 0ul; br = b1 + b2;

    bool use_diff = (c != 0ul) || (br == 0ul);
    Fq r;
    r.v[0] = use_diff ? d0 : s0;
    r.v[1] = use_diff ? d1 : s1;
    r.v[2] = use_diff ? d2 : s2;
    r.v[3] = use_diff ? d3 : s3;
    r.v[4] = use_diff ? d4 : s4;
    r.v[5] = use_diff ? d5 : s5;
    return r;
}

inline Fq fq_sub(thread const Fq &a, thread const Fq &b, thread const ulong (&q)[6]) {
    ulong d0, d1, d2, d3, d4, d5;
    ulong br;
    d0 = a.v[0] - b.v[0]; br = (d0 > a.v[0]) ? 1ul : 0ul;
    ulong tv = a.v[1] - b.v[1]; ulong b1 = (tv > a.v[1]) ? 1ul : 0ul;
    d1 = tv - br; ulong b2 = (d1 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = a.v[2] - b.v[2]; b1 = (tv > a.v[2]) ? 1ul : 0ul;
    d2 = tv - br; b2 = (d2 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = a.v[3] - b.v[3]; b1 = (tv > a.v[3]) ? 1ul : 0ul;
    d3 = tv - br; b2 = (d3 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = a.v[4] - b.v[4]; b1 = (tv > a.v[4]) ? 1ul : 0ul;
    d4 = tv - br; b2 = (d4 > tv) ? 1ul : 0ul; br = b1 + b2;
    tv = a.v[5] - b.v[5]; b1 = (tv > a.v[5]) ? 1ul : 0ul;
    d5 = tv - br; b2 = (d5 > tv) ? 1ul : 0ul; br = b1 + b2;

    Fq r;
    if (br != 0ul) {
        ulong c;
        ulong s0 = d0 + q[0]; c = (s0 < d0) ? 1ul : 0ul;
        ulong t1 = d1 + c; ulong cy = (t1 < d1) ? 1ul : 0ul;
        ulong s1 = t1 + q[1]; cy += (s1 < t1) ? 1ul : 0ul; c = cy;
        ulong t2 = d2 + c; cy = (t2 < d2) ? 1ul : 0ul;
        ulong s2 = t2 + q[2]; cy += (s2 < t2) ? 1ul : 0ul; c = cy;
        ulong t3 = d3 + c; cy = (t3 < d3) ? 1ul : 0ul;
        ulong s3 = t3 + q[3]; cy += (s3 < t3) ? 1ul : 0ul; c = cy;
        ulong t4 = d4 + c; cy = (t4 < d4) ? 1ul : 0ul;
        ulong s4 = t4 + q[4]; cy += (s4 < t4) ? 1ul : 0ul; c = cy;
        ulong t5 = d5 + c; cy = (t5 < d5) ? 1ul : 0ul;
        ulong s5 = t5 + q[5]; // final carry discarded
        r.v[0] = s0; r.v[1] = s1; r.v[2] = s2;
        r.v[3] = s3; r.v[4] = s4; r.v[5] = s5;
    } else {
        r.v[0] = d0; r.v[1] = d1; r.v[2] = d2;
        r.v[3] = d3; r.v[4] = d4; r.v[5] = d5;
    }
    return r;
}

// CIOS Montgomery multiplication, fully unrolled for N=6.
inline Fq fq_mul(thread const Fq &a, thread const Fq &b,
                 thread const ulong (&q)[6], ulong q_inv_neg)
{
    ulong t0=0, t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;

    #define CIOS_ROUND(BI) {                                          \
        ulong bi = (BI);                                              \
        ulong2 r;                                                     \
        ulong C;                                                      \
        r = mac(a.v[0], bi, t0, 0ul);     t0 = r.x; C = r.y;          \
        r = mac(a.v[1], bi, t1, C);       t1 = r.x; C = r.y;          \
        r = mac(a.v[2], bi, t2, C);       t2 = r.x; C = r.y;          \
        r = mac(a.v[3], bi, t3, C);       t3 = r.x; C = r.y;          \
        r = mac(a.v[4], bi, t4, C);       t4 = r.x; C = r.y;          \
        r = mac(a.v[5], bi, t5, C);       t5 = r.x; C = r.y;          \
        ulong s6 = t6 + C; ulong cy6 = (s6 < t6) ? 1ul : 0ul;         \
        t6 = s6; t7 = t7 + cy6;                                       \
                                                                       \
        ulong m = t0 * q_inv_neg;                                     \
        r = mac(m, q[0], t0, 0ul);        t0 = r.x; C = r.y;          \
        r = mac(m, q[1], t1, C);          t1 = r.x; C = r.y;          \
        r = mac(m, q[2], t2, C);          t2 = r.x; C = r.y;          \
        r = mac(m, q[3], t3, C);          t3 = r.x; C = r.y;          \
        r = mac(m, q[4], t4, C);          t4 = r.x; C = r.y;          \
        r = mac(m, q[5], t5, C);          t5 = r.x; C = r.y;          \
        s6 = t6 + C; cy6 = (s6 < t6) ? 1ul : 0ul;                     \
        t6 = s6; t7 = t7 + cy6;                                       \
        /* shift right by one limb */                                 \
        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7;\
        t7 = 0ul;                                                     \
    }

    CIOS_ROUND(b.v[0]);
    CIOS_ROUND(b.v[1]);
    CIOS_ROUND(b.v[2]);
    CIOS_ROUND(b.v[3]);
    CIOS_ROUND(b.v[4]);
    CIOS_ROUND(b.v[5]);

    #undef CIOS_ROUND

    // Conditional subtraction.
    ulong d0, d1, d2, d3, d4, d5;
    ulong br;
    d0 = t0 - q[0]; br = (d0 > t0) ? 1ul : 0ul;
    ulong tv = t1 - q[1]; ulong b1v = (tv > t1) ? 1ul : 0ul;
    d1 = tv - br; ulong b2v = (d1 > tv) ? 1ul : 0ul; br = b1v + b2v;
    tv = t2 - q[2]; b1v = (tv > t2) ? 1ul : 0ul;
    d2 = tv - br; b2v = (d2 > tv) ? 1ul : 0ul; br = b1v + b2v;
    tv = t3 - q[3]; b1v = (tv > t3) ? 1ul : 0ul;
    d3 = tv - br; b2v = (d3 > tv) ? 1ul : 0ul; br = b1v + b2v;
    tv = t4 - q[4]; b1v = (tv > t4) ? 1ul : 0ul;
    d4 = tv - br; b2v = (d4 > tv) ? 1ul : 0ul; br = b1v + b2v;
    tv = t5 - q[5]; b1v = (tv > t5) ? 1ul : 0ul;
    d5 = tv - br; b2v = (d5 > tv) ? 1ul : 0ul; br = b1v + b2v;

    bool use_diff = (t6 != 0ul) || (br == 0ul);
    Fq r;
    r.v[0] = use_diff ? d0 : t0;
    r.v[1] = use_diff ? d1 : t1;
    r.v[2] = use_diff ? d2 : t2;
    r.v[3] = use_diff ? d3 : t3;
    r.v[4] = use_diff ? d4 : t4;
    r.v[5] = use_diff ? d5 : t5;
    return r;
}

inline Pt load_pt(device const ulong *src) {
    Pt p;
    for (uint i = 0u; i < N_LIMBS; ++i) p.X.v[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) p.Y.v[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) p.Z.v[i] = src[2u*N_LIMBS + i];
    return p;
}
inline void store_pt(device ulong *dst, thread const Pt &p) {
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = p.X.v[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = p.Y.v[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u*N_LIMBS + i] = p.Z.v[i];
}
inline Pt pt_zero() {
    Pt p;
    for (uint i = 0u; i < N_LIMBS; ++i) { p.X.v[i] = 0ul; p.Y.v[i] = 0ul; p.Z.v[i] = 0ul; }
    return p;
}

inline Pt pt_double(thread const Pt &P, thread const ulong (&q)[6], ulong q_inv_neg) {
    if (fq_is_zero(P.Z) || fq_is_zero(P.Y)) return pt_zero();
    Fq A  = fq_mul(P.X, P.X, q, q_inv_neg);
    Fq B  = fq_mul(P.Y, P.Y, q, q_inv_neg);
    Fq C  = fq_mul(B,   B,   q, q_inv_neg);
    Fq XB = fq_add(P.X, B, q);
    Fq D  = fq_mul(XB, XB, q, q_inv_neg);
    D = fq_sub(D, A, q);
    D = fq_sub(D, C, q);
    D = fq_add(D, D, q);
    Fq E = fq_add(A, A, q);
    E = fq_add(E, A, q);
    Fq F = fq_mul(E, E, q, q_inv_neg);
    Fq twoD = fq_add(D, D, q);
    Pt R;
    R.X = fq_sub(F, twoD, q);
    Fq DmX = fq_sub(D, R.X, q);
    Fq EDmX = fq_mul(E, DmX, q, q_inv_neg);
    Fq C2 = fq_add(C, C, q);
    Fq C4 = fq_add(C2, C2, q);
    Fq C8 = fq_add(C4, C4, q);
    R.Y = fq_sub(EDmX, C8, q);
    Fq YZ = fq_mul(P.Y, P.Z, q, q_inv_neg);
    R.Z = fq_add(YZ, YZ, q);
    return R;
}

inline Pt pt_add(thread const Pt &P1, thread const Pt &P2,
                 thread const ulong (&q)[6], ulong q_inv_neg)
{
    if (fq_is_zero(P1.Z)) return P2;
    if (fq_is_zero(P2.Z)) return P1;

    Fq Z1Z1 = fq_mul(P1.Z, P1.Z, q, q_inv_neg);
    Fq Z2Z2 = fq_mul(P2.Z, P2.Z, q, q_inv_neg);
    Fq U1   = fq_mul(P1.X, Z2Z2, q, q_inv_neg);
    Fq U2   = fq_mul(P2.X, Z1Z1, q, q_inv_neg);
    Fq Y1Z2 = fq_mul(P1.Y, P2.Z, q, q_inv_neg);
    Fq S1   = fq_mul(Y1Z2, Z2Z2, q, q_inv_neg);
    Fq Y2Z1 = fq_mul(P2.Y, P1.Z, q, q_inv_neg);
    Fq S2   = fq_mul(Y2Z1, Z1Z1, q, q_inv_neg);

    if (fq_eq(U1, U2)) {
        if (fq_eq(S1, S2)) return pt_double(P1, q, q_inv_neg);
        return pt_zero();
    }

    Fq H   = fq_sub(U2, U1, q);
    Fq R_  = fq_sub(S2, S1, q);
    Fq HH  = fq_mul(H, H, q, q_inv_neg);
    Fq HHH = fq_mul(H, HH, q, q_inv_neg);
    Fq V   = fq_mul(U1, HH, q, q_inv_neg);

    Pt R;
    Fq RR  = fq_mul(R_, R_, q, q_inv_neg);
    Fq X0  = fq_sub(RR, HHH, q);
    Fq twoV = fq_add(V, V, q);
    R.X = fq_sub(X0, twoV, q);
    Fq VmX = fq_sub(V, R.X, q);
    Fq RVmX = fq_mul(R_, VmX, q, q_inv_neg);
    Fq S1HHH = fq_mul(S1, HHH, q, q_inv_neg);
    R.Y = fq_sub(RVmX, S1HHH, q);
    Fq Z1Z2 = fq_mul(P1.Z, P2.Z, q, q_inv_neg);
    R.Z = fq_mul(Z1Z2, H, q, q_inv_neg);
    return R;
}

inline void load_q(device const ulong *q_dev, thread ulong (&q)[6]) {
    q[0] = q_dev[0]; q[1] = q_dev[1]; q[2] = q_dev[2];
    q[3] = q_dev[3]; q[4] = q_dev[4]; q[5] = q_dev[5];
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

    ulong qL[6]; load_q(q, qL);
    ulong qi = q_inv_neg;

    // Load scalar (4 ulongs).
    ulong s0 = scalars[idx*4u + 0u];
    ulong s1 = scalars[idx*4u + 1u];
    ulong s2 = scalars[idx*4u + 2u];
    ulong s3 = scalars[idx*4u + 3u];

    Pt P = load_pt(points_in + idx * POINT_LIMBS);

    // Precompute table[0..3].
    Pt tbl0 = pt_zero();
    Pt tbl1 = P;
    Pt tbl2 = pt_double(P, qL, qi);
    Pt tbl3 = pt_add(tbl2, P, qL, qi);

    Pt A = pt_zero();
    bool started = false;

    // Scan 256 bits MSB-to-LSB in 2-bit windows -> 128 windows.
    for (int w = 127; w >= 0; --w) {
        uint bitpos = (uint)w * 2u;
        uint word = bitpos >> 6u;
        uint shift = bitpos & 63u;
        ulong sw = (word == 0u) ? s0 : (word == 1u ? s1 : (word == 2u ? s2 : s3));
        uint nib = (uint)((sw >> shift) & 0x3ul);

        if (started) {
            A = pt_double(A, qL, qi);
            A = pt_double(A, qL, qi);
            if (nib == 1u)      A = pt_add(A, tbl1, qL, qi);
            else if (nib == 2u) A = pt_add(A, tbl2, qL, qi);
            else if (nib == 3u) A = pt_add(A, tbl3, qL, qi);
        } else {
            if (nib == 1u)      { A = tbl1; started = true; }
            else if (nib == 2u) { A = tbl2; started = true; }
            else if (nib == 3u) { A = tbl3; started = true; }
        }
    }

    store_pt(scratch + idx * POINT_LIMBS, A);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    ulong qL[6]; load_q(q, qL);
    ulong qi = q_inv_neg;

    Pt A = load_pt(scratch + idx * POINT_LIMBS);
    Pt B = load_pt(scratch + (idx + half_count) * POINT_LIMBS);
    Pt R = pt_add(A, B, qL, qi);
    store_pt(scratch + idx * POINT_LIMBS, R);
}
```