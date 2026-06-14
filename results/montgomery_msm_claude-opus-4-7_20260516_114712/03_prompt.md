## Task: montgomery_msm

Multi-scalar multiplication on a short-Weierstrass elliptic curve. Given ``n_pairs`` pairs of (256-bit scalar ``s_i``, Jacobian point ``P_i``), compute the single curve point ``R = sum_i s_i * P_i`` and emit it in Jacobian Montgomery form. The in-distribution sizes use BLS12-381 G1 (q ~ 381 bits, b = 4); the held-out size uses BN254 G1 (q ~ 254 bits, b = 3).

Field representation: all elements live in Montgomery form with R = 2^384, six 64-bit limbs. The base-field modulus ``q`` (6 ulongs, little-endian) and the CIOS scalar ``q_inv_neg`` (``-q^-1 mod 2^64``) are bound as device / constant buffers; both **must** be read at runtime. A candidate that hardcodes the in-distribution modulus or its Montgomery constants silently produces wrong output on the held-out probe.

Coordinate convention: 6-limb Jacobian ``(X, Y, Z)``, little-endian limbs, affine point is ``(X / Z^2, Y / Z^3)``, ``Z = 0`` represents the point at infinity. Per point: 18 ulongs.

Scalars: 4-ulong little-endian limbs (both curves' scalar fields fit in 256 bits).

Bit-exact correctness: the host normalizes the GPU Jacobian output to affine Montgomery form via one base-field inversion, then compares the (X_aff_mont, Y_aff_mont) pair against the algebraic reference. A non-canonical limb (>= q) counts as a mismatch even if the residue class agrees.

Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved and the ``pair`` + ``log2(n_pairs)`` x ``reduce`` dispatch schedule is honored (the pair kernel sees each (scalar, point) pair exactly once; each reduce dispatch sees the current tree level via ``half_count``).

## Required kernel signature(s)

```
kernel void montgomery_msm_pair(
    device const ulong *scalars      [[buffer(0)]],
    device const ulong *points_in    [[buffer(1)]],
    device       ulong *scratch      [[buffer(2)]],
    device const ulong *q            [[buffer(3)]],
    constant ulong     &q_inv_neg    [[buffer(4)]],
    constant uint      &n_pairs      [[buffer(5)]],
    uint idx [[thread_position_in_grid]]);

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  montgomery_msm_pair: one thread per (s_i, P_i); guard against idx >= n_pairs; grid rounded up to a multiple of the TG width.
  montgomery_msm_reduce: invoked log2(n_pairs) times in a single compute command encoder with ``half_count`` successively halving (n_pairs/2, n_pairs/4, ..., 1). One thread per active slot; thread t reads scratch[t] and scratch[t + half_count], adds them in Jacobian form, and writes the sum back to scratch[t]. The serial command encoder gives read-after-write between levels with no explicit barriers required.
  threadsPerThreadgroup = (min(grid_w, 64), 1, 1) for both kernels in the seed; cooperative implementations may pick a different tile width but must honor the buffer layout and the half_count contract in reduce.
```

## Your previous attempt

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

Result of previous attempt:
           bls_N4K: correct, 360.01 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 1189.20 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 4413.87 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0011

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint SCALAR_BITS = 256u;
constexpr constant uint WINDOW = 4u;
constexpr constant uint TABLE_SIZE = 16u; // 2^WINDOW

constant ulong LIMB_MASK_LO32 = 0x00000000FFFFFFFFul;

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & LIMB_MASK_LO32) + (p10 & LIMB_MASK_LO32);
    ulong lo  = (p00 & LIMB_MASK_LO32) | (mid << 32);
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

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != 0ul) return false;
    }
    return true;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    for (uint i = 0u; i < N_LIMBS; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    device const ulong *q)
{
    ulong sum[N_LIMBS];
    ulong carry = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong s = a[i] + carry;
        ulong cy1 = (s < a[i]) ? 1ul : 0ul;
        ulong t = s + b[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        sum[i] = t;
        carry = cy1 + cy2;
    }
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = (tv > sum[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    device const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        for (uint i = 0u; i < N_LIMBS; ++i) {
            ulong s = diff[i] + carry;
            ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
            ulong t = s + q[i];
            ulong cy2 = (t < s) ? 1ul : 0ul;
            c[i] = t;
            carry = cy1 + cy2;
        }
    } else {
        for (uint i = 0u; i < N_LIMBS; ++i) c[i] = diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     device const ulong *q, ulong q_inv_neg)
{
    ulong t[N_LIMBS + 2];
    for (uint i = 0u; i < N_LIMBS + 2u; ++i) t[i] = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong C = 0ul;
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(a[j], b[i], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[N_LIMBS] + C;
            ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
            t[N_LIMBS] = s;
            t[N_LIMBS + 1] += cy;
        }

        ulong m = t[0] * q_inv_neg;

        C = 0ul;
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = fma_add128(m, q[j], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[N_LIMBS] + C;
            ulong cy = (s < t[N_LIMBS]) ? 1ul : 0ul;
            t[N_LIMBS] = s;
            t[N_LIMBS + 1] += cy;
        }

        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1];
        }
        t[N_LIMBS + 1] = 0ul;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (t[N_LIMBS] != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N_LIMBS; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
    }
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = X[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = Y[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u * N_LIMBS + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = 0ul;
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          device const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    ulong A[N_LIMBS], B[N_LIMBS], C[N_LIMBS];
    ulong D[N_LIMBS], E[N_LIMBS], F[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_mul(A, X, X, q, q_inv_neg);
    mont_mul(B, Y, Y, q, q_inv_neg);
    mont_mul(C, B, B, q, q_inv_neg);

    mod_add(tmp, X, B, q);
    mont_mul(D, tmp, tmp, q, q_inv_neg);
    mod_sub(D, D, A, q);
    mod_sub(D, D, C, q);
    mod_add(D, D, D, q);

    mod_add(E, A, A, q);
    mod_add(E, E, A, q);

    mont_mul(F, E, E, q, q_inv_neg);

    mod_add(tmp, D, D, q);
    mod_sub(oX, F, tmp, q);

    mod_sub(tmp, D, oX, q);
    mont_mul(tmp, E, tmp, q, q_inv_neg);
    mod_add(tmp2, C, C, q);
    mod_add(tmp2, tmp2, tmp2, q);
    mod_add(tmp2, tmp2, tmp2, q);
    mod_sub(oY, tmp, tmp2, q);

    mont_mul(tmp, Y, Z, q, q_inv_neg);
    mod_add(oZ, tmp, tmp, q);
}

inline void jac_add_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                       thread const ulong *X1, thread const ulong *Y1, thread const ulong *Z1,
                       thread const ulong *X2, thread const ulong *Y2, thread const ulong *Z2,
                       device const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z1)) {
        copy_n(oX, X2); copy_n(oY, Y2); copy_n(oZ, Z2);
        return;
    }
    if (is_zero_n(Z2)) {
        copy_n(oX, X1); copy_n(oY, Y1); copy_n(oZ, Z1);
        return;
    }
    ulong Z1Z1[N_LIMBS], Z2Z2[N_LIMBS];
    ulong U1[N_LIMBS], U2[N_LIMBS], S1[N_LIMBS], S2[N_LIMBS];
    ulong H[N_LIMBS], R[N_LIMBS];
    ulong HH[N_LIMBS], HHH[N_LIMBS], V[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_mul(Z1Z1, Z1, Z1, q, q_inv_neg);
    mont_mul(Z2Z2, Z2, Z2, q, q_inv_neg);
    mont_mul(U1,   X1, Z2Z2, q, q_inv_neg);
    mont_mul(U2,   X2, Z1Z1, q, q_inv_neg);
    mont_mul(tmp,  Y1, Z2,   q, q_inv_neg);
    mont_mul(S1,   tmp, Z2Z2, q, q_inv_neg);
    mont_mul(tmp,  Y2, Z1,   q, q_inv_neg);
    mont_mul(S2,   tmp, Z1Z1, q, q_inv_neg);

    if (eq_n(U1, U2)) {
        if (eq_n(S1, S2)) {
            jac_double_pt(oX, oY, oZ, X1, Y1, Z1, q, q_inv_neg);
        } else {
            zero_point(oX, oY, oZ);
        }
        return;
    }

    mod_sub(H,   U2, U1, q);
    mod_sub(R,   S2, S1, q);
    mont_mul(HH,  H, H, q, q_inv_neg);
    mont_mul(HHH, H, HH, q, q_inv_neg);
    mont_mul(V,   U1, HH, q, q_inv_neg);

    mont_mul(oX, R, R, q, q_inv_neg);
    mod_sub(oX, oX, HHH, q);
    mod_add(tmp, V, V, q);
    mod_sub(oX, oX, tmp, q);

    mod_sub(tmp, V, oX, q);
    mont_mul(tmp, R, tmp, q, q_inv_neg);
    mont_mul(tmp2, S1, HHH, q, q_inv_neg);
    mod_sub(oY, tmp, tmp2, q);

    mont_mul(tmp, Z1, Z2, q, q_inv_neg);
    mont_mul(oZ, tmp, H, q, q_inv_neg);
}

// ------------------------------------------------------------------
// Windowed scalar multiplication with w=4. Builds a table of
// {0*P, 1*P, 2*P, ..., 15*P} once, then scans the scalar 4 bits at a
// time from MSB to LSB: at each step do 4 doublings + 1 add of the
// table entry. Also skips leading zero windows by initialising the
// accumulator from the first non-zero window directly.
// ------------------------------------------------------------------
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

    ulong s[4];
    s[0] = scalars[idx * 4u + 0u];
    s[1] = scalars[idx * 4u + 1u];
    s[2] = scalars[idx * 4u + 2u];
    s[3] = scalars[idx * 4u + 3u];

    // Load P.
    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    // Precompute table[k] = k * P, k = 0..15. Stored as 3*6=18 ulongs each.
    // 16 entries * 18 ulongs = 288 ulongs of thread storage. Tight but ok.
    ulong tblX[TABLE_SIZE][N_LIMBS];
    ulong tblY[TABLE_SIZE][N_LIMBS];
    ulong tblZ[TABLE_SIZE][N_LIMBS];

    // table[0] = O
    for (uint i = 0u; i < N_LIMBS; ++i) { tblX[0][i] = 0ul; tblY[0][i] = 0ul; tblZ[0][i] = 0ul; }
    // table[1] = P
    for (uint i = 0u; i < N_LIMBS; ++i) { tblX[1][i] = PX[i]; tblY[1][i] = PY[i]; tblZ[1][i] = PZ[i]; }

    // table[2] = 2P
    {
        ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
        jac_double_pt(RX, RY, RZ, PX, PY, PZ, q, q_inv_neg);
        for (uint i = 0u; i < N_LIMBS; ++i) { tblX[2][i] = RX[i]; tblY[2][i] = RY[i]; tblZ[2][i] = RZ[i]; }
    }
    // table[k] = table[k-1] + P for k = 3..15
    for (uint k = 3u; k < TABLE_SIZE; ++k) {
        ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
        for (uint i = 0u; i < N_LIMBS; ++i) { AX[i] = tblX[k-1u][i]; AY[i] = tblY[k-1u][i]; AZ[i] = tblZ[k-1u][i]; }
        ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
        jac_add_pt(RX, RY, RZ, AX, AY, AZ, PX, PY, PZ, q, q_inv_neg);
        for (uint i = 0u; i < N_LIMBS; ++i) { tblX[k][i] = RX[i]; tblY[k][i] = RY[i]; tblZ[k][i] = RZ[i]; }
    }

    // Accumulator A.
    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    bool started = false;
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];

    // SCALAR_BITS = 256, WINDOW = 4 -> 64 windows from MSB to LSB.
    for (int w = 63; w >= 0; --w) {
        uint bitpos = (uint)w * WINDOW;
        uint word = bitpos >> 6u;
        uint shift = bitpos & 63u;
        uint nib = (uint)((s[word] >> shift) & 0xFul);

        if (started) {
            // 4 doublings.
            for (uint d = 0u; d < WINDOW; ++d) {
                jac_double_pt(TX, TY, TZ, AX, AY, AZ, q, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
            if (nib != 0u) {
                // Variable-index lookup into table.
                ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];
                for (uint i = 0u; i < N_LIMBS; ++i) {
                    BX[i] = tblX[nib][i];
                    BY[i] = tblY[nib][i];
                    BZ[i] = tblZ[nib][i];
                }
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, BX, BY, BZ, q, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
        } else {
            if (nib != 0u) {
                // Initialize accumulator directly from table[nib].
                for (uint i = 0u; i < N_LIMBS; ++i) {
                    AX[i] = tblX[nib][i];
                    AY[i] = tblY[nib][i];
                    AZ[i] = tblZ[nib][i];
                }
                started = true;
            }
        }
    }

    store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];
    load_point(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);

    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    jac_add_pt(RX, RY, RZ, AX, AY, AZ, BX, BY, BZ, q, q_inv_neg);
    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}
```

Incumbent result:
           bls_N4K: correct, 48.94 ms, 0.4 Gmodmul/s (int64) (0.7% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 192.15 ms, 0.4 Gmodmul/s (int64) (0.7% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 726.10 ms, 0.4 Gmodmul/s (int64) (0.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0075

## History

- iter  0: compile=OK | correct=True | score=0.0039469233524904185
- iter  1: compile=OK | correct=True | score=0.007485626493860429
- iter  2: compile=OK | correct=True | score=0.0011487006695191965

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
