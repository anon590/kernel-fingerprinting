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

constexpr constant uint N = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WINDOW = 4u;
constexpr constant uint TABLE_SIZE = 16u;

// mac: (hi,lo) accumulator helpers
// Compute r = a + b, returning carry-out via overflow detection.
inline ulong add_cc(ulong a, ulong b, thread ulong &c) {
    ulong s = a + b;
    c = (s < a) ? 1ul : 0ul;
    return s;
}
inline ulong adc(ulong a, ulong b, ulong cin, thread ulong &cout) {
    ulong s = a + b;
    ulong c1 = (s < a) ? 1ul : 0ul;
    ulong s2 = s + cin;
    ulong c2 = (s2 < s) ? 1ul : 0ul;
    cout = c1 + c2;
    return s2;
}
inline ulong sbb(ulong a, ulong b, ulong bin, thread ulong &bout) {
    ulong d = a - b;
    ulong b1 = (a < b) ? 1ul : 0ul;
    ulong d2 = d - bin;
    ulong b2 = (d < bin) ? 1ul : 0ul;
    bout = b1 + b2;
    return d2;
}

// -------------------- Field arithmetic, 6x ulong --------------------

inline bool fe_is_zero(thread const ulong *a) {
    ulong acc = 0ul;
    for (uint i = 0u; i < N; ++i) acc |= a[i];
    return acc == 0ul;
}
inline bool fe_eq(thread const ulong *a, thread const ulong *b) {
    ulong acc = 0ul;
    for (uint i = 0u; i < N; ++i) acc |= (a[i] ^ b[i]);
    return acc == 0ul;
}
inline void fe_copy(thread ulong *o, thread const ulong *a) {
    for (uint i = 0u; i < N; ++i) o[i] = a[i];
}
inline void fe_zero(thread ulong *o) {
    for (uint i = 0u; i < N; ++i) o[i] = 0ul;
}

inline void fe_add(thread ulong *c, thread const ulong *a, thread const ulong *b,
                   thread const ulong *q)
{
    ulong s[N];
    ulong carry = 0ul;
    for (uint i = 0u; i < N; ++i) {
        ulong cc;
        s[i] = adc(a[i], b[i], carry, cc);
        carry = cc;
    }
    ulong d[N];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N; ++i) {
        ulong bb;
        d[i] = sbb(s[i], q[i], borrow, bb);
        borrow = bb;
    }
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N; ++i) c[i] = use_diff ? d[i] : s[i];
}

inline void fe_sub(thread ulong *c, thread const ulong *a, thread const ulong *b,
                   thread const ulong *q)
{
    ulong d[N];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N; ++i) {
        ulong bb;
        d[i] = sbb(a[i], b[i], borrow, bb);
        borrow = bb;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        for (uint i = 0u; i < N; ++i) {
            ulong cc;
            c[i] = adc(d[i], q[i], carry, cc);
            carry = cc;
        }
    } else {
        for (uint i = 0u; i < N; ++i) c[i] = d[i];
    }
}

// CIOS Montgomery multiplication on 6 x ulong with 64-bit mu.
// t has 7 limbs. After 6 iterations t[0..5] is result, t[6] is top.
inline void fe_mul(thread ulong *out, thread const ulong *a, thread const ulong *b,
                   thread const ulong *q, ulong mu)
{
    ulong t0=0ul, t1=0ul, t2=0ul, t3=0ul, t4=0ul, t5=0ul, t6=0ul;

    for (uint i = 0u; i < N; ++i) {
        ulong bi = b[i];

        // t += a * bi  (length 7)
        ulong lo, hi, s, c1, c2, C;

        // j=0
        lo = a[0]*bi; hi = mulhi(a[0], bi);
        s = t0 + lo;          c1 = (s < t0) ? 1ul : 0ul;
        t0 = s; C = hi + c1;

        // j=1
        lo = a[1]*bi; hi = mulhi(a[1], bi);
        s = t1 + lo;          c1 = (s < t1) ? 1ul : 0ul;
        ulong s2 = s + C;     c2 = (s2 < s) ? 1ul : 0ul;
        t1 = s2; C = hi + c1 + c2;

        // j=2
        lo = a[2]*bi; hi = mulhi(a[2], bi);
        s = t2 + lo;          c1 = (s < t2) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t2 = s2; C = hi + c1 + c2;

        // j=3
        lo = a[3]*bi; hi = mulhi(a[3], bi);
        s = t3 + lo;          c1 = (s < t3) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t3 = s2; C = hi + c1 + c2;

        // j=4
        lo = a[4]*bi; hi = mulhi(a[4], bi);
        s = t4 + lo;          c1 = (s < t4) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t4 = s2; C = hi + c1 + c2;

        // j=5
        lo = a[5]*bi; hi = mulhi(a[5], bi);
        s = t5 + lo;          c1 = (s < t5) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t5 = s2; C = hi + c1 + c2;

        // t6 += C (t6 could grow by up to 1 bit per iter, but stays small)
        ulong nt6 = t6 + C;
        ulong t7  = (nt6 < t6) ? 1ul : 0ul;
        t6 = nt6;

        // m = t0 * mu (mod 2^64)
        ulong m = t0 * mu;

        // t += m * q. After this t[0] == 0, then shift right one limb.
        // j=0
        lo = m*q[0]; hi = mulhi(m, q[0]);
        s = t0 + lo;          c1 = (s < t0) ? 1ul : 0ul;
        // t0 result discarded (should be 0); carry C = hi + c1
        C = hi + c1;

        // j=1
        lo = m*q[1]; hi = mulhi(m, q[1]);
        s = t1 + lo;          c1 = (s < t1) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t0 = s2; C = hi + c1 + c2;

        // j=2
        lo = m*q[2]; hi = mulhi(m, q[2]);
        s = t2 + lo;          c1 = (s < t2) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t1 = s2; C = hi + c1 + c2;

        // j=3
        lo = m*q[3]; hi = mulhi(m, q[3]);
        s = t3 + lo;          c1 = (s < t3) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t2 = s2; C = hi + c1 + c2;

        // j=4
        lo = m*q[4]; hi = mulhi(m, q[4]);
        s = t4 + lo;          c1 = (s < t4) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t3 = s2; C = hi + c1 + c2;

        // j=5
        lo = m*q[5]; hi = mulhi(m, q[5]);
        s = t5 + lo;          c1 = (s < t5) ? 1ul : 0ul;
        s2 = s + C;           c2 = (s2 < s) ? 1ul : 0ul;
        t4 = s2; C = hi + c1 + c2;

        // t6 += C, then shift
        ulong sx = t6 + C;
        ulong cx = (sx < t6) ? 1ul : 0ul;
        t5 = sx;
        t6 = t7 + cx;
    }

    ulong r[N];
    r[0]=t0; r[1]=t1; r[2]=t2; r[3]=t3; r[4]=t4; r[5]=t5;

    ulong d[N];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N; ++i) {
        ulong bb;
        d[i] = sbb(r[i], q[i], borrow, bb);
        borrow = bb;
    }
    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N; ++i) out[i] = use_diff ? d[i] : r[i];
}

// -------------------- Jacobian operations --------------------

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N; ++i) X[i] = src[i];
    for (uint i = 0u; i < N; ++i) Y[i] = src[N + i];
    for (uint i = 0u; i < N; ++i) Z[i] = src[2u*N + i];
}
inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    for (uint i = 0u; i < N; ++i) dst[i] = X[i];
    for (uint i = 0u; i < N; ++i) dst[N + i] = Y[i];
    for (uint i = 0u; i < N; ++i) dst[2u*N + i] = Z[i];
}
inline void zero_pt(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    fe_zero(X); fe_zero(Y); fe_zero(Z);
}

inline void jac_dbl(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                    thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                    thread const ulong *q, ulong mu)
{
    if (fe_is_zero(Z) || fe_is_zero(Y)) { zero_pt(oX, oY, oZ); return; }
    ulong A[N], B[N], C[N], D[N], E[N], F[N], t1[N], t2[N];
    fe_mul(A, X, X, q, mu);
    fe_mul(B, Y, Y, q, mu);
    fe_mul(C, B, B, q, mu);
    fe_add(t1, X, B, q);
    fe_mul(D, t1, t1, q, mu);
    fe_sub(D, D, A, q);
    fe_sub(D, D, C, q);
    fe_add(D, D, D, q);
    fe_add(E, A, A, q);
    fe_add(E, E, A, q);
    fe_mul(F, E, E, q, mu);
    fe_add(t1, D, D, q);
    fe_sub(oX, F, t1, q);
    fe_sub(t1, D, oX, q);
    fe_mul(t1, E, t1, q, mu);
    fe_add(t2, C, C, q);
    fe_add(t2, t2, t2, q);
    fe_add(t2, t2, t2, q);
    fe_sub(oY, t1, t2, q);
    fe_mul(t1, Y, Z, q, mu);
    fe_add(oZ, t1, t1, q);
}

inline void jac_add(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                    thread const ulong *X1, thread const ulong *Y1, thread const ulong *Z1,
                    thread const ulong *X2, thread const ulong *Y2, thread const ulong *Z2,
                    thread const ulong *q, ulong mu)
{
    if (fe_is_zero(Z1)) { fe_copy(oX, X2); fe_copy(oY, Y2); fe_copy(oZ, Z2); return; }
    if (fe_is_zero(Z2)) { fe_copy(oX, X1); fe_copy(oY, Y1); fe_copy(oZ, Z1); return; }
    ulong Z1Z1[N], Z2Z2[N], U1[N], U2[N], S1[N], S2[N];
    ulong H[N], R[N], HH[N], HHH[N], V[N];
    ulong t1[N], t2[N];
    fe_mul(Z1Z1, Z1, Z1, q, mu);
    fe_mul(Z2Z2, Z2, Z2, q, mu);
    fe_mul(U1,   X1, Z2Z2, q, mu);
    fe_mul(U2,   X2, Z1Z1, q, mu);
    fe_mul(t1,   Y1, Z2,   q, mu);
    fe_mul(S1,   t1, Z2Z2, q, mu);
    fe_mul(t1,   Y2, Z1,   q, mu);
    fe_mul(S2,   t1, Z1Z1, q, mu);
    if (fe_eq(U1, U2)) {
        if (fe_eq(S1, S2)) {
            jac_dbl(oX, oY, oZ, X1, Y1, Z1, q, mu);
        } else {
            zero_pt(oX, oY, oZ);
        }
        return;
    }
    fe_sub(H, U2, U1, q);
    fe_sub(R, S2, S1, q);
    fe_mul(HH,  H, H,  q, mu);
    fe_mul(HHH, H, HH, q, mu);
    fe_mul(V,   U1, HH, q, mu);
    fe_mul(oX, R, R, q, mu);
    fe_sub(oX, oX, HHH, q);
    fe_add(t1, V, V, q);
    fe_sub(oX, oX, t1, q);
    fe_sub(t1, V, oX, q);
    fe_mul(t1, R, t1, q, mu);
    fe_mul(t2, S1, HHH, q, mu);
    fe_sub(oY, t1, t2, q);
    fe_mul(t1, Z1, Z2, q, mu);
    fe_mul(oZ, t1, H, q, mu);
}

// -------------------- Kernels --------------------

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

    ulong qL[N];
    for (uint i = 0u; i < N; ++i) qL[i] = q[i];
    ulong mu = q_inv_neg;

    ulong s[4];
    s[0] = scalars[idx * 4u + 0u];
    s[1] = scalars[idx * 4u + 1u];
    s[2] = scalars[idx * 4u + 2u];
    s[3] = scalars[idx * 4u + 3u];

    ulong PX[N], PY[N], PZ[N];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    // Precompute table: tbl[k] = k * P, for k = 0..15.
    ulong tblX[TABLE_SIZE][N];
    ulong tblY[TABLE_SIZE][N];
    ulong tblZ[TABLE_SIZE][N];
    for (uint i = 0u; i < N; ++i) { tblX[0][i]=0ul; tblY[0][i]=0ul; tblZ[0][i]=0ul; }
    for (uint i = 0u; i < N; ++i) { tblX[1][i]=PX[i]; tblY[1][i]=PY[i]; tblZ[1][i]=PZ[i]; }
    {
        ulong RX[N], RY[N], RZ[N];
        jac_dbl(RX, RY, RZ, PX, PY, PZ, qL, mu);
        for (uint i = 0u; i < N; ++i) { tblX[2][i]=RX[i]; tblY[2][i]=RY[i]; tblZ[2][i]=RZ[i]; }
    }
    for (uint k = 3u; k < TABLE_SIZE; ++k) {
        ulong AX[N], AY[N], AZ[N];
        for (uint i = 0u; i < N; ++i) { AX[i]=tblX[k-1u][i]; AY[i]=tblY[k-1u][i]; AZ[i]=tblZ[k-1u][i]; }
        ulong RX[N], RY[N], RZ[N];
        jac_add(RX, RY, RZ, AX, AY, AZ, PX, PY, PZ, qL, mu);
        for (uint i = 0u; i < N; ++i) { tblX[k][i]=RX[i]; tblY[k][i]=RY[i]; tblZ[k][i]=RZ[i]; }
    }

    ulong AX[N], AY[N], AZ[N];
    zero_pt(AX, AY, AZ);
    bool started = false;
    ulong TX[N], TY[N], TZ[N];

    for (int w = 63; w >= 0; --w) {
        uint bitpos = (uint)w * WINDOW;
        uint word = bitpos >> 6u;
        uint shift = bitpos & 63u;
        uint nib = (uint)((s[word] >> shift) & 0xFul);

        if (started) {
            for (uint d = 0u; d < WINDOW; ++d) {
                jac_dbl(TX, TY, TZ, AX, AY, AZ, qL, mu);
                fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);
            }
            if (nib != 0u) {
                ulong BX[N], BY[N], BZ[N];
                for (uint i = 0u; i < N; ++i) {
                    BX[i] = tblX[nib][i];
                    BY[i] = tblY[nib][i];
                    BZ[i] = tblZ[nib][i];
                }
                jac_add(TX, TY, TZ, AX, AY, AZ, BX, BY, BZ, qL, mu);
                fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);
            }
        } else {
            if (nib != 0u) {
                for (uint i = 0u; i < N; ++i) {
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
    ulong qL[N];
    for (uint i = 0u; i < N; ++i) qL[i] = q[i];
    ulong mu = q_inv_neg;

    ulong AX[N], AY[N], AZ[N];
    ulong BX[N], BY[N], BZ[N];
    load_point(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);
    ulong RX[N], RY[N], RZ[N];
    jac_add(RX, RY, RZ, AX, AY, AZ, BX, BY, BZ, qL, mu);
    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}
```

Result of previous attempt:
           bls_N4K: correct, 111.25 ms, 0.2 Gmodmul/s (int64) (0.3% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 348.73 ms, 0.2 Gmodmul/s (int64) (0.4% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 1193.02 ms, 0.3 Gmodmul/s (int64) (0.5% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0040

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;       // 64-bit limbs (storage)
constexpr constant uint N32 = 12u;          // 32-bit limbs (arith)
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WINDOW = 4u;
constexpr constant uint TABLE_SIZE = 16u;

// -------------------- 32-bit-limb field arithmetic --------------------
// Field elements stored as 12 x uint (little-endian 32-bit limbs).

inline void fe_from64(thread uint *o, thread const ulong *a) {
    for (uint i = 0u; i < N_LIMBS; ++i) {
        o[2u*i]     = (uint)(a[i]);
        o[2u*i + 1] = (uint)(a[i] >> 32);
    }
}
inline void fe_to64(thread ulong *o, thread const uint *a) {
    for (uint i = 0u; i < N_LIMBS; ++i) {
        o[i] = ((ulong)a[2u*i]) | (((ulong)a[2u*i + 1]) << 32);
    }
}
inline void fe_copy(thread uint *o, thread const uint *a) {
    for (uint i = 0u; i < N32; ++i) o[i] = a[i];
}
inline void fe_zero(thread uint *o) {
    for (uint i = 0u; i < N32; ++i) o[i] = 0u;
}
inline bool fe_is_zero(thread const uint *a) {
    uint acc = 0u;
    for (uint i = 0u; i < N32; ++i) acc |= a[i];
    return acc == 0u;
}
inline bool fe_eq(thread const uint *a, thread const uint *b) {
    uint acc = 0u;
    for (uint i = 0u; i < N32; ++i) acc |= (a[i] ^ b[i]);
    return acc == 0u;
}

// Add with q (already-32-bit). Caller pre-computes q32.
inline void fe_add(thread uint *c, thread const uint *a, thread const uint *b,
                   thread const uint *q32)
{
    uint s[N32];
    ulong carry = 0ul;
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)a[i] + (ulong)b[i] + carry;
        s[i] = (uint)t;
        carry = t >> 32;
    }
    // Try s - q.
    uint d[N32];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)s[i] - (ulong)q32[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 32) & 1ul; // 1 if underflow
    }
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N32; ++i) c[i] = use_diff ? d[i] : s[i];
}

inline void fe_sub(thread uint *c, thread const uint *a, thread const uint *b,
                   thread const uint *q32)
{
    uint d[N32];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)a[i] - (ulong)b[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 32) & 1ul;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        for (uint i = 0u; i < N32; ++i) {
            ulong t = (ulong)d[i] + (ulong)q32[i] + carry;
            c[i] = (uint)t;
            carry = t >> 32;
        }
    } else {
        for (uint i = 0u; i < N32; ++i) c[i] = d[i];
    }
}

// CIOS Montgomery multiplication on 12 x uint with 32-bit mu (low 32 bits of q_inv_neg).
inline void fe_mul(thread uint *out, thread const uint *a, thread const uint *b,
                   thread const uint *q32, uint mu32)
{
    ulong t[N32 + 1];
    for (uint i = 0u; i < N32 + 1u; ++i) t[i] = 0ul;
    // t fits in 33 bits at each limb during the inner loop; we'll
    // normalize carries inline.

    for (uint i = 0u; i < N32; ++i) {
        // t += a * b[i]
        ulong C = 0ul;
        ulong bi = (ulong)b[i];
        for (uint j = 0u; j < N32; ++j) {
            ulong p = (ulong)a[j] * bi + (t[j] & 0xFFFFFFFFul) + C;
            t[j] = p & 0xFFFFFFFFul;
            C = p >> 32;
        }
        // t[N32] may be > 32 bits; merge.
        ulong s = t[N32] + C;
        t[N32] = s; // up to ~34 bits; fine

        // m = (t[0] * mu32) mod 2^32
        uint m = (uint)t[0] * mu32;
        ulong mm = (ulong)m;

        // t += m * q
        C = 0ul;
        for (uint j = 0u; j < N32; ++j) {
            ulong p = mm * (ulong)q32[j] + (t[j] & 0xFFFFFFFFul) + C;
            t[j] = p & 0xFFFFFFFFul;
            C = p >> 32;
        }
        s = t[N32] + C;
        // After adding m*q, t[0] must be 0; shift right by 32 bits (one 32-bit limb).
        for (uint j = 0u; j < N32; ++j) {
            t[j] = t[j + 1];
        }
        t[N32] = s >> 32;        // overflow above limb N32 is at most a few bits
        t[N32 - 1] |= (s & 0xFFFFFFFFul) << 0; // wait — need to put low 32 of s at limb N32-1? Actually after shift, original t[N32] becomes new t[N32-1], so we need to handle correctly.
        // The above two lines collectively perform: new t[N32-1] = low32(s); new t[N32] = high(s)
        t[N32 - 1] = s & 0xFFFFFFFFul;
        // (overwrite — the previous shift put t[N32] into t[N32-1], but we want s's low into t[N32-1])
    }

    // At this point t[0..N32-1] are 32-bit limbs, t[N32] is the top carry (0 or 1, possibly a few bits).
    uint r[N32];
    for (uint i = 0u; i < N32; ++i) r[i] = (uint)t[i];

    // Conditional subtraction of q.
    uint d[N32];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N32; ++i) {
        ulong tv = (ulong)r[i] - (ulong)q32[i] - borrow;
        d[i] = (uint)tv;
        borrow = (tv >> 32) & 1ul;
    }
    bool use_diff = (t[N32] != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N32; ++i) out[i] = use_diff ? d[i] : r[i];
}

// ----------- Jacobian EC ops (operate on 12 x uint field elts) -----------

inline void load_point32(thread uint *X, thread uint *Y, thread uint *Z,
                         device const ulong *src)
{
    ulong tmp[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) tmp[i] = src[i];
    fe_from64(X, tmp);
    for (uint i = 0u; i < N_LIMBS; ++i) tmp[i] = src[N_LIMBS + i];
    fe_from64(Y, tmp);
    for (uint i = 0u; i < N_LIMBS; ++i) tmp[i] = src[2u*N_LIMBS + i];
    fe_from64(Z, tmp);
}
inline void store_point32(device ulong *dst,
                          thread const uint *X, thread const uint *Y, thread const uint *Z)
{
    ulong tmp[N_LIMBS];
    fe_to64(tmp, X);
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = tmp[i];
    fe_to64(tmp, Y);
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = tmp[i];
    fe_to64(tmp, Z);
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u*N_LIMBS + i] = tmp[i];
}

inline void zero_pt(thread uint *X, thread uint *Y, thread uint *Z) {
    fe_zero(X); fe_zero(Y); fe_zero(Z);
}

inline void jac_dbl(thread uint *oX, thread uint *oY, thread uint *oZ,
                    thread const uint *X, thread const uint *Y, thread const uint *Z,
                    thread const uint *q32, uint mu32)
{
    if (fe_is_zero(Z) || fe_is_zero(Y)) { zero_pt(oX, oY, oZ); return; }
    uint A[N32], B[N32], C[N32], D[N32], E[N32], F[N32], t1[N32], t2[N32];
    fe_mul(A, X, X, q32, mu32);
    fe_mul(B, Y, Y, q32, mu32);
    fe_mul(C, B, B, q32, mu32);
    fe_add(t1, X, B, q32);
    fe_mul(D, t1, t1, q32, mu32);
    fe_sub(D, D, A, q32);
    fe_sub(D, D, C, q32);
    fe_add(D, D, D, q32);
    fe_add(E, A, A, q32);
    fe_add(E, E, A, q32);
    fe_mul(F, E, E, q32, mu32);
    fe_add(t1, D, D, q32);
    fe_sub(oX, F, t1, q32);
    fe_sub(t1, D, oX, q32);
    fe_mul(t1, E, t1, q32, mu32);
    fe_add(t2, C, C, q32);
    fe_add(t2, t2, t2, q32);
    fe_add(t2, t2, t2, q32);
    fe_sub(oY, t1, t2, q32);
    fe_mul(t1, Y, Z, q32, mu32);
    fe_add(oZ, t1, t1, q32);
}

inline void jac_add(thread uint *oX, thread uint *oY, thread uint *oZ,
                    thread const uint *X1, thread const uint *Y1, thread const uint *Z1,
                    thread const uint *X2, thread const uint *Y2, thread const uint *Z2,
                    thread const uint *q32, uint mu32)
{
    if (fe_is_zero(Z1)) {
        fe_copy(oX, X2); fe_copy(oY, Y2); fe_copy(oZ, Z2); return;
    }
    if (fe_is_zero(Z2)) {
        fe_copy(oX, X1); fe_copy(oY, Y1); fe_copy(oZ, Z1); return;
    }
    uint Z1Z1[N32], Z2Z2[N32], U1[N32], U2[N32], S1[N32], S2[N32];
    uint H[N32], R[N32], HH[N32], HHH[N32], V[N32];
    uint t1[N32], t2[N32];
    fe_mul(Z1Z1, Z1, Z1, q32, mu32);
    fe_mul(Z2Z2, Z2, Z2, q32, mu32);
    fe_mul(U1,   X1, Z2Z2, q32, mu32);
    fe_mul(U2,   X2, Z1Z1, q32, mu32);
    fe_mul(t1,   Y1, Z2,   q32, mu32);
    fe_mul(S1,   t1, Z2Z2, q32, mu32);
    fe_mul(t1,   Y2, Z1,   q32, mu32);
    fe_mul(S2,   t1, Z1Z1, q32, mu32);
    if (fe_eq(U1, U2)) {
        if (fe_eq(S1, S2)) {
            jac_dbl(oX, oY, oZ, X1, Y1, Z1, q32, mu32);
        } else {
            zero_pt(oX, oY, oZ);
        }
        return;
    }
    fe_sub(H, U2, U1, q32);
    fe_sub(R, S2, S1, q32);
    fe_mul(HH,  H, H,  q32, mu32);
    fe_mul(HHH, H, HH, q32, mu32);
    fe_mul(V,   U1, HH, q32, mu32);
    fe_mul(oX, R, R, q32, mu32);
    fe_sub(oX, oX, HHH, q32);
    fe_add(t1, V, V, q32);
    fe_sub(oX, oX, t1, q32);
    fe_sub(t1, V, oX, q32);
    fe_mul(t1, R, t1, q32, mu32);
    fe_mul(t2, S1, HHH, q32, mu32);
    fe_sub(oY, t1, t2, q32);
    fe_mul(t1, Z1, Z2, q32, mu32);
    fe_mul(oZ, t1, H, q32, mu32);
}

// -------------------- Kernels --------------------

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

    ulong qL[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qL[i] = q[i];
    uint q32[N32];
    fe_from64(q32, qL);
    uint mu32 = (uint)q_inv_neg;

    ulong s[4];
    s[0] = scalars[idx * 4u + 0u];
    s[1] = scalars[idx * 4u + 1u];
    s[2] = scalars[idx * 4u + 2u];
    s[3] = scalars[idx * 4u + 3u];

    uint PX[N32], PY[N32], PZ[N32];
    load_point32(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    // Precompute table 0..15.
    uint tblX[TABLE_SIZE][N32];
    uint tblY[TABLE_SIZE][N32];
    uint tblZ[TABLE_SIZE][N32];
    for (uint i = 0u; i < N32; ++i) { tblX[0][i] = 0u; tblY[0][i] = 0u; tblZ[0][i] = 0u; }
    for (uint i = 0u; i < N32; ++i) { tblX[1][i] = PX[i]; tblY[1][i] = PY[i]; tblZ[1][i] = PZ[i]; }
    {
        uint RX[N32], RY[N32], RZ[N32];
        jac_dbl(RX, RY, RZ, PX, PY, PZ, q32, mu32);
        for (uint i = 0u; i < N32; ++i) { tblX[2][i]=RX[i]; tblY[2][i]=RY[i]; tblZ[2][i]=RZ[i]; }
    }
    for (uint k = 3u; k < TABLE_SIZE; ++k) {
        uint AX[N32], AY[N32], AZ[N32];
        for (uint i = 0u; i < N32; ++i) { AX[i]=tblX[k-1u][i]; AY[i]=tblY[k-1u][i]; AZ[i]=tblZ[k-1u][i]; }
        uint RX[N32], RY[N32], RZ[N32];
        jac_add(RX, RY, RZ, AX, AY, AZ, PX, PY, PZ, q32, mu32);
        for (uint i = 0u; i < N32; ++i) { tblX[k][i]=RX[i]; tblY[k][i]=RY[i]; tblZ[k][i]=RZ[i]; }
    }

    uint AX[N32], AY[N32], AZ[N32];
    zero_pt(AX, AY, AZ);
    bool started = false;
    uint TX[N32], TY[N32], TZ[N32];

    for (int w = 63; w >= 0; --w) {
        uint bitpos = (uint)w * WINDOW;
        uint word = bitpos >> 6u;
        uint shift = bitpos & 63u;
        uint nib = (uint)((s[word] >> shift) & 0xFul);

        if (started) {
            for (uint d = 0u; d < WINDOW; ++d) {
                jac_dbl(TX, TY, TZ, AX, AY, AZ, q32, mu32);
                fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);
            }
            if (nib != 0u) {
                uint BX[N32], BY[N32], BZ[N32];
                for (uint i = 0u; i < N32; ++i) {
                    BX[i] = tblX[nib][i];
                    BY[i] = tblY[nib][i];
                    BZ[i] = tblZ[nib][i];
                }
                jac_add(TX, TY, TZ, AX, AY, AZ, BX, BY, BZ, q32, mu32);
                fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);
            }
        } else {
            if (nib != 0u) {
                for (uint i = 0u; i < N32; ++i) {
                    AX[i] = tblX[nib][i];
                    AY[i] = tblY[nib][i];
                    AZ[i] = tblZ[nib][i];
                }
                started = true;
            }
        }
    }

    store_point32(scratch + idx * POINT_LIMBS, AX, AY, AZ);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;
    ulong qL[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) qL[i] = q[i];
    uint q32[N32];
    fe_from64(q32, qL);
    uint mu32 = (uint)q_inv_neg;

    uint AX[N32], AY[N32], AZ[N32];
    uint BX[N32], BY[N32], BZ[N32];
    load_point32(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point32(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);
    uint RX[N32], RY[N32], RZ[N32];
    jac_add(RX, RY, RZ, AX, AY, AZ, BX, BY, BZ, q32, mu32);
    store_point32(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}
```

Incumbent result:
           bls_N4K: correct, 34.12 ms, 0.6 Gmodmul/s (int64) (1.0% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 135.95 ms, 0.6 Gmodmul/s (int64) (1.0% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 512.24 ms, 0.6 Gmodmul/s (int64) (1.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0106

## History

- iter  0: compile=OK | correct=True | score=0.0039469233524904185
- iter  1: compile=OK | correct=True | score=0.007485626493860429
- iter  2: compile=OK | correct=True | score=0.0011487006695191965
- iter  3: compile=OK | correct=True | score=0.010642882225853382
- iter  4: compile=OK | correct=True | score=0.003968420959799621
- iter  5: compile=OK | correct=True | score=0.003955388150359739

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
