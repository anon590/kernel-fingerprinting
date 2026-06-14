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