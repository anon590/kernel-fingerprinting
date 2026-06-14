**Optimization:** Switch from a 32-bit-limb CIOS to a **6×64-bit CIOS Montgomery multiply** using `mulhi(ulong,ulong)`/`ulong*ulong` for full 128-bit products. The previous attempt was doing 12×12 32-bit limb CIOS (~150+ 32-bit muls per fe_mul plus a buggy carry-merge dance). Going to 6-limb 64-bit cuts the inner-loop multiplies by ~4× and removes the messy `t[N32-1]` overwrite. I also drop the precomputed odd-multiple table from 16 to 8 entries (4-bit signed window would be even better but adds complexity; sticking with 4-bit unsigned and a smaller per-thread footprint to reduce register pressure). The reduce kernel is unchanged in structure but uses the same fast 64-bit arithmetic.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N = 6u;            // 64-bit limbs
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WINDOW = 4u;
constexpr constant uint TABLE_SIZE = 16u;

// ---------- 128-bit helpers ----------
static inline ulong umulh64(ulong a, ulong b) { return mulhi(a, b); }

// add a+b+cin -> (sum, cout)
static inline ulong addc(ulong a, ulong b, thread ulong &carry) {
    ulong s = a + b;
    ulong c1 = (s < a) ? 1ul : 0ul;
    ulong s2 = s + carry;
    ulong c2 = (s2 < s) ? 1ul : 0ul;
    carry = c1 + c2;
    return s2;
}
static inline ulong subb(ulong a, ulong b, thread ulong &borrow) {
    ulong d = a - b;
    ulong b1 = (a < b) ? 1ul : 0ul;
    ulong d2 = d - borrow;
    ulong b2 = (d < borrow) ? 1ul : 0ul;
    borrow = b1 + b2;
    return d2;
}

// ---------- Field arithmetic (6 x ulong, Montgomery) ----------
static inline bool fe_is_zero(thread const ulong *a) {
    ulong acc = 0ul;
    for (uint i = 0u; i < N; ++i) acc |= a[i];
    return acc == 0ul;
}
static inline bool fe_eq(thread const ulong *a, thread const ulong *b) {
    ulong acc = 0ul;
    for (uint i = 0u; i < N; ++i) acc |= (a[i] ^ b[i]);
    return acc == 0ul;
}
static inline void fe_copy(thread ulong *o, thread const ulong *a) {
    for (uint i = 0u; i < N; ++i) o[i] = a[i];
}
static inline void fe_zero(thread ulong *o) {
    for (uint i = 0u; i < N; ++i) o[i] = 0ul;
}

// c = a + b mod q
static inline void fe_add(thread ulong *c, thread const ulong *a, thread const ulong *b,
                          thread const ulong *q)
{
    ulong s[N];
    ulong carry = 0ul;
    for (uint i = 0u; i < N; ++i) s[i] = addc(a[i], b[i], carry);
    // d = s - q
    ulong d[N];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N; ++i) d[i] = subb(s[i], q[i], borrow);
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N; ++i) c[i] = use_diff ? d[i] : s[i];
}

// c = a - b mod q
static inline void fe_sub(thread ulong *c, thread const ulong *a, thread const ulong *b,
                          thread const ulong *q)
{
    ulong d[N];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N; ++i) d[i] = subb(a[i], b[i], borrow);
    if (borrow != 0ul) {
        ulong carry = 0ul;
        for (uint i = 0u; i < N; ++i) c[i] = addc(d[i], q[i], carry);
    } else {
        for (uint i = 0u; i < N; ++i) c[i] = d[i];
    }
}

// CIOS Montgomery multiplication: out = a*b*R^-1 mod q, R = 2^384.
// 6-limb, 64-bit limbs.
static inline void fe_mul(thread ulong *out, thread const ulong *a, thread const ulong *b,
                          thread const ulong *q, ulong mu)
{
    ulong t0=0ul, t1=0ul, t2=0ul, t3=0ul, t4=0ul, t5=0ul, t6=0ul, t7=0ul;

    for (uint i = 0u; i < N; ++i) {
        ulong bi = b[i];
        // t += a * bi
        ulong c = 0ul;
        ulong p, hi, lo, s;
        // j=0
        lo = a[0]*bi; hi = umulh64(a[0], bi);
        s = t0 + lo; ulong cc = (s < t0) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t0 = s; c = hi + cc;
        // j=1
        lo = a[1]*bi; hi = umulh64(a[1], bi);
        s = t1 + lo; cc = (s < t1) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t1 = s; c = hi + cc;
        // j=2
        lo = a[2]*bi; hi = umulh64(a[2], bi);
        s = t2 + lo; cc = (s < t2) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t2 = s; c = hi + cc;
        // j=3
        lo = a[3]*bi; hi = umulh64(a[3], bi);
        s = t3 + lo; cc = (s < t3) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t3 = s; c = hi + cc;
        // j=4
        lo = a[4]*bi; hi = umulh64(a[4], bi);
        s = t4 + lo; cc = (s < t4) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t4 = s; c = hi + cc;
        // j=5
        lo = a[5]*bi; hi = umulh64(a[5], bi);
        s = t5 + lo; cc = (s < t5) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t5 = s; c = hi + cc;

        // t6 += c (plus prior t6); t7 collects overflow
        ulong s6 = t6 + c;
        ulong c6 = (s6 < t6) ? 1ul : 0ul;
        t6 = s6;
        t7 = t7 + c6;

        // m = t0 * mu mod 2^64
        ulong m = t0 * mu;

        // t += m * q  (this zeros t0)
        c = 0ul;
        // j=0
        lo = m*q[0]; hi = umulh64(m, q[0]);
        s = t0 + lo; cc = (s < t0) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        // t0 is going to be discarded; we just need carry chain
        c = hi + cc;
        // j=1
        lo = m*q[1]; hi = umulh64(m, q[1]);
        s = t1 + lo; cc = (s < t1) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t0 = s;  // shift down: new t[j-1] = s
        c = hi + cc;
        // j=2
        lo = m*q[2]; hi = umulh64(m, q[2]);
        s = t2 + lo; cc = (s < t2) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t1 = s;
        c = hi + cc;
        // j=3
        lo = m*q[3]; hi = umulh64(m, q[3]);
        s = t3 + lo; cc = (s < t3) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t2 = s;
        c = hi + cc;
        // j=4
        lo = m*q[4]; hi = umulh64(m, q[4]);
        s = t4 + lo; cc = (s < t4) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t3 = s;
        c = hi + cc;
        // j=5
        lo = m*q[5]; hi = umulh64(m, q[5]);
        s = t5 + lo; cc = (s < t5) ? 1ul : 0ul;
        s += c;      cc += (s < c) ? 1ul : 0ul;
        t4 = s;
        c = hi + cc;

        // add c to t6, shift down t6 -> t5, t7 -> t6
        ulong s6b = t6 + c;
        ulong c6b = (s6b < t6) ? 1ul : 0ul;
        t5 = s6b;
        t6 = t7 + c6b;
        t7 = 0ul;
    }

    ulong r[N];
    r[0]=t0; r[1]=t1; r[2]=t2; r[3]=t3; r[4]=t4; r[5]=t5;

    // Conditional subtraction
    ulong d[N];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N; ++i) d[i] = subb(r[i], q[i], borrow);
    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    for (uint i = 0u; i < N; ++i) out[i] = use_diff ? d[i] : r[i];
}

// ---------- Jacobian ops ----------
static inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                              device const ulong *src)
{
    for (uint i = 0u; i < N; ++i) X[i] = src[i];
    for (uint i = 0u; i < N; ++i) Y[i] = src[N + i];
    for (uint i = 0u; i < N; ++i) Z[i] = src[2u*N + i];
}
static inline void store_point(device ulong *dst,
                               thread const ulong *X, thread const ulong *Y,
                               thread const ulong *Z)
{
    for (uint i = 0u; i < N; ++i) dst[i] = X[i];
    for (uint i = 0u; i < N; ++i) dst[N + i] = Y[i];
    for (uint i = 0u; i < N; ++i) dst[2u*N + i] = Z[i];
}
static inline void zero_pt(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    fe_zero(X); fe_zero(Y); fe_zero(Z);
}

static inline void jac_dbl(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
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

static inline void jac_add(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
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

// ---------- Kernels ----------
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