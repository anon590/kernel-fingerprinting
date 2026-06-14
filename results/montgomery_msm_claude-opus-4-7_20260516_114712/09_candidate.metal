#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint N32 = 12u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint WINDOW = 4u;
constexpr constant uint TABLE_SIZE = 16u;

inline void fe_from64(thread uint *o, thread const ulong *a) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        o[2u*i]     = (uint)(a[i]);
        o[2u*i + 1] = (uint)(a[i] >> 32);
    }
}
inline void fe_to64(thread ulong *o, thread const uint *a) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        o[i] = ((ulong)a[2u*i]) | (((ulong)a[2u*i + 1]) << 32);
    }
}
inline void fe_copy(thread uint *o, thread const uint *a) {
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) o[i] = a[i];
}
inline void fe_zero(thread uint *o) {
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) o[i] = 0u;
}
inline bool fe_is_zero(thread const uint *a) {
    uint acc = 0u;
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) acc |= a[i];
    return acc == 0u;
}
inline bool fe_eq(thread const uint *a, thread const uint *b) {
    uint acc = 0u;
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) acc |= (a[i] ^ b[i]);
    return acc == 0u;
}

inline void fe_add(thread uint *c, thread const uint *a, thread const uint *b,
                   thread const uint *q32)
{
    uint s[N32];
    ulong carry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)a[i] + (ulong)b[i] + carry;
        s[i] = (uint)t;
        carry = t >> 32;
    }
    uint d[N32];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)s[i] - (ulong)q32[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 32) & 1ul;
    }
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) c[i] = use_diff ? d[i] : s[i];
}

inline void fe_sub(thread uint *c, thread const uint *a, thread const uint *b,
                   thread const uint *q32)
{
    uint d[N32];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)a[i] - (ulong)b[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 32) & 1ul;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        #pragma unroll
        for (uint i = 0u; i < N32; ++i) {
            ulong t = (ulong)d[i] + (ulong)q32[i] + carry;
            c[i] = (uint)t;
            carry = t >> 32;
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < N32; ++i) c[i] = d[i];
    }
}

// CIOS Montgomery multiplication: 32-bit-limb Acar form.
// t[0..N32] are 32-bit; one extra word t[N32] holds the top carry (<= 1 typically).
inline void fe_mul(thread uint *out, thread const uint *a, thread const uint *b,
                   thread const uint *q32, uint mu32)
{
    uint t[N32 + 2];
    #pragma unroll
    for (uint i = 0u; i < N32 + 2u; ++i) t[i] = 0u;

    #pragma unroll
    for (uint i = 0u; i < N32; ++i) {
        ulong bi = (ulong)b[i];
        ulong C = 0ul;
        // t[j] += a[j] * bi + C
        #pragma unroll
        for (uint j = 0u; j < N32; ++j) {
            ulong p = (ulong)a[j] * bi + (ulong)t[j] + C;
            t[j] = (uint)p;
            C = p >> 32;
        }
        ulong s = (ulong)t[N32] + C;
        t[N32]     = (uint)s;
        t[N32 + 1] = t[N32 + 1] + (uint)(s >> 32);

        // m = t[0] * mu32 mod 2^32
        uint m = t[0] * mu32;
        ulong mm = (ulong)m;

        // t[0] += m * q[0]  (low 32 should become 0)
        ulong p0 = mm * (ulong)q32[0] + (ulong)t[0];
        C = p0 >> 32;
        // t[0] discarded (will be shifted out).
        #pragma unroll
        for (uint j = 1u; j < N32; ++j) {
            ulong p = mm * (ulong)q32[j] + (ulong)t[j] + C;
            t[j - 1] = (uint)p;     // shift-right by one limb on the fly
            C = p >> 32;
        }
        s = (ulong)t[N32] + C;
        t[N32 - 1] = (uint)s;
        t[N32]     = t[N32 + 1] + (uint)(s >> 32);
        t[N32 + 1] = 0u;
    }

    uint r[N32];
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) r[i] = t[i];

    // Conditional subtract q.
    uint d[N32];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) {
        ulong tv = (ulong)r[i] - (ulong)q32[i] - borrow;
        d[i] = (uint)tv;
        borrow = (tv >> 32) & 1ul;
    }
    bool use_diff = (t[N32] != 0u) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) out[i] = use_diff ? d[i] : r[i];
}

inline void load_point32(thread uint *X, thread uint *Y, thread uint *Z,
                         device const ulong *src)
{
    ulong tmp[N_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) tmp[i] = src[i];
    fe_from64(X, tmp);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) tmp[i] = src[N_LIMBS + i];
    fe_from64(Y, tmp);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) tmp[i] = src[2u*N_LIMBS + i];
    fe_from64(Z, tmp);
}
inline void store_point32(device ulong *dst,
                          thread const uint *X, thread const uint *Y, thread const uint *Z)
{
    ulong tmp[N_LIMBS];
    fe_to64(tmp, X);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = tmp[i];
    fe_to64(tmp, Y);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = tmp[i];
    fe_to64(tmp, Z);
    #pragma unroll
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
    #pragma unroll
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

    uint tblX[TABLE_SIZE][N32];
    uint tblY[TABLE_SIZE][N32];
    uint tblZ[TABLE_SIZE][N32];
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) { tblX[0][i] = 0u; tblY[0][i] = 0u; tblZ[0][i] = 0u; }
    #pragma unroll
    for (uint i = 0u; i < N32; ++i) { tblX[1][i] = PX[i]; tblY[1][i] = PY[i]; tblZ[1][i] = PZ[i]; }
    {
        uint RX[N32], RY[N32], RZ[N32];
        jac_dbl(RX, RY, RZ, PX, PY, PZ, q32, mu32);
        #pragma unroll
        for (uint i = 0u; i < N32; ++i) { tblX[2][i]=RX[i]; tblY[2][i]=RY[i]; tblZ[2][i]=RZ[i]; }
    }
    for (uint k = 3u; k < TABLE_SIZE; ++k) {
        uint AX[N32], AY[N32], AZ[N32];
        #pragma unroll
        for (uint i = 0u; i < N32; ++i) { AX[i]=tblX[k-1u][i]; AY[i]=tblY[k-1u][i]; AZ[i]=tblZ[k-1u][i]; }
        uint RX[N32], RY[N32], RZ[N32];
        jac_add(RX, RY, RZ, AX, AY, AZ, PX, PY, PZ, q32, mu32);
        #pragma unroll
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
            #pragma unroll
            for (uint d = 0u; d < WINDOW; ++d) {
                jac_dbl(TX, TY, TZ, AX, AY, AZ, q32, mu32);
                fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);
            }
            if (nib != 0u) {
                uint BX[N32], BY[N32], BZ[N32];
                #pragma unroll
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
                #pragma unroll
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
    #pragma unroll
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