#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint SCALAR_BITS = 256u;
constexpr constant uint WINDOW = 4u;            // 4-bit signed window
constexpr constant uint TABLE_SIZE = 8u;        // 2^(WINDOW-1) odd multiples

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
    ulong r = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) r |= a[i];
    return r == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    ulong r = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) r |= (a[i] ^ b[i]);
    return r == 0ul;
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

// c = q - a (negation mod q). a assumed canonical (< q). If a == 0, c == 0.
inline void mod_neg(thread ulong *c, thread const ulong *a, device const ulong *q) {
    if (is_zero_n(a)) {
        for (uint i = 0u; i < N_LIMBS; ++i) c[i] = 0ul;
        return;
    }
    ulong borrow = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = q[i] - a[i];
        ulong b1 = (tv > q[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        c[i] = d;
        borrow = b1 + b2;
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
// Kernel A: per-pair scalar multiplication using 4-bit signed window.
// Precomputes odd multiples [P, 3P, 5P, ..., 15P] (8 points).
// Recodes scalar into signed 4-bit digits (booth-style) and
// processes from MSB to LSB.
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

    // Load scalar limbs.
    ulong s[4];
    s[0] = scalars[idx * 4u + 0u];
    s[1] = scalars[idx * 4u + 1u];
    s[2] = scalars[idx * 4u + 2u];
    s[3] = scalars[idx * 4u + 3u];

    // Load base point P.
    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    // Precompute table: tbl[i] = (2i+1)*P, for i=0..7.
    // Storage: tableX[i][limb], tableY[i][limb], tableZ[i][limb].
    ulong tX[TABLE_SIZE * N_LIMBS];
    ulong tY[TABLE_SIZE * N_LIMBS];
    ulong tZ[TABLE_SIZE * N_LIMBS];

    // tbl[0] = P
    for (uint j = 0u; j < N_LIMBS; ++j) {
        tX[j] = PX[j];
        tY[j] = PY[j];
        tZ[j] = PZ[j];
    }

    // Compute 2P
    ulong P2X[N_LIMBS], P2Y[N_LIMBS], P2Z[N_LIMBS];
    jac_double_pt(P2X, P2Y, P2Z, PX, PY, PZ, q, q_inv_neg);

    // tbl[i] = tbl[i-1] + 2P
    ulong curX[N_LIMBS], curY[N_LIMBS], curZ[N_LIMBS];
    copy_n(curX, PX); copy_n(curY, PY); copy_n(curZ, PZ);
    for (uint i = 1u; i < TABLE_SIZE; ++i) {
        ulong nX[N_LIMBS], nY[N_LIMBS], nZ[N_LIMBS];
        jac_add_pt(nX, nY, nZ, curX, curY, curZ, P2X, P2Y, P2Z, q, q_inv_neg);
        for (uint j = 0u; j < N_LIMBS; ++j) {
            tX[i * N_LIMBS + j] = nX[j];
            tY[i * N_LIMBS + j] = nY[j];
            tZ[i * N_LIMBS + j] = nZ[j];
        }
        copy_n(curX, nX); copy_n(curY, nY); copy_n(curZ, nZ);
    }

    // Accumulator A = O.
    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    // Helper: extract bit b of scalar (0..255).
    // Process from MSB to LSB, but in 4-bit windows with signed digits.
    // Algorithm: scan 4 bits at a time. Use simple windowed method:
    // for each 4-bit window from top, do 4 doublings, then add nibble*P.
    // We use signed digits to halve the table size: nibble in [0,15]
    // If nibble <= 8: digit = nibble (use tbl[(d-1)/2] for odd d, double-and-add for even)
    // Simpler: just use the full 16-entry concept but with 8 odd entries
    // and handle sign for window approach.
    //
    // Use Joye-Tunstall style: process windows, where each digit is signed odd
    // in {-15,-13,...,-1,1,3,...,15}. This requires variable-length recoding.
    //
    // Simpler choice: fixed-window of width 4. Digits in [0,15].
    // For digit d > 0: A += d*P. Precompute all 16 multiples? Or use signed.
    // Use signed window: for each window d in [0,15], if d > 8, d = d - 16,
    // and propagate carry to next higher window (i.e. add 1 to next digit).
    // This makes digits in [-8, 8]. Odd digit handling:
    // For d in [-8,8], skip if 0; else lookup tbl[(|d|-1)/2] if odd, otherwise
    // we need a different approach.
    //
    // Cleanest: standard 4-bit fixed window, digits [0,15], 16-entry table,
    // but only precompute 8 odd entries. Convert to signed [-8,8] then handle
    // even digits by extracting power of 2.
    //
    // Even simpler and still good: do fixed 4-bit window with 15 precomputed
    // multiples. To save space, use signed: precompute only odd multiples
    // [1,3,5,7,9,11,13,15] (8 entries), recode scalar so every digit is odd.
    // We use "modified booth" / NAF_w with w=4:
    //
    // Process from LSB. While k != 0:
    //   if k is odd:
    //     d = k mod 2^w; if d >= 2^(w-1): d -= 2^w
    //     k -= d
    //   else:
    //     d = 0
    //   output d
    //   k >>= 1
    //
    // This is wNAF. But it requires variable-length output. Since scalar is
    // <= 256 bits, output has at most 257 digits. Each odd digit |d| in {1,3,..,15}.
    //
    // Then accumulator scan from MSB: for each digit, double; if d != 0, add d*P.
    //
    // Number of non-zero digits ~ 256/(w+1) = ~51. Each non-zero adds one jac_add.
    // Plus 256 doublings. Much better than 128 adds in binary method.

    // Compute wNAF digits. We need int8 digits, signed.
    // Max length: 257.
    char naf[260];
    uint naf_len = 0u;
    {
        // Local copy of scalar.
        ulong k[4];
        k[0] = s[0]; k[1] = s[1]; k[2] = s[2]; k[3] = s[3];
        const uint w = WINDOW;
        const int half_pow = 1 << (w - 1u);     // 8
        const int full_pow = 1 << w;            // 16
        const ulong mask_w = (1ul << w) - 1ul;  // 0xF

        // Up to 257 iterations max.
        for (uint i = 0u; i < 260u; ++i) {
            // Check if k is zero.
            if ((k[0] | k[1] | k[2] | k[3]) == 0ul) {
                naf_len = i;
                break;
            }
            int d = 0;
            if ((k[0] & 1ul) != 0ul) {
                // d = k mod 2^w (low w bits)
                d = (int)(k[0] & mask_w);
                if (d >= half_pow) d -= full_pow;
                // k -= d. If d > 0, subtract; if d < 0, add.
                if (d > 0) {
                    ulong sub = (ulong)d;
                    ulong borrow = 0ul;
                    for (uint j = 0u; j < 4u; ++j) {
                        ulong sv = (j == 0u) ? sub : 0ul;
                        ulong tv = k[j] - sv;
                        ulong b1 = (tv > k[j]) ? 1ul : 0ul;
                        ulong nv = tv - borrow;
                        ulong b2 = (nv > tv) ? 1ul : 0ul;
                        k[j] = nv;
                        borrow = b1 + b2;
                    }
                } else {
                    ulong add = (ulong)(-d);
                    ulong carry = 0ul;
                    for (uint j = 0u; j < 4u; ++j) {
                        ulong av = (j == 0u) ? add : 0ul;
                        ulong s1 = k[j] + carry;
                        ulong cy1 = (s1 < k[j]) ? 1ul : 0ul;
                        ulong t1 = s1 + av;
                        ulong cy2 = (t1 < s1) ? 1ul : 0ul;
                        k[j] = t1;
                        carry = cy1 + cy2;
                    }
                }
            }
            naf[i] = (char)d;
            // k >>= 1
            k[0] = (k[0] >> 1) | (k[1] << 63);
            k[1] = (k[1] >> 1) | (k[2] << 63);
            k[2] = (k[2] >> 1) | (k[3] << 63);
            k[3] = k[3] >> 1;
            naf_len = i + 1u;
        }
    }

    // Scan from MSB.
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
    if (naf_len > 0u) {
        for (int i = (int)naf_len - 1; i >= 0; --i) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            int d = (int)naf[i];
            if (d != 0) {
                int ad = d > 0 ? d : -d;
                uint tidx = (uint)((ad - 1) >> 1);
                ulong QX[N_LIMBS], QY[N_LIMBS], QZ[N_LIMBS];
                for (uint j = 0u; j < N_LIMBS; ++j) {
                    QX[j] = tX[tidx * N_LIMBS + j];
                    QZ[j] = tZ[tidx * N_LIMBS + j];
                }
                if (d > 0) {
                    for (uint j = 0u; j < N_LIMBS; ++j) {
                        QY[j] = tY[tidx * N_LIMBS + j];
                    }
                } else {
                    ulong tmpY[N_LIMBS];
                    for (uint j = 0u; j < N_LIMBS; ++j) {
                        tmpY[j] = tY[tidx * N_LIMBS + j];
                    }
                    mod_neg(QY, tmpY, q);
                }
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, QX, QY, QZ, q, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
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
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q, q_inv_neg);
    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}