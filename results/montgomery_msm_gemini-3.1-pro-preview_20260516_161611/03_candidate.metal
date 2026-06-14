#include <metal_stdlib>
using namespace metal;

// ------------------------------------------------------------------
// Optimized 64x64 + 64 + 64 -> 128 bit Multiply-Accumulate
// Recombines via 32-bit halves since Apple Silicon lacks single-instruction 
// u64 x u64 -> u128. Native (ulong)uint * uint maps to 32x32->64 perfectly.
// ------------------------------------------------------------------
inline ulong2 mac_add(ulong a, ulong b, ulong c, ulong d) {
    uint a_lo = (uint)a;
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)b;
    uint b_hi = (uint)(b >> 32);

    ulong p00 = (ulong)a_lo * b_lo;
    ulong p01 = (ulong)a_lo * b_hi;
    ulong p10 = (ulong)a_hi * b_lo;
    ulong p11 = (ulong)a_hi * b_hi;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong prod_lo = (p00 & 0xFFFFFFFFul) | (mid << 32);
    ulong prod_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    
    ulong lo1 = prod_lo + c;
    ulong cy1 = (lo1 < prod_lo) ? 1ul : 0ul;
    ulong lo2 = lo1 + d;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    
    return ulong2(lo2, prod_hi + cy1 + cy2);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
    dst[3] = src[3]; dst[4] = src[4]; dst[5] = src[5];
}

inline bool is_zero_n(thread const ulong *a) {
    return (a[0] | a[1] | a[2] | a[3] | a[4] | a[5]) == 0ul;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    return ((a[0] ^ b[0]) | (a[1] ^ b[1]) | (a[2] ^ b[2]) |
            (a[3] ^ b[3]) | (a[4] ^ b[4]) | (a[5] ^ b[5])) == 0ul;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
{
    ulong sum[6];
    ulong s, cy1, t, cy2;
    ulong carry = 0ul;

    s = a[0] + carry; cy1 = (s < a[0]) ? 1ul:0ul; t = s + b[0]; cy2 = (t < s) ? 1ul:0ul; sum[0] = t; carry = cy1 + cy2;
    s = a[1] + carry; cy1 = (s < a[1]) ? 1ul:0ul; t = s + b[1]; cy2 = (t < s) ? 1ul:0ul; sum[1] = t; carry = cy1 + cy2;
    s = a[2] + carry; cy1 = (s < a[2]) ? 1ul:0ul; t = s + b[2]; cy2 = (t < s) ? 1ul:0ul; sum[2] = t; carry = cy1 + cy2;
    s = a[3] + carry; cy1 = (s < a[3]) ? 1ul:0ul; t = s + b[3]; cy2 = (t < s) ? 1ul:0ul; sum[3] = t; carry = cy1 + cy2;
    s = a[4] + carry; cy1 = (s < a[4]) ? 1ul:0ul; t = s + b[4]; cy2 = (t < s) ? 1ul:0ul; sum[4] = t; carry = cy1 + cy2;
    s = a[5] + carry; cy1 = (s < a[5]) ? 1ul:0ul; t = s + b[5]; cy2 = (t < s) ? 1ul:0ul; sum[5] = t; carry = cy1 + cy2;

    ulong diff[6];
    ulong borrow = 0ul;
    ulong tv, b1, d, b2;

    tv = sum[0] - q[0]; b1 = (tv > sum[0]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[0] = d; borrow = b1 + b2;
    tv = sum[1] - q[1]; b1 = (tv > sum[1]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[1] = d; borrow = b1 + b2;
    tv = sum[2] - q[2]; b1 = (tv > sum[2]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[2] = d; borrow = b1 + b2;
    tv = sum[3] - q[3]; b1 = (tv > sum[3]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[3] = d; borrow = b1 + b2;
    tv = sum[4] - q[4]; b1 = (tv > sum[4]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[4] = d; borrow = b1 + b2;
    tv = sum[5] - q[5]; b1 = (tv > sum[5]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[5] = d; borrow = b1 + b2;

    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    c[0] = use_diff ? diff[0] : sum[0];
    c[1] = use_diff ? diff[1] : sum[1];
    c[2] = use_diff ? diff[2] : sum[2];
    c[3] = use_diff ? diff[3] : sum[3];
    c[4] = use_diff ? diff[4] : sum[4];
    c[5] = use_diff ? diff[5] : sum[5];
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
{
    ulong diff[6];
    ulong borrow = 0ul;
    ulong tv, b1, d, b2;

    tv = a[0] - b[0]; b1 = (tv > a[0]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[0] = d; borrow = b1 + b2;
    tv = a[1] - b[1]; b1 = (tv > a[1]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[1] = d; borrow = b1 + b2;
    tv = a[2] - b[2]; b1 = (tv > a[2]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[2] = d; borrow = b1 + b2;
    tv = a[3] - b[3]; b1 = (tv > a[3]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[3] = d; borrow = b1 + b2;
    tv = a[4] - b[4]; b1 = (tv > a[4]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[4] = d; borrow = b1 + b2;
    tv = a[5] - b[5]; b1 = (tv > a[5]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[5] = d; borrow = b1 + b2;

    ulong t_arr[6];
    ulong carry = 0ul;
    ulong s, cy1, t, cy2;

    s = diff[0] + carry; cy1 = (s < diff[0]) ? 1ul:0ul; t = s + q[0]; cy2 = (t < s) ? 1ul:0ul; t_arr[0] = t; carry = cy1 + cy2;
    s = diff[1] + carry; cy1 = (s < diff[1]) ? 1ul:0ul; t = s + q[1]; cy2 = (t < s) ? 1ul:0ul; t_arr[1] = t; carry = cy1 + cy2;
    s = diff[2] + carry; cy1 = (s < diff[2]) ? 1ul:0ul; t = s + q[2]; cy2 = (t < s) ? 1ul:0ul; t_arr[2] = t; carry = cy1 + cy2;
    s = diff[3] + carry; cy1 = (s < diff[3]) ? 1ul:0ul; t = s + q[3]; cy2 = (t < s) ? 1ul:0ul; t_arr[3] = t; carry = cy1 + cy2;
    s = diff[4] + carry; cy1 = (s < diff[4]) ? 1ul:0ul; t = s + q[4]; cy2 = (t < s) ? 1ul:0ul; t_arr[4] = t; carry = cy1 + cy2;
    s = diff[5] + carry; cy1 = (s < diff[5]) ? 1ul:0ul; t = s + q[5]; cy2 = (t < s) ? 1ul:0ul; t_arr[5] = t; carry = cy1 + cy2;

    c[0] = borrow ? t_arr[0] : diff[0];
    c[1] = borrow ? t_arr[1] : diff[1];
    c[2] = borrow ? t_arr[2] : diff[2];
    c[3] = borrow ? t_arr[3] : diff[3];
    c[4] = borrow ? t_arr[4] : diff[4];
    c[5] = borrow ? t_arr[5] : diff[5];
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;
    ulong C;
    ulong m;

    for (uint i = 0u; i < 6u; ++i) {
        ulong bi = b[i];
        C = 0ul;
        { ulong2 r = mac_add(a[0], bi, t0, C); t0 = r.x; C = r.y; }
        { ulong2 r = mac_add(a[1], bi, t1, C); t1 = r.x; C = r.y; }
        { ulong2 r = mac_add(a[2], bi, t2, C); t2 = r.x; C = r.y; }
        { ulong2 r = mac_add(a[3], bi, t3, C); t3 = r.x; C = r.y; }
        { ulong2 r = mac_add(a[4], bi, t4, C); t4 = r.x; C = r.y; }
        { ulong2 r = mac_add(a[5], bi, t5, C); t5 = r.x; C = r.y; }
        ulong s = t6 + C;
        ulong cy = (s < t6) ? 1ul : 0ul;
        t6 = s;
        ulong t7 = cy;

        m = t0 * q_inv_neg;

        C = 0ul;
        { ulong2 r = mac_add(m, q[0], t0, C); t0 = r.x; C = r.y; }
        { ulong2 r = mac_add(m, q[1], t1, C); t1 = r.x; C = r.y; }
        { ulong2 r = mac_add(m, q[2], t2, C); t2 = r.x; C = r.y; }
        { ulong2 r = mac_add(m, q[3], t3, C); t3 = r.x; C = r.y; }
        { ulong2 r = mac_add(m, q[4], t4, C); t4 = r.x; C = r.y; }
        { ulong2 r = mac_add(m, q[5], t5, C); t5 = r.x; C = r.y; }
        ulong s2 = t6 + C;
        ulong cy2 = (s2 < t6) ? 1ul : 0ul;
        t6 = s2;
        t7 += cy2;

        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7;
    }

    ulong diff[6];
    ulong borrow = 0ul;
    ulong tv, b1, d, b2;

    tv = t0 - q[0]; b1 = (tv > t0) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[0] = d; borrow = b1 + b2;
    tv = t1 - q[1]; b1 = (tv > t1) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[1] = d; borrow = b1 + b2;
    tv = t2 - q[2]; b1 = (tv > t2) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[2] = d; borrow = b1 + b2;
    tv = t3 - q[3]; b1 = (tv > t3) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[3] = d; borrow = b1 + b2;
    tv = t4 - q[4]; b1 = (tv > t4) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[4] = d; borrow = b1 + b2;
    tv = t5 - q[5]; b1 = (tv > t5) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; diff[5] = d; borrow = b1 + b2;

    bool use_diff = (t6 != 0ul) || (borrow == 0ul);
    out[0] = use_diff ? diff[0] : t0;
    out[1] = use_diff ? diff[1] : t1;
    out[2] = use_diff ? diff[2] : t2;
    out[3] = use_diff ? diff[3] : t3;
    out[4] = use_diff ? diff[4] : t4;
    out[5] = use_diff ? diff[5] : t5;
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z, device const ulong *src) {
    X[0] = src[0]; X[1] = src[1]; X[2] = src[2]; X[3] = src[3]; X[4] = src[4]; X[5] = src[5];
    Y[0] = src[6]; Y[1] = src[7]; Y[2] = src[8]; Y[3] = src[9]; Y[4] = src[10]; Y[5] = src[11];
    Z[0] = src[12]; Z[1] = src[13]; Z[2] = src[14]; Z[3] = src[15]; Z[4] = src[16]; Z[5] = src[17];
}

inline void store_point(device ulong *dst, thread const ulong *X, thread const ulong *Y, thread const ulong *Z) {
    dst[0] = X[0]; dst[1] = X[1]; dst[2] = X[2]; dst[3] = X[3]; dst[4] = X[4]; dst[5] = X[5];
    dst[6] = Y[0]; dst[7] = Y[1]; dst[8] = Y[2]; dst[9] = Y[3]; dst[10] = Y[4]; dst[11] = Y[5];
    dst[12] = Z[0]; dst[13] = Z[1]; dst[14] = Z[2]; dst[15] = Z[3]; dst[16] = Z[4]; dst[17] = Z[5];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    X[0] = 0ul; X[1] = 0ul; X[2] = 0ul; X[3] = 0ul; X[4] = 0ul; X[5] = 0ul;
    Y[0] = 0ul; Y[1] = 0ul; Y[2] = 0ul; Y[3] = 0ul; Y[4] = 0ul; Y[5] = 0ul;
    Z[0] = 0ul; Z[1] = 0ul; Z[2] = 0ul; Z[3] = 0ul; Z[4] = 0ul; Z[5] = 0ul;
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          thread const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    ulong A[6], B[6], C[6];
    ulong D[6], E[6], F[6];
    ulong tmp[6], tmp2[6];

    mont_mul(A, X, X, q, q_inv_neg);
    mont_mul(B, Y, Y, q, q_inv_neg);
    mont_mul(C, B, B, q, q_inv_neg);

    mont_mul(D, X, B, q, q_inv_neg);
    mod_add(D, D, D, q);
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
                       thread const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z1)) {
        copy_n(oX, X2); copy_n(oY, Y2); copy_n(oZ, Z2);
        return;
    }
    if (is_zero_n(Z2)) {
        copy_n(oX, X1); copy_n(oY, Y1); copy_n(oZ, Z1);
        return;
    }
    
    ulong Z1Z1[6], Z2Z2[6];
    ulong U1[6], U2[6], S1[6], S2[6];
    ulong H[6], R[6];
    ulong HH[6], HHH[6], V[6];
    ulong tmp[6], tmp2[6];

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

    ulong q_local[6];
    q_local[0] = q[0]; q_local[1] = q[1]; q_local[2] = q[2]; 
    q_local[3] = q[3]; q_local[4] = q[4]; q_local[5] = q[5];

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    ulong naf0=0, naf1=0, naf2=0, naf3=0, naf4=0, naf5=0, naf6=0, naf7=0, naf8=0;
    int len = 0;
    
    while ((s0 | s1 | s2 | s3) != 0ul) {
        if (s0 & 1ul) {
            int bit = s0 & 3ul;
            ulong val = (bit == 3) ? 2ul : 1ul;
            
            int idx_naf = len / 32;
            int shift = (len % 32) * 2;
            if (idx_naf == 0) naf0 |= (val << shift);
            else if (idx_naf == 1) naf1 |= (val << shift);
            else if (idx_naf == 2) naf2 |= (val << shift);
            else if (idx_naf == 3) naf3 |= (val << shift);
            else if (idx_naf == 4) naf4 |= (val << shift);
            else if (idx_naf == 5) naf5 |= (val << shift);
            else if (idx_naf == 6) naf6 |= (val << shift);
            else if (idx_naf == 7) naf7 |= (val << shift);
            else if (idx_naf == 8) naf8 |= (val << shift);
            
            if (bit == 3) {
                ulong cy = 1ul;
                ulong n0 = s0 + cy; cy = (n0 < s0) ? 1ul : 0ul; s0 = n0;
                ulong n1 = s1 + cy; cy = (n1 < s1) ? 1ul : 0ul; s1 = n1;
                ulong n2 = s2 + cy; cy = (n2 < s2) ? 1ul : 0ul; s2 = n2;
                s3 += cy;
            } else {
                s0 -= 1ul;
            }
        }
        len++;
        s0 = (s0 >> 1) | (s1 << 63);
        s1 = (s1 >> 1) | (s2 << 63);
        s2 = (s2 >> 1) | (s3 << 63);
        s3 >>= 1;
    }

    ulong PX[6], PY[6], PZ[6];
    load_point(PX, PY, PZ, points_in + idx * 18u);

    ulong nPY[6];
    if (is_zero_n(PY)) {
        copy_n(nPY, PY);
    } else {
        ulong borrow = 0ul;
        ulong tv, b1, d, b2;
        tv = q_local[0] - PY[0]; b1 = (tv > q_local[0]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; nPY[0] = d; borrow = b1 + b2;
        tv = q_local[1] - PY[1]; b1 = (tv > q_local[1]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; nPY[1] = d; borrow = b1 + b2;
        tv = q_local[2] - PY[2]; b1 = (tv > q_local[2]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; nPY[2] = d; borrow = b1 + b2;
        tv = q_local[3] - PY[3]; b1 = (tv > q_local[3]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; nPY[3] = d; borrow = b1 + b2;
        tv = q_local[4] - PY[4]; b1 = (tv > q_local[4]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; nPY[4] = d; borrow = b1 + b2;
        tv = q_local[5] - PY[5]; b1 = (tv > q_local[5]) ? 1ul:0ul; d = tv - borrow; b2 = (d > tv) ? 1ul:0ul; nPY[5] = d; borrow = b1 + b2;
    }

    ulong AX[6], AY[6], AZ[6];
    zero_point(AX, AY, AZ);
    ulong TX[6], TY[6], TZ[6];

    if (len > 0) {
        for (int j = len - 1; j >= 0; --j) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            
            int idx_naf = j / 32;
            int shift = (j % 32) * 2;
            ulong word = (idx_naf == 0) ? naf0 :
                         (idx_naf == 1) ? naf1 :
                         (idx_naf == 2) ? naf2 :
                         (idx_naf == 3) ? naf3 :
                         (idx_naf == 4) ? naf4 :
                         (idx_naf == 5) ? naf5 :
                         (idx_naf == 6) ? naf6 :
                         (idx_naf == 7) ? naf7 : naf8;
            ulong d = (word >> shift) & 3ul;
            
            if (d == 1ul) {
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_local, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            } else if (d == 2ul) {
                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, nPY, PZ, q_local, q_inv_neg);
                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
            }
        }
    }

    store_point(scratch + idx * 18u, AX, AY, AZ);
}

kernel void montgomery_msm_reduce(
    device       ulong *scratch      [[buffer(0)]],
    device const ulong *q            [[buffer(1)]],
    constant ulong     &q_inv_neg    [[buffer(2)]],
    constant uint      &half_count   [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= half_count) return;

    ulong q_local[6];
    q_local[0] = q[0]; q_local[1] = q[1]; q_local[2] = q[2]; 
    q_local[3] = q[3]; q_local[4] = q[4]; q_local[5] = q[5];

    ulong AX[6], AY[6], AZ[6];
    ulong BX[6], BY[6], BZ[6];
    load_point(AX, AY, AZ, scratch + idx * 18u);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * 18u);

    ulong RX[6], RY[6], RZ[6];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q_local, q_inv_neg);
               
    store_point(scratch + idx * 18u, RX, RY, RZ);
}