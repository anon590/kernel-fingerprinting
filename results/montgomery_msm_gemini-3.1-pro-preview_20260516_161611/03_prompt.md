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

inline ulong2 umul128(ulong x, ulong y) {
    ulong lo = x * y;
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32);
    uint y0 = (uint)y;
    uint y1 = (uint)(y >> 32);
    
    ulong p01 = (ulong)x0 * y1;
    ulong p10 = (ulong)x1 * y0;
    ulong p11 = (ulong)x1 * y1;
    
    ulong mid = p01 + (uint)p10 + (((ulong)x0 * y0) >> 32);
    ulong hi = p11 + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = src[i];
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
    ulong sum[N_LIMBS];
    ulong carry = 0ul;
    #pragma unroll
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
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = (tv > sum[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    ulong t_arr[N_LIMBS];
    ulong carry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong s = diff[i] + carry;
        ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
        ulong t = s + q[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        t_arr[i] = t;
        carry = cy1 + cy2;
    }
    
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        c[i] = borrow ? t_arr[i] : diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[7] = {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul};

    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong C = 0ul;
        ulong bi = b[i];
        
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = umul128(a[j], bi);
            ulong lo1 = r.x + t[j];
            ulong cy1 = (lo1 < r.x) ? 1ul : 0ul;
            ulong lo2 = lo1 + C;
            ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
            t[j] = lo2;
            C = r.y + cy1 + cy2;
        }
        ulong s1 = t[6] + C;
        ulong cy1 = (s1 < t[6]) ? 1ul : 0ul;
        t[6] = s1;
        ulong t7 = cy1;

        ulong m = t[0] * q_inv_neg;

        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            ulong2 r = umul128(m, q[j]);
            ulong lo1 = r.x + t[j];
            ulong cy2 = (lo1 < r.x) ? 1ul : 0ul;
            ulong lo2 = lo1 + C;
            ulong cy3 = (lo2 < lo1) ? 1ul : 0ul;
            t[j] = lo2;
            C = r.y + cy2 + cy3;
        }
        ulong s2 = t[6] + C;
        ulong cy4 = (s2 < t[6]) ? 1ul : 0ul;
        t[6] = s2;
        t7 += cy4;

        #pragma unroll
        for (uint j = 0u; j < N_LIMBS; ++j) {
            t[j] = t[j + 1];
        }
        t[6] = t7;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    
    bool use_diff = (t[6] != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
    }
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = X[i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = Y[i];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u * N_LIMBS + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = 0ul;
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = 0ul;
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          thread const ulong *q, ulong q_inv_neg)
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

    // D = 4 * X * B
    mont_mul(D, X, B, q, q_inv_neg);
    mod_add(D, D, D, q);
    mod_add(D, D, D, q);

    // E = 3 * A
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

    ulong q_local[N_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) q_local[i] = q[i];

    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];
    ulong s[4] = {s0, s1, s2, s3};

    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    ulong nPY[N_LIMBS];
    if (is_zero_n(PY)) {
        copy_n(nPY, PY);
    } else {
        ulong borrow = 0ul;
        #pragma unroll
        for (uint i = 0u; i < N_LIMBS; ++i) {
            ulong tv = q_local[i] - PY[i];
            ulong b1 = (tv > q_local[i]) ? 1ul : 0ul;
            ulong d = tv - borrow;
            ulong b2 = (d > tv) ? 1ul : 0ul;
            nPY[i] = d;
            borrow = b1 + b2;
        }
    }

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    ulong naf[9] = {0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul, 0ul};
    int len = 0;
    while ((s[0] | s[1] | s[2] | s[3]) != 0ul) {
        if ((s[0] & 1ul) != 0ul) {
            int d = 2 - (int)(s[0] & 3ul);
            ulong val = (d == 1) ? 1ul : 2ul;
            naf[len / 32] |= (val << ((len % 32) * 2));
            
            if (d == 1) {
                ulong b0 = s[0] - 1ul; ulong br = (b0 > s[0]) ? 1ul : 0ul; s[0] = b0;
                ulong b1 = s[1] - br;  br = (b1 > s[1]) ? 1ul : 0ul;  s[1] = b1;
                ulong b2 = s[2] - br;  br = (b2 > s[2]) ? 1ul : 0ul;  s[2] = b2;
                s[3] -= br;
            } else {
                ulong a0 = s[0] + 1ul; ulong cy = (a0 < s[0]) ? 1ul : 0ul; s[0] = a0;
                ulong a1 = s[1] + cy;  cy = (a1 < s[1]) ? 1ul : 0ul;  s[1] = a1;
                ulong a2 = s[2] + cy;  cy = (a2 < s[2]) ? 1ul : 0ul;  s[2] = a2;
                s[3] += cy;
            }
        }
        len++;
        s[0] = (s[0] >> 1) | (s[1] << 63);
        s[1] = (s[1] >> 1) | (s[2] << 63);
        s[2] = (s[2] >> 1) | (s[3] << 63);
        s[3] >>= 1;
    }

    if (len > 0) {
        ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
        for (uint bit = (uint)(len - 1); bit != 0xFFFFFFFFu; --bit) {
            jac_double_pt(TX, TY, TZ, AX, AY, AZ, q_local, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);

            ulong val = (naf[bit / 32] >> ((bit % 32) * 2)) & 3ul;
            if (val != 0ul) {
                if (val == 1ul) {
                    jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_local, q_inv_neg);
                } else {
                    jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, nPY, PZ, q_local, q_inv_neg);
                }
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

    ulong q_local[N_LIMBS];
    #pragma unroll
    for (uint i = 0u; i < N_LIMBS; ++i) q_local[i] = q[i];

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    ulong BX[N_LIMBS], BY[N_LIMBS], BZ[N_LIMBS];
    load_point(AX, AY, AZ, scratch + idx * POINT_LIMBS);
    load_point(BX, BY, BZ, scratch + (idx + half_count) * POINT_LIMBS);

    ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];
    jac_add_pt(RX, RY, RZ,
               AX, AY, AZ,
               BX, BY, BZ,
               q_local, q_inv_neg);
               
    store_point(scratch + idx * POINT_LIMBS, RX, RY, RZ);
}
```

Result of previous attempt:
           bls_N4K: correct, 308.12 ms, 0.1 Gmodmul/s (int64) (0.1% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 944.41 ms, 0.1 Gmodmul/s (int64) (0.2% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 3484.19 ms, 0.1 Gmodmul/s (int64) (0.2% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0014

## Current best (incumbent)

```metal
// Naive seed for multi-scalar multiplication.
//
// One thread per (scalar, point) pair runs MSB-to-LSB double-and-add
// to compute t_i = s_i * P_i; the host then dispatches a tree
// reduction (one kernel call per level, halving the active count)
// to sum all t_i in place. Six 64-bit limbs throughout; the modulus
// is bound as a device buffer and read at runtime.
//
// Coordinate convention: Jacobian (X, Y, Z) with X, Y, Z each
// stored as 6 uint64 limbs in little-endian. Affine point at
// (X / Z^2, Y / Z^3). Z = 0 represents the point at infinity.
// All field elements live in Montgomery form with R = 2^384.
//
// Buffer layout (host-fixed, must be preserved by candidate):
//
//   montgomery_msm_pair:
//     buffer 0: device const ulong *scalars      (n_pairs * 4 ulongs,
//                                                  little-endian)
//     buffer 1: device const ulong *points_in    (n_pairs * 18 ulongs,
//                                                  Jacobian Montgomery)
//     buffer 2: device       ulong *scratch      (n_pairs * 18 ulongs,
//                                                  per-pair products)
//     buffer 3: device const ulong *q            (6 ulongs, base field
//                                                  modulus, little-endian)
//     buffer 4: constant ulong &q_inv_neg        (-q^-1 mod 2^64,
//                                                  the CIOS scalar)
//     buffer 5: constant uint  &n_pairs
//
//   montgomery_msm_reduce:
//     buffer 0: device       ulong *scratch      (Jacobian Montgomery
//                                                  buffer, modified in-place)
//     buffer 1: device const ulong *q            (6 ulongs)
//     buffer 2: constant ulong &q_inv_neg
//     buffer 3: constant uint  &half_count       (current tree half-step:
//                                                  thread t reads
//                                                  scratch[t] += scratch[t + half_count])
//
// Dispatch (host-fixed):
//   montgomery_msm_pair:   grid = (n_pairs,) rounded up to TG width
//   montgomery_msm_reduce: grid = (half,) rounded up; one dispatch per
//                          level with half = n_pairs/2, n_pairs/4, ..., 1
//   threadsPerThreadgroup = min(grid_w, 64)
//
// Correctness model: bit-exact agreement after both sides normalize the
// final Jacobian point to affine in Montgomery form. Non-canonical limbs
// (>= q) are mismatches even if the residue class matches.
//
// Held-out twist: at the BN254 G1 size the runtime modulus q and the
// CIOS scalar q_inv_neg both change. A candidate that hardcodes the
// in-distribution modulus / Montgomery constants / arithmetic shape
// silently produces wrong output on the held-out probe.

#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;    // 3 * N_LIMBS
constexpr constant uint SCALAR_BITS = 256u;   // fixed window; r < 2^256 for both curves

constant ulong LIMB_MASK_LO32 = 0x00000000FFFFFFFFul;

// ------------------------------------------------------------------
// 128-bit multiplication via 32-bit decomposition. ulong2.x = lo,
// ulong2.y = hi.
// ------------------------------------------------------------------
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

// (lo, hi) = a * b + t + c. Fits in 128 bits for any 64-bit inputs
// because (2^64-1)^2 + (2^64-1) + (2^64-1) = 2^128 - 1.
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

// ------------------------------------------------------------------
// Multi-precision helpers (operate on N_LIMBS=6 limb thread arrays).
// ------------------------------------------------------------------

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

// c = (a + b) mod q. Inputs and output canonical (each < q).
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
    // Provisionally compute sum - q; if no borrow then sum >= q so we
    // use the subtracted form; otherwise we keep sum unchanged.
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

// c = (a - b) mod q.
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
        // Underflow: add q.
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

// CIOS Montgomery multiplication: out = a * b * R^-1 mod q with
// R = 2^(64 * N_LIMBS) = 2^384. ``q_inv_neg`` is (-q^-1) mod 2^64.
//
// Standard Coarsely-Integrated Operand Scanning: each outer iteration
// (i) accumulates a partial product a * b[i] into t, then chooses m
// such that t + m * q is divisible by 2^64 and shifts right by one
// limb. After N iterations, t holds (a * b * R^-1) reduced to < 2*q,
// and one conditional subtraction of q gives the canonical result.
inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     device const ulong *q, ulong q_inv_neg)
{
    // (N_LIMBS + 2) limbs of scratch: t[N_LIMBS] absorbs one carry
    // from the inner FMA chain; t[N_LIMBS+1] absorbs the *very* rare
    // case where (prod.hi + c1 + c2) plus the top-limb carry crosses
    // 2^64 (cumulative invariant <= a small constant across the
    // N outer iterations).
    ulong t[N_LIMBS + 2];
    for (uint i = 0u; i < N_LIMBS + 2u; ++i) t[i] = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        // Phase 1: t += a * b[i]
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

        // Phase 2: m = (t[0] * q_inv_neg) mod 2^64
        ulong m = t[0] * q_inv_neg;

        // Phase 3: t += m * q (makes t[0] divisible by 2^64; we drop
        // it on the shift).
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

        // Shift t right by one limb (drop the now-zero t[0]).
        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1];
        }
        t[N_LIMBS + 1] = 0ul;
    }

    // Final canonicalisation. After CIOS, t in [0, 2q). If t >= q,
    // subtract q. ``t[N_LIMBS] != 0`` is sufficient on its own
    // (because t < 2q < 2^(64*N+1) so the (N+1)-th limb is 0 or 1).
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

// ------------------------------------------------------------------
// Jacobian point ops on a short-Weierstrass curve with a = 0
// (y^2 = x^3 + b). The a=0 doubling formula does not reference b,
// and the addition formula does not reference a or b either.
// ------------------------------------------------------------------

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

// (out_X, out_Y, out_Z) = 2 * (X, Y, Z). Curve coefficient a == 0.
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

    mont_mul(A, X, X, q, q_inv_neg);              // A = X^2
    mont_mul(B, Y, Y, q, q_inv_neg);              // B = Y^2
    mont_mul(C, B, B, q, q_inv_neg);              // C = B^2

    mod_add(tmp, X, B, q);                         // X + B
    mont_mul(D, tmp, tmp, q, q_inv_neg);           // (X+B)^2
    mod_sub(D, D, A, q);
    mod_sub(D, D, C, q);
    mod_add(D, D, D, q);                           // D = 2*((X+B)^2 - A - C)

    mod_add(E, A, A, q);
    mod_add(E, E, A, q);                           // E = 3A

    mont_mul(F, E, E, q, q_inv_neg);               // F = E^2

    mod_add(tmp, D, D, q);
    mod_sub(oX, F, tmp, q);                        // X3 = F - 2D

    mod_sub(tmp, D, oX, q);                        // D - X3
    mont_mul(tmp, E, tmp, q, q_inv_neg);           // E*(D - X3)
    mod_add(tmp2, C, C, q);
    mod_add(tmp2, tmp2, tmp2, q);
    mod_add(tmp2, tmp2, tmp2, q);                  // 8C
    mod_sub(oY, tmp, tmp2, q);                     // Y3

    mont_mul(tmp, Y, Z, q, q_inv_neg);
    mod_add(oZ, tmp, tmp, q);                      // Z3 = 2*Y*Z
}

// (out_X, out_Y, out_Z) = (X1, Y1, Z1) + (X2, Y2, Z2).
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
            zero_point(oX, oY, oZ);     // P + (-P) = O
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
    mod_sub(oX, oX, tmp, q);                       // X3 = R^2 - HHH - 2V

    mod_sub(tmp, V, oX, q);
    mont_mul(tmp, R, tmp, q, q_inv_neg);
    mont_mul(tmp2, S1, HHH, q, q_inv_neg);
    mod_sub(oY, tmp, tmp2, q);                     // Y3 = R*(V - X3) - S1*HHH

    mont_mul(tmp, Z1, Z2, q, q_inv_neg);
    mont_mul(oZ, tmp, H, q, q_inv_neg);            // Z3 = Z1*Z2*H
}

// ------------------------------------------------------------------
// Kernel A: per-pair scalar multiplication, MSB-to-LSB double-and-add.
// One thread owns ONE pair end-to-end.
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

    // Load scalar (4 ulongs, little-endian -- bit b lives in word
    // b >> 6 at position b & 63).
    ulong s0 = scalars[idx * 4u + 0u];
    ulong s1 = scalars[idx * 4u + 1u];
    ulong s2 = scalars[idx * 4u + 2u];
    ulong s3 = scalars[idx * 4u + 3u];

    // Load base point P.
    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, points_in + idx * POINT_LIMBS);

    // Accumulator A = O (point at infinity).
    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    // Double-and-add, MSB -> LSB. The scan walks the fixed
    // SCALAR_BITS-bit window; leading zero bits land on the
    // infinity-Z early return in jac_double_pt.
    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
    for (int bit = (int)SCALAR_BITS - 1; bit >= 0; --bit) {
        jac_double_pt(TX, TY, TZ, AX, AY, AZ, q, q_inv_neg);
        copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);

        uint word = (uint)bit >> 6u;
        uint b = (uint)bit & 63u;
        ulong w = (word == 0u) ? s0 :
                  (word == 1u) ? s1 :
                  (word == 2u) ? s2 : s3;
        if (((w >> b) & 1ul) != 0ul) {
            jac_add_pt(TX, TY, TZ,
                       AX, AY, AZ,
                       PX, PY, PZ,
                       q, q_inv_neg);
            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
        }
    }

    store_point(scratch + idx * POINT_LIMBS, AX, AY, AZ);
}

// ------------------------------------------------------------------
// Kernel B: one level of pairwise tree reduction. Thread t writes
//   scratch[t] := scratch[t] + scratch[t + half]
// in place. After log2(n_pairs) dispatches with half =
// n_pairs/2, n_pairs/4, ..., 1, slot 0 holds the full MSM.
// ------------------------------------------------------------------
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
```

Incumbent result:
           bls_N4K: correct, 95.53 ms, 0.2 Gmodmul/s (int64) (0.4% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 359.56 ms, 0.2 Gmodmul/s (int64) (0.4% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 1356.45 ms, 0.2 Gmodmul/s (int64) (0.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0039

## History

- iter  0: compile=OK | correct=True | score=0.0039467370058945865
- iter  1: compile=OK | correct=True | score=0.00275860845363212
- iter  2: compile=OK | correct=True | score=0.0014136548427909807

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
