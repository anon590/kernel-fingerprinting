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
constexpr constant uint N32 = 12u;
constexpr constant uint POINT_LIMBS = 18u;

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
    uint d[N32];
    ulong borrow = 0ul;
    for (uint i = 0u; i < N32; ++i) {
        ulong t = (ulong)s[i] - (ulong)q32[i] - borrow;
        d[i] = (uint)t;
        borrow = (t >> 32) & 1ul;
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

inline void fe_mul(thread uint *out, thread const uint *a, thread const uint *b,
                   thread const uint *q32, uint mu32)
{
    ulong t[N32 + 1];
    for (uint i = 0u; i < N32 + 1u; ++i) t[i] = 0ul;

    for (uint i = 0u; i < N32; ++i) {
        ulong C = 0ul;
        ulong bi = (ulong)b[i];
        for (uint j = 0u; j < N32; ++j) {
            ulong p = (ulong)a[j] * bi + (t[j] & 0xFFFFFFFFul) + C;
            t[j] = p & 0xFFFFFFFFul;
            C = p >> 32;
        }
        ulong s = t[N32] + C;
        t[N32] = s;

        uint m = (uint)t[0] * mu32;
        ulong mm = (ulong)m;

        C = 0ul;
        for (uint j = 0u; j < N32; ++j) {
            ulong p = mm * (ulong)q32[j] + (t[j] & 0xFFFFFFFFul) + C;
            t[j] = p & 0xFFFFFFFFul;
            C = p >> 32;
        }
        s = t[N32] + C;
        for (uint j = 0u; j < N32; ++j) {
            t[j] = t[j + 1];
        }
        t[N32 - 1] = s & 0xFFFFFFFFul;
        t[N32] = s >> 32;
    }

    uint r[N32];
    for (uint i = 0u; i < N32; ++i) r[i] = (uint)t[i];

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

    // Find top bit set in scalar (skip leading zeros).
    int top = -1;
    for (int w = 3; w >= 0; --w) {
        if (s[w] != 0ul) {
            // Find MSB position in s[w]
            ulong v = s[w];
            int b = 63;
            while (b >= 0 && ((v >> (uint)b) & 1ul) == 0ul) --b;
            top = w * 64 + b;
            break;
        }
    }

    uint AX[N32], AY[N32], AZ[N32];
    uint TX[N32], TY[N32], TZ[N32];

    if (top < 0) {
        zero_pt(AX, AY, AZ);
    } else {
        // Initialize with P at the top bit (which is 1).
        fe_copy(AX, PX); fe_copy(AY, PY); fe_copy(AZ, PZ);

        // Binary double-and-add from bit (top-1) down to 0.
        for (int b = top - 1; b >= 0; --b) {
            jac_dbl(TX, TY, TZ, AX, AY, AZ, q32, mu32);
            fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);

            uint word = (uint)b >> 6u;
            uint shift = (uint)b & 63u;
            uint bit = (uint)((s[word] >> shift) & 1ul);
            if (bit != 0u) {
                jac_add(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q32, mu32);
                fe_copy(AX, TX); fe_copy(AY, TY); fe_copy(AZ, TZ);
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

Result of previous attempt:
           bls_N4K: correct, 62.54 ms, 0.3 Gmodmul/s (int64) (0.6% of 53 Gops/s (int64 mul, est))
          bls_N16K: correct, 235.35 ms, 0.3 Gmodmul/s (int64) (0.6% of 53 Gops/s (int64 mul, est))
          bls_N64K: correct, 910.82 ms, 0.3 Gmodmul/s (int64) (0.6% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0060

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
- iter  6: compile=OK | correct=True | score=0.0059781477821268855

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
