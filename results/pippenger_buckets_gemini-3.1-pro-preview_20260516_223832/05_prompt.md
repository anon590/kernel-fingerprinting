## Task: pippenger_buckets

Pippenger bucket-scatter on a short-Weierstrass elliptic curve. Given ``n_pairs`` 256-bit scalars and ``n_pairs`` Jacobian Montgomery points on BLS12-381 G1, compute the ``num_windows * (2^w - 1)`` bucket sums of Pippenger's MSM. For each pair index ``i`` and window index ``k in [0, num_windows)``, extract the ``w``-bit window value ``b = (s_i >> (k*w)) & ((1 << w) - 1)``. If ``b == 0`` the pair contributes nothing to window ``k``; otherwise add ``P_i`` (Jacobian Montgomery) to ``buckets[k][b-1]``. Buckets start as the point at infinity (all-zero Jacobian).

Field representation: six-limb Montgomery form (``R = 2^384``); the base-field modulus ``q`` (6 ulongs, little-endian) and the CIOS scalar ``q_inv_neg = -q^-1 mod 2^64`` are bound as device / constant buffers and must be read at runtime.

Window decomposition (host-bound runtime parameters):
  * ``window_bits`` = 16
  * ``num_windows`` = 4
The kernel processes the bottom ``num_windows * window_bits = 64`` bits of each scalar. Buckets are addressed in [1, 2^w); index ``b = 0`` is elided. The output buffer's slot ``[k][b - 1]`` holds the sum for window ``k`` and bucket value ``b``.

Coordinate convention: 6-limb Jacobian ``(X, Y, Z)`` in Montgomery form, little-endian limbs, 18 ulongs per point. ``Z == 0`` represents the point at infinity (the initial state of every bucket).

Scalars: 4-ulong little-endian limbs (256-bit).

Bit-exact correctness: the order in which a bucket's contributing points are summed is implementation-defined, so the Jacobian representation of each bucket may vary. The host normalizes every GPU bucket ``(X, Y, Z)`` to affine Montgomery ``(X / Z^2, Y / Z^3) * R mod q`` via one batched modular inversion and compares ``(X_aff_mont, Y_aff_mont)`` limb-for-limb against the CPU reference. A non-canonical limb (>= q) on the GPU side counts as a mismatch even if the residue class matches.

The kernel must read ``q``, ``q_inv_neg``, ``n_pairs``, ``num_windows`` and ``window_bits`` at runtime. Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved and the final bucket buffer is in Jacobian Montgomery form ready for host-side affine normalization.

## Required kernel signature(s)

```
kernel void pippenger_bucket_scatter(
    device const ulong *scalars     [[buffer(0)]],
    device const ulong *points_in   [[buffer(1)]],
    device       ulong *buckets     [[buffer(2)]],
    device const ulong *q           [[buffer(3)]],
    constant ulong     &q_inv_neg   [[buffer(4)]],
    constant uint      &n_pairs     [[buffer(5)]],
    constant uint      &num_windows [[buffer(6)]],
    constant uint      &window_bits [[buffer(7)]],
    uint idx [[thread_position_in_grid]]);

Buffer sizes (host-allocated):
  * scalars:     n_pairs * 4 ulongs
  * points_in:   n_pairs * 18 ulongs (Jacobian Montgomery)
  * buckets:     num_windows * (2^window_bits - 1) * 18 ulongs (zeroed before each dispatch)
  * q:           6 ulongs

Dispatch (host-fixed by the seed): one thread per (window, bucket). Total grid width is ``num_windows * ((1 << window_bits) - 1)`` rounded up to the threadgroup width. Thread ``idx`` decodes to ``(window_idx, bucket_value - 1) = (idx / num_buckets, idx % num_buckets)`` where ``num_buckets = (1 << window_bits) - 1``; guard against ``idx >= num_windows * num_buckets``. The seed uses ``threadsPerThreadgroup = (min(grid_w, 64), 1, 1)``. Alternative thread / threadgroup layouts are valid as long as the external buffer layout is preserved and every output bucket slot is populated with the correct Jacobian Montgomery sum on completion.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong2 umul128(ulong a, ulong b) {
    ulong lo = a * b;
    ulong a0 = (uint)a, a1 = a >> 32;
    ulong b0 = (uint)b, b1 = b >> 32;
    ulong p01 = a0 * b1;
    ulong p10 = a1 * b0;
    ulong mid = (p01 & 0xFFFFFFFFul) + (p10 & 0xFFFFFFFFul) + ((a0 * b0) >> 32);
    ulong hi = (a1 * b1) + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong2 fma_add128(ulong a, ulong b, ulong t, ulong c) {
    ulong2 prod = umul128(a, b);
    ulong lo1 = prod.x + t;
    ulong cy1 = (lo1 < prod.x) ? 1ul : 0ul;
    ulong lo2 = lo1 + c;
    ulong cy2 = (lo2 < lo1) ? 1ul : 0ul;
    return ulong2(lo2, prod.y + cy1 + cy2);
}

inline void copy_n(thread ulong *dst, thread const ulong *src) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[i] = src[i];
}

inline bool is_zero_n(thread const ulong *a) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        if (a[i] != 0ul) return false;
    }
    return true;
}

inline bool eq_n(thread const ulong *a, thread const ulong *b) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline void mod_add(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
{
    ulong sum[6];
    ulong carry = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong s = a[i] + carry;
        ulong cy1 = (s < a[i]) ? 1ul : 0ul;
        ulong t = s + b[i];
        ulong cy2 = (t < s) ? 1ul : 0ul;
        sum[i] = t;
        carry = cy1 + cy2;
    }
    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = sum[i] - q[i];
        ulong b1 = (tv > sum[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (carry != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        c[i] = use_diff ? diff[i] : sum[i];
    }
}

inline void mod_sub(thread ulong *c,
                    thread const ulong *a, thread const ulong *b,
                    thread const ulong *q)
{
    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    if (borrow != 0ul) {
        ulong carry = 0ul;
        #pragma unroll
        for (uint i = 0u; i < 6u; ++i) {
            ulong s = diff[i] + carry;
            ulong cy1 = (s < diff[i]) ? 1ul : 0ul;
            ulong t = s + q[i];
            ulong cy2 = (t < s) ? 1ul : 0ul;
            c[i] = t;
            carry = cy1 + cy2;
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < 6u; ++i) c[i] = diff[i];
    }
}

inline void mont_mul(thread ulong *out,
                     thread const ulong *a, thread const ulong *b,
                     thread const ulong *q, ulong q_inv_neg)
{
    ulong t[8];
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) t[i] = 0ul;

    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            ulong2 r = fma_add128(a[j], b[i], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[6] + C;
            ulong cy = (s < t[6]) ? 1ul : 0ul;
            t[6] = s;
            t[7] += cy;
        }
        ulong m = t[0] * q_inv_neg;
        C = 0ul;
        #pragma unroll
        for (uint j = 0u; j < 6u; ++j) {
            ulong2 r = fma_add128(m, q[j], t[j], C);
            t[j] = r.x;
            C = r.y;
        }
        {
            ulong s = t[6] + C;
            ulong cy = (s < t[6]) ? 1ul : 0ul;
            t[6] = s;
            t[7] += cy;
        }
        #pragma unroll
        for (uint j = 0u; j < 7u; ++j) {
            t[j] = t[j + 1];
        }
        t[7] = 0ul;
    }

    ulong diff[6];
    ulong borrow = 0ul;
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d = tv - borrow;
        ulong b2 = (d > tv) ? 1ul : 0ul;
        diff[i] = d;
        borrow = b1 + b2;
    }
    bool use_diff = (t[6] != 0ul) || (borrow == 0ul);
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        out[i] = use_diff ? diff[i] : t[i];
    }
}

inline void load_point(thread ulong *X, thread ulong *Y, thread ulong *Z,
                       device const ulong *src)
{
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) X[i] = src[i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) Y[i] = src[6u + i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) Z[i] = src[12u + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X, thread const ulong *Y, thread const ulong *Z)
{
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[i] = X[i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[6u + i] = Y[i];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) dst[12u + i] = Z[i];
}

inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) {
        X[i] = 0ul;
        Y[i] = 0ul;
        Z[i] = 0ul;
    }
}

inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X, thread const ulong *Y, thread const ulong *Z,
                          thread const ulong *q, ulong q_inv_neg)
{
    if (is_zero_n(Z) || is_zero_n(Y)) {
        zero_point(oX, oY, oZ);
        return;
    }
    
    ulong T1[6], T2[6], T3[6];
    mont_mul(T1, X, X, q, q_inv_neg);
    mont_mul(T2, Y, Y, q, q_inv_neg);
    mont_mul(T3, T2, T2, q, q_inv_neg);
    
    ulong D[6];
    mod_add(D, X, T2, q);
    mont_mul(D, D, D, q, q_inv_neg);
    mod_sub(D, D, T1, q);
    mod_sub(D, D, T3, q);
    mod_add(D, D, D, q);
    
    ulong E[6];
    mod_add(E, T1, T1, q);
    mod_add(E, E, T1, q);
    
    ulong F[6];
    mont_mul(F, E, E, q, q_inv_neg);
    
    mod_add(T1, D, D, q);
    mod_sub(oX, F, T1, q);
    
    mod_sub(T2, D, oX, q);
    mont_mul(oY, E, T2, q, q_inv_neg);
    
    mod_add(T3, T3, T3, q);
    mod_add(T3, T3, T3, q);
    mod_add(T3, T3, T3, q);
    mod_sub(oY, oY, T3, q);
    
    mont_mul(oZ, Y, Z, q, q_inv_neg);
    mod_add(oZ, oZ, oZ, q);
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
    
    ulong T1[6], T2[6], T3[6], T4[6];
    
    mont_mul(T1, Z1, Z1, q, q_inv_neg);
    mont_mul(T2, Z2, Z2, q, q_inv_neg);
    
    ulong U1[6];
    mont_mul(U1, X1, T2, q, q_inv_neg);
    
    ulong U2[6];
    mont_mul(U2, X2, T1, q, q_inv_neg);
    
    ulong S1[6];
    mont_mul(S1, Y1, Z2, q, q_inv_neg);
    mont_mul(S1, S1, T2, q, q_inv_neg);
    
    ulong S2[6];
    mont_mul(S2, Y2, Z1, q, q_inv_neg);
    mont_mul(S2, S2, T1, q, q_inv_neg);
    
    if (eq_n(U1, U2)) {
        if (eq_n(S1, S2)) {
            jac_double_pt(oX, oY, oZ, X1, Y1, Z1, q, q_inv_neg);
            return;
        } else {
            zero_point(oX, oY, oZ);
            return;
        }
    }
    
    mod_sub(T1, U2, U1, q);
    mod_sub(T2, S2, S1, q);
    
    mont_mul(T3, T1, T1, q, q_inv_neg);
    mont_mul(T4, T1, T3, q, q_inv_neg);
    
    mont_mul(U2, U1, T3, q, q_inv_neg);
    
    mont_mul(oX, T2, T2, q, q_inv_neg);
    mod_sub(oX, oX, T4, q);
    mod_sub(oX, oX, U2, q);
    mod_sub(oX, oX, U2, q);
    
    mod_sub(U1, U2, oX, q);
    mont_mul(oY, T2, U1, q, q_inv_neg);
    mont_mul(S2, S1, T4, q, q_inv_neg);
    mod_sub(oY, oY, S2, q);
    
    mont_mul(oZ, Z1, Z2, q, q_inv_neg);
    mont_mul(oZ, oZ, T1, q, q_inv_neg);
}

kernel void pippenger_bucket_scatter(
    device const ulong *scalars     [[buffer(0)]],
    device const ulong *points_in   [[buffer(1)]],
    device       ulong *buckets     [[buffer(2)]],
    device const ulong *q           [[buffer(3)]],
    constant ulong     &q_inv_neg   [[buffer(4)]],
    constant uint      &n_pairs     [[buffer(5)]],
    constant uint      &num_windows [[buffer(6)]],
    constant uint      &window_bits [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    uint num_buckets = (1u << window_bits) - 1u;
    if (idx >= num_windows * num_buckets) return;

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx % num_buckets;
    
    uint shift = (window_idx * window_bits) & 63u;
    ulong mask = (1ul << window_bits) - 1ul;

    ulong AX[6], AY[6], AZ[6];
    zero_point(AX, AY, AZ);

    ulong q_reg[6];
    #pragma unroll
    for (uint i = 0u; i < 6u; ++i) q_reg[i] = q[i];

    for (uint i = 0u; i < n_pairs; ++i) {
        ulong S = scalars[i * 4u];
        uint w_val = (uint)((S >> shift) & mask);
        
        if (w_val != 0u && (w_val - 1u) == bucket_minus_1) {
            ulong PX[6], PY[6], PZ[6];
            load_point(PX, PY, PZ, points_in + i * 18u);
            
            ulong TX[6], TY[6], TZ[6];
            jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
            copy_n(AX, TX);
            copy_n(AY, TY);
            copy_n(AZ, TZ);
        }
    }

    store_point(buckets + idx * 18u, AX, AY, AZ);
}
```

Result of previous attempt:
       uniform_N4K: correct, 685.71 ms, 0.0 GB/s (0.0% of 200 GB/s)
      uniform_N16K: correct, 2759.01 ms, 0.0 GB/s (0.0% of 200 GB/s)
      uniform_N64K: correct, 10934.41 ms, 0.0 GB/s (0.0% of 200 GB/s)
  score (gmean of fraction): 0.0000

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;

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
                    thread const ulong *q)
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
                    thread const ulong *q)
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
                     thread const ulong *q, ulong q_inv_neg)
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

kernel void pippenger_bucket_scatter(
    device const ulong *scalars     [[buffer(0)]],
    device const ulong *points_in   [[buffer(1)]],
    device       ulong *buckets     [[buffer(2)]],
    device const ulong *q           [[buffer(3)]],
    constant ulong     &q_inv_neg   [[buffer(4)]],
    constant uint      &n_pairs     [[buffer(5)]],
    constant uint      &num_windows [[buffer(6)]],
    constant uint      &window_bits [[buffer(7)]],
    uint idx [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    uint w = window_bits;
    uint num_buckets = (1u << w) - 1u;
    uint total = num_windows * num_buckets;
    ulong mask = (1ul << w) - 1ul;

    ulong q_reg[N_LIMBS];
    for (uint i = 0u; i < N_LIMBS; ++i) {
        q_reg[i] = q[i];
    }

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx - window_idx * num_buckets;
    
    uint simd_start_idx = idx - lane_id;
    uint simd_end_idx = simd_start_idx + 31u;

    bool same_window = (simd_start_idx / num_buckets) == (simd_end_idx / num_buckets);
    uint common_window = simd_start_idx / num_buckets;
    uint shift = (common_window * w) & 63u;
    uint bucket_start = simd_start_idx % num_buckets;

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    if (same_window && common_window < num_windows) {
        for (uint chunk = 0; chunk < n_pairs; chunk += 128u) {
            uint p0 = chunk + lane_id;
            uint p1 = p0 + 32u;
            uint p2 = p0 + 64u;
            uint p3 = p0 + 96u;

            ulong S0 = (p0 < n_pairs) ? scalars[p0 * 4u] : 0ul;
            ulong S1 = (p1 < n_pairs) ? scalars[p1 * 4u] : 0ul;
            ulong S2 = (p2 < n_pairs) ? scalars[p2 * 4u] : 0ul;
            ulong S3 = (p3 < n_pairs) ? scalars[p3 * 4u] : 0ul;

            uint w0 = (uint)((S0 >> shift) & mask);
            uint b0 = w0 - 1u;
            bool m0 = (w0 != 0u) && (b0 >= bucket_start) && (b0 < bucket_start + 32u);

            uint w1 = (uint)((S1 >> shift) & mask);
            uint b1 = w1 - 1u;
            bool m1 = (w1 != 0u) && (b1 >= bucket_start) && (b1 < bucket_start + 32u);

            uint w2 = (uint)((S2 >> shift) & mask);
            uint b2 = w2 - 1u;
            bool m2 = (w2 != 0u) && (b2 >= bucket_start) && (b2 < bucket_start + 32u);

            uint w3 = (uint)((S3 >> shift) & mask);
            uint b3 = w3 - 1u;
            bool m3 = (w3 != 0u) && (b3 >= bucket_start) && (b3 < bucket_start + 32u);

            bool any_match = simd_any(m0 || m1 || m2 || m3);
            if (any_match) {
                if (simd_any(m0)) {
                    uint m0_u = m0 ? 1u : 0u;
                    for (uint i = 0u; i < 32u; ++i) {
                        if (chunk + i >= n_pairs) break;
                        if (simd_broadcast(m0_u, (ushort)i) != 0u) {
                            uint target = simd_broadcast(b0, (ushort)i);
                            if (bucket_minus_1 == target && idx < total) {
                                ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                                load_point(PX, PY, PZ, points_in + (chunk + i) * POINT_LIMBS);
                                ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                            }
                        }
                    }
                }
                if (simd_any(m1)) {
                    uint m1_u = m1 ? 1u : 0u;
                    for (uint i = 0u; i < 32u; ++i) {
                        if (chunk + 32u + i >= n_pairs) break;
                        if (simd_broadcast(m1_u, (ushort)i) != 0u) {
                            uint target = simd_broadcast(b1, (ushort)i);
                            if (bucket_minus_1 == target && idx < total) {
                                ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                                load_point(PX, PY, PZ, points_in + (chunk + 32u + i) * POINT_LIMBS);
                                ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                            }
                        }
                    }
                }
                if (simd_any(m2)) {
                    uint m2_u = m2 ? 1u : 0u;
                    for (uint i = 0u; i < 32u; ++i) {
                        if (chunk + 64u + i >= n_pairs) break;
                        if (simd_broadcast(m2_u, (ushort)i) != 0u) {
                            uint target = simd_broadcast(b2, (ushort)i);
                            if (bucket_minus_1 == target && idx < total) {
                                ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                                load_point(PX, PY, PZ, points_in + (chunk + 64u + i) * POINT_LIMBS);
                                ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                            }
                        }
                    }
                }
                if (simd_any(m3)) {
                    uint m3_u = m3 ? 1u : 0u;
                    for (uint i = 0u; i < 32u; ++i) {
                        if (chunk + 96u + i >= n_pairs) break;
                        if (simd_broadcast(m3_u, (ushort)i) != 0u) {
                            uint target = simd_broadcast(b3, (ushort)i);
                            if (bucket_minus_1 == target && idx < total) {
                                ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                                load_point(PX, PY, PZ, points_in + (chunk + 96u + i) * POINT_LIMBS);
                                ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                                jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                                copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (uint chunk = 0; chunk < n_pairs; chunk += 32u) {
            uint pair_idx = chunk + lane_id;
            ulong S = (pair_idx < n_pairs) ? scalars[pair_idx * 4u] : 0ul;
            
            bool m = false;
            uint t_arr[8];
            for (uint k = 0u; k < num_windows && k < 8u; ++k) {
                uint wk = (uint)((S >> (k * w)) & mask);
                uint t = wk ? k * num_buckets + wk - 1u : 0xFFFFFFFFu;
                if (t >= simd_start_idx && t <= simd_end_idx) {
                    m = true;
                }
                t_arr[k] = t;
            }
            
            if (simd_any(m)) {
                uint m_u = m ? 1u : 0u;
                for (uint i = 0u; i < 32u; ++i) {
                    if (chunk + i >= n_pairs) break;
                    if (simd_broadcast(m_u, (ushort)i) != 0u) {
                        bool my_match = false;
                        for (uint k = 0u; k < num_windows && k < 8u; ++k) {
                            uint b_t = simd_broadcast(t_arr[k], (ushort)i);
                            if (idx == b_t) my_match = true;
                        }
                        
                        if (my_match && idx < total) {
                            ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
                            load_point(PX, PY, PZ, points_in + (chunk + i) * POINT_LIMBS);
                            ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
                            jac_add_pt(TX, TY, TZ, AX, AY, AZ, PX, PY, PZ, q_reg, q_inv_neg);
                            copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
                        }
                    }
                }
            }
        }
    }

    if (idx < total) {
        uint bucket_idx = window_idx * num_buckets + bucket_minus_1;
        store_point(buckets + bucket_idx * POINT_LIMBS, AX, AY, AZ);
    }
}
```

Incumbent result:
       uniform_N4K: correct, 5.50 ms, 1.0 GB/s (0.5% of 200 GB/s)
      uniform_N16K: correct, 37.93 ms, 0.6 GB/s (0.3% of 200 GB/s)
      uniform_N64K: correct, 388.39 ms, 0.2 GB/s (0.1% of 200 GB/s)
  score (gmean of fraction): 0.0025

## History

- iter  0: compile=OK | correct=True | score=0.0004153332584870956
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.0017300673935013318
- iter  3: compile=OK | correct=True | score=0.0025138921913871147
- iter  4: compile=OK | correct=True | score=3.9630114449598926e-05

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
