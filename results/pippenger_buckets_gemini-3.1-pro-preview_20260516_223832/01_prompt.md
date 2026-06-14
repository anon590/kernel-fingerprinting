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

## Baseline: naive seed kernel

```metal
// Naive seed for Pippenger's bucket-scatter (one thread per (window,
// bucket); per-thread sequential scan over all pairs).
//
// For each (window k, bucket value b in [1, 2^w)), a single thread
// scans all N pairs in index order. For every pair whose window value
// equals b, the thread Jacobian-adds P_i into a thread-private
// accumulator. After the scan completes the accumulator is written
// to buckets[k][b - 1].
//
// Coordinate convention: Jacobian (X, Y, Z) with X, Y, Z each stored
// as 6 uint64 limbs in little-endian (six-limb Montgomery form,
// R = 2^384). Z = 0 represents the point at infinity; every bucket
// starts at (0, 0, 0). The base field modulus q and the CIOS scalar
// q_inv_neg = -q^-1 mod 2^64 are bound as device / constant buffers
// and read at runtime.
//
// Window decomposition (host-bound runtime parameters):
//   num_windows  = 4   (number of consecutive w-bit windows scanned)
//   window_bits  = 16
// The kernel processes the bottom num_windows * window_bits = 64 bits
// of each scalar; the upper bits of the scalar are not consulted.
// Bucket layout buckets[k][b-1] for b in [1, 1 << window_bits): a
// total of num_windows * ((1 << w) - 1) Jacobian points in the
// bucket buffer.
//
// Buffer layout (host-fixed, preserved by candidate):
//   buffer 0: device const ulong *scalars     (n_pairs * 4 ulongs,
//                                              little-endian; only the
//                                              bottom limb is read)
//   buffer 1: device const ulong *points_in   (n_pairs * 18 ulongs,
//                                              Jacobian Montgomery)
//   buffer 2: device       ulong *buckets     (num_windows *
//                                              ((1 << w) - 1) * 18
//                                              ulongs, Jacobian
//                                              Montgomery; host zeros
//                                              between dispatches)
//   buffer 3: device const ulong *q           (6 ulongs, base field
//                                              modulus, little-endian)
//   buffer 4: constant ulong &q_inv_neg       (-q^-1 mod 2^64; CIOS
//                                              scalar)
//   buffer 5: constant uint  &n_pairs
//   buffer 6: constant uint  &num_windows
//   buffer 7: constant uint  &window_bits
//
// Dispatch (host-fixed):
//   grid_w = (num_windows * ((1 << window_bits) - 1)) rounded up to
//            TG width.
//   threadsPerThreadgroup = (min(grid_w, 64), 1, 1).
//   Thread idx decodes (window_idx, bucket_minus_1) =
//       (idx / num_buckets, idx % num_buckets); the actual bucket
//   value scanned is bucket_minus_1 + 1 (b in [1, 2^w)). Guard
//   idx >= num_windows * num_buckets where num_buckets =
//   (1 << window_bits) - 1.
//
// Correctness: bit-exact in affine Montgomery form. The host
// normalizes every bucket Jacobian point (X, Y, Z) to affine
// (X / Z^2, Y / Z^3) * R mod q via one batched modular inversion
// and compares (X_aff_mont, Y_aff_mont) limb-for-limb against the
// CPU reference. The order in which a given bucket's contributing
// points are summed is implementation-defined; only the affine
// point must agree. Non-canonical limbs (>= q) on the GPU side
// count as mismatches.

#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;       // 3 * N_LIMBS

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

// CIOS Montgomery multiplication (matches the Z1 montgomery_msm seed
// bit-for-bit).
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

// ------------------------------------------------------------------
// Jacobian point ops on a short-Weierstrass curve with a = 0
// (y^2 = x^3 + b). The a=0 doubling formula does not reference b;
// the addition formula does not reference a or b either.
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
// Kernel: per-(window, bucket) sequential scan. Thread owns one
// (window_idx, bucket_value) pair; it walks pair indices 0..n_pairs-1
// and Jacobian-adds P_i whenever (s_i.bottom64 >> (k * w)) & mask ==
// bucket_value. The accumulator starts at Jacobian infinity (Z = 0).
// Each thread writes a distinct slot of the bucket buffer at the end.
// ------------------------------------------------------------------
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
    uint w = window_bits;
    uint num_buckets = (1u << w) - 1u;            // buckets are [1, 1<<w)
    uint total = num_windows * num_buckets;
    if (idx >= total) return;

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx - window_idx * num_buckets;
    uint bucket_val = bucket_minus_1 + 1u;
    uint bit_offset = window_idx * w;
    ulong mask = (1ul << w) - 1ul;

    // Thread-private Jacobian accumulator, starts at infinity.
    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);

    for (uint i = 0u; i < n_pairs; ++i) {
        ulong s0 = scalars[i * 4u + 0u];
        uint b = (uint)((s0 >> bit_offset) & mask);
        if (b != bucket_val) continue;

        ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
        load_point(PX, PY, PZ, points_in + i * POINT_LIMBS);

        ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
        jac_add_pt(TX, TY, TZ,
                   AX, AY, AZ,
                   PX, PY, PZ,
                   q, q_inv_neg);
        copy_n(AX, TX); copy_n(AY, TY); copy_n(AZ, TZ);
    }

    uint bucket_idx = window_idx * num_buckets + bucket_minus_1;
    store_point(buckets + bucket_idx * POINT_LIMBS, AX, AY, AZ);
}
```

Measured baseline (seed):
       uniform_N4K: correct, 63.35 ms, 0.1 GB/s (0.0% of 200 GB/s)
      uniform_N16K: correct, 251.73 ms, 0.1 GB/s (0.0% of 200 GB/s)
      uniform_N64K: correct, 1126.87 ms, 0.1 GB/s (0.0% of 200 GB/s)
  score (gmean of fraction): 0.0004

## Your task

Write an improved Metal kernel that produces bit-exact-correct
results AND runs faster than the seed across all problem sizes. The
fitness score is the geometric mean of `achieved / ceiling` across
sizes; score 0 if any size fails bit-exact correctness.

Output ONE fenced ```metal``` code block containing the kernel(s).
Preserve the kernel name(s) and buffer indices exactly.
