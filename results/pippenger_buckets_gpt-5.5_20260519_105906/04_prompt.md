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

constexpr constant uint N_LIMBS     = 6u;
constexpr constant uint POINT_LIMBS = 18u;

constexpr constant uint Z_BASE_LIMB = 12u;
constexpr constant uint Z5_LIMB     = 17u;
constexpr constant uint Z5_LO_WORD  = 34u;
constexpr constant uint Z5_HI_WORD  = 35u;

constant ulong LIMB_MASK_LO32 = 0x00000000FFFFFFFFul;
constant uint  LOCK_BIT32     = 0x80000000u;

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
// Multi-precision helpers.
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
                    thread const ulong *a,
                    thread const ulong *b,
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
        ulong d  = tv - borrow;
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
                    thread const ulong *a,
                    thread const ulong *b,
                    device const ulong *q)
{
    ulong diff[N_LIMBS];
    ulong borrow = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = a[i] - b[i];
        ulong b1 = (tv > a[i]) ? 1ul : 0ul;
        ulong d  = tv - borrow;
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

// CIOS Montgomery multiplication.
inline void mont_mul(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *b,
                     device const ulong *q,
                     ulong q_inv_neg)
{
    ulong t[N_LIMBS + 2u];
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
            t[N_LIMBS + 1u] += cy;
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
            t[N_LIMBS + 1u] += cy;
        }

        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1u];
        }
        t[N_LIMBS + 1u] = 0ul;
    }

    ulong diff[N_LIMBS];
    ulong borrow = 0ul;

    for (uint i = 0u; i < N_LIMBS; ++i) {
        ulong tv = t[i] - q[i];
        ulong b1 = (tv > t[i]) ? 1ul : 0ul;
        ulong d  = tv - borrow;
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
// Point helpers.
// ------------------------------------------------------------------
inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = 0ul;
}

inline void load_point(thread ulong *X,
                       thread ulong *Y,
                       thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

// ------------------------------------------------------------------
// Jacobian point ops, short-Weierstrass a = 0.
// ------------------------------------------------------------------
inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X,
                          thread const ulong *Y,
                          thread const ulong *Z,
                          device const ulong *q,
                          ulong q_inv_neg)
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
                       thread const ulong *X1,
                       thread const ulong *Y1,
                       thread const ulong *Z1,
                       thread const ulong *X2,
                       thread const ulong *Y2,
                       thread const ulong *Z2,
                       device const ulong *q,
                       ulong q_inv_neg)
{
    if (is_zero_n(Z1)) {
        copy_n(oX, X2);
        copy_n(oY, Y2);
        copy_n(oZ, Z2);
        return;
    }

    if (is_zero_n(Z2)) {
        copy_n(oX, X1);
        copy_n(oY, Y1);
        copy_n(oZ, Z1);
        return;
    }

    ulong Z1Z1[N_LIMBS], Z2Z2[N_LIMBS];
    ulong U1[N_LIMBS], U2[N_LIMBS], S1[N_LIMBS], S2[N_LIMBS];
    ulong H[N_LIMBS], R[N_LIMBS];
    ulong HH[N_LIMBS], HHH[N_LIMBS], V[N_LIMBS];
    ulong tmp[N_LIMBS], tmp2[N_LIMBS];

    mont_mul(Z1Z1, Z1, Z1, q, q_inv_neg);
    mont_mul(Z2Z2, Z2, Z2, q, q_inv_neg);

    mont_mul(U1, X1, Z2Z2, q, q_inv_neg);
    mont_mul(U2, X2, Z1Z1, q, q_inv_neg);

    mont_mul(tmp, Y1, Z2, q, q_inv_neg);
    mont_mul(S1, tmp, Z2Z2, q, q_inv_neg);

    mont_mul(tmp, Y2, Z1, q, q_inv_neg);
    mont_mul(S2, tmp, Z1Z1, q, q_inv_neg);

    if (eq_n(U1, U2)) {
        if (eq_n(S1, S2)) {
            jac_double_pt(oX, oY, oZ, X1, Y1, Z1, q, q_inv_neg);
        } else {
            zero_point(oX, oY, oZ);
        }
        return;
    }

    mod_sub(H, U2, U1, q);
    mod_sub(R, S2, S1, q);

    mont_mul(HH,  H, HH /* dummy init overwritten below */, q, q_inv_neg);
    // The line above is intentionally overwritten by the correct square below
    // to keep all temporaries declared uniformly on older runtime compilers.
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
// Bucket lock helpers.
// Lock bit lives in bit 31 of the high uint of Z[5]. Since BLS12-381
// field elements are < 2^381, this bit is never set in canonical output.
// ------------------------------------------------------------------
inline uint acquire_bucket_lock(device atomic_uint *lockp) {
    for (;;) {
        uint cur = atomic_load_explicit(lockp, memory_order_relaxed);

        if ((cur & LOCK_BIT32) == 0u) {
            uint expected = cur;
            uint desired  = cur | LOCK_BIT32;

            if (atomic_compare_exchange_weak_explicit(lockp,
                                                      &expected,
                                                      desired,
                                                      memory_order_acquire,
                                                      memory_order_relaxed)) {
                return cur;
            }
        }
    }
}

inline void load_bucket_z_locked(thread ulong *Z,
                                 device ulong *bucket,
                                 device uint *words,
                                 uint old_z5_hi)
{
    for (uint i = 0u; i < 5u; ++i) {
        Z[i] = bucket[Z_BASE_LIMB + i];
    }

    uint z5_lo = words[Z5_LO_WORD];
    Z[5] = ((ulong)old_z5_hi << 32) | (ulong)z5_lo;
}

inline void load_bucket_xy_locked(thread ulong *X,
                                  thread ulong *Y,
                                  device ulong *bucket)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = bucket[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = bucket[N_LIMBS + i];
}

inline void store_bucket_release(device ulong *bucket,
                                 device uint *words,
                                 device atomic_uint *lockp,
                                 thread const ulong *X,
                                 thread const ulong *Y,
                                 thread const ulong *Z)
{
    for (uint i = 0u; i < N_LIMBS; ++i) {
        bucket[i] = X[i];
    }

    for (uint i = 0u; i < N_LIMBS; ++i) {
        bucket[N_LIMBS + i] = Y[i];
    }

    for (uint i = 0u; i < 5u; ++i) {
        bucket[Z_BASE_LIMB + i] = Z[i];
    }

    words[Z5_LO_WORD] = (uint)Z[5];

    uint z5_hi = (uint)(Z[5] >> 32) & (~LOCK_BIT32);
    atomic_store_explicit(lockp, z5_hi, memory_order_release);
}

inline uint scalar_window(device const ulong *scalars,
                          uint pair_i,
                          uint window_idx,
                          uint w)
{
    device const ulong *sp = scalars + pair_i * 4u;

    uint bit   = window_idx * w;
    uint limb  = bit >> 6;
    uint shift = bit & 63u;

    ulong v = 0ul;

    if (limb < 4u) {
        v = sp[limb] >> shift;

        if ((shift != 0u) && ((shift + w) > 64u) && ((limb + 1u) < 4u)) {
            v |= sp[limb + 1u] << (64u - shift);
        }
    }

    ulong mask = (w >= 64u) ? (~0ul) : ((1ul << w) - 1ul);
    return (uint)(v & mask);
}

// ------------------------------------------------------------------
// Lock-protected scatter. Each in-range launched thread acts as a
// pair/window work lane with stride equal to the number of bucket slots.
// Padding threads return. Empty buckets are populated by direct copy;
// only bucket collisions execute Jacobian addition.
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
    if (num_windows == 0u) return;

    uint w = window_bits;
    uint num_buckets = (1u << w) - 1u;
    uint bucket_slots = num_windows * num_buckets;

    if (bucket_slots == 0u) return;
    if (idx >= bucket_slots) return;

    uint total_jobs = n_pairs * num_windows;

    for (uint job = idx; job < total_jobs; job += bucket_slots) {
        uint pair_i     = job / num_windows;
        uint window_idx = job - pair_i * num_windows;

        uint b = scalar_window(scalars, pair_i, window_idx, w);
        if (b == 0u) continue;

        uint bucket_slot = window_idx * num_buckets + (b - 1u);

        ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
        load_point(PX, PY, PZ, points_in + pair_i * POINT_LIMBS);

        if (is_zero_n(PZ)) continue;

        device ulong *bucket = buckets + bucket_slot * POINT_LIMBS;
        device uint  *words  = reinterpret_cast<device uint *>(bucket);
        device atomic_uint *lockp =
            reinterpret_cast<device atomic_uint *>(words + Z5_HI_WORD);

        uint old_z5_hi = acquire_bucket_lock(lockp);

        ulong BZ[N_LIMBS];
        load_bucket_z_locked(BZ, bucket, words, old_z5_hi);

        if (is_zero_n(BZ)) {
            store_bucket_release(bucket, words, lockp, PX, PY, PZ);
            continue;
        }

        ulong BX[N_LIMBS], BY[N_LIMBS];
        load_bucket_xy_locked(BX, BY, bucket);

        ulong RX[N_LIMBS], RY[N_LIMBS], RZ[N_LIMBS];

        jac_add_pt(RX, RY, RZ,
                   BX, BY, BZ,
                   PX, PY, PZ,
                   q, q_inv_neg);

        store_bucket_release(bucket, words, lockp, RX, RY, RZ);
    }
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:363:55: error: use of undeclared identifier 'memory_order_acquire'; did you mean 'memory_order_relaxed'?
                                                      memory_order_acquire,
                                                      ^~~~~~~~~~~~~~~~~~~~
                                                      memory_order_relaxed
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_types:192:3: note: 'memory_order_relaxed' declared here
  memory_order_relaxed = __METAL_MEMORY_ORDER_RELAXED__,
  ^
program_source:414:41: error: use of undeclared identifier 'memory_order_release'; did you mean 'memory_order_relaxed'?
    atomic_store_explicit(lockp, z5_hi, memory_order_release);
                                        ^~~~~~~~~~~~~~~~~~~~
                                        memory_order_relaxed
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_types:192:3: note: 'memory_order_relaxed' declared here
  memory_order_relaxed = __METAL_MEMORY_ORDER_RELAXED__,
  ^
" UserInfo={NSLocalizedDescription=program_source:363:55: error: use of undeclared identifier 'memory_order_acquire'; did you mean 'memory_order_relaxed'?
                                                      memory_order_acquire,
                                                      ^~~~~~~~~~~~~~~~~~~~
                                                      memory_order_relaxed
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_types:192:3: note: 'memory_order_relaxed' declared here
  memory_order_relaxed = __METAL_MEMORY_ORDER_RELAXED__,
  ^
program_source:414:41: error: use of undeclared identifier 'memory_order_release'; did you mean 'memory_order_relaxed'?
    atomic_store_explicit(lockp, z5_hi, memory_order_release);
                                        ^~~~~~~~~~~~~~~~~~~~
                                        memory_order_relaxed
/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.883/include/metal/metal_types:192:3: note: 'memory_order_relaxed' declared here
  memory_order_relaxed = __METAL_MEMORY_ORDER_RELAXED__,
  ^
}

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS = 6u;
constexpr constant uint POINT_LIMBS = 18u;
constexpr constant uint SIMD_W = 32u;

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
// Multi-precision helpers.
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

// CIOS Montgomery multiplication.
inline void mont_mul(thread ulong *out,
                     thread const ulong *a,
                     thread const ulong *b,
                     device const ulong *q,
                     ulong q_inv_neg)
{
    ulong t[N_LIMBS + 2u];
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
            t[N_LIMBS + 1u] += cy;
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
            t[N_LIMBS + 1u] += cy;
        }

        for (uint j = 0u; j < N_LIMBS + 1u; ++j) {
            t[j] = t[j + 1u];
        }
        t[N_LIMBS + 1u] = 0ul;
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
// Point helpers.
// ------------------------------------------------------------------
inline void zero_point(thread ulong *X, thread ulong *Y, thread ulong *Z) {
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = 0ul;
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = 0ul;
}

inline void load_point(thread ulong *X,
                       thread ulong *Y,
                       thread ulong *Z,
                       device const ulong *src)
{
    for (uint i = 0u; i < N_LIMBS; ++i) X[i] = src[i];
    for (uint i = 0u; i < N_LIMBS; ++i) Y[i] = src[N_LIMBS + i];
    for (uint i = 0u; i < N_LIMBS; ++i) Z[i] = src[2u * N_LIMBS + i];
}

inline void store_point(device ulong *dst,
                        thread const ulong *X,
                        thread const ulong *Y,
                        thread const ulong *Z)
{
    for (uint i = 0u; i < N_LIMBS; ++i) dst[i] = X[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[N_LIMBS + i] = Y[i];
    for (uint i = 0u; i < N_LIMBS; ++i) dst[2u * N_LIMBS + i] = Z[i];
}

// ------------------------------------------------------------------
// Jacobian point ops, short-Weierstrass a = 0.
// ------------------------------------------------------------------
inline void jac_double_pt(thread ulong *oX, thread ulong *oY, thread ulong *oZ,
                          thread const ulong *X,
                          thread const ulong *Y,
                          thread const ulong *Z,
                          device const ulong *q,
                          ulong q_inv_neg)
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
                       thread const ulong *X1,
                       thread const ulong *Y1,
                       thread const ulong *Z1,
                       thread const ulong *X2,
                       thread const ulong *Y2,
                       thread const ulong *Z2,
                       device const ulong *q,
                       ulong q_inv_neg)
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

    mod_sub(H, U2, U1, q);
    mod_sub(R, S2, S1, q);

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

inline bool accum_add_point(thread ulong *AX,
                            thread ulong *AY,
                            thread ulong *AZ,
                            bool have_acc,
                            device const ulong *point_src,
                            device const ulong *q,
                            ulong q_inv_neg)
{
    ulong PX[N_LIMBS], PY[N_LIMBS], PZ[N_LIMBS];
    load_point(PX, PY, PZ, point_src);

    if (is_zero_n(PZ)) {
        return have_acc;
    }

    if (!have_acc) {
        copy_n(AX, PX);
        copy_n(AY, PY);
        copy_n(AZ, PZ);
        return true;
    }

    ulong TX[N_LIMBS], TY[N_LIMBS], TZ[N_LIMBS];
    jac_add_pt(TX, TY, TZ,
               AX, AY, AZ,
               PX, PY, PZ,
               q, q_inv_neg);

    copy_n(AX, TX);
    copy_n(AY, TY);
    copy_n(AZ, TZ);

    return !is_zero_n(AZ);
}

// ------------------------------------------------------------------
// SIMD-group cooperative bucket scatter.
// Each SIMD group owns up to 32 consecutive bucket slots from one
// window. All lanes cooperatively scan scalar chunks; if a source
// scalar's window value falls in this 32-bucket range, the matching
// output lane accumulates that point.
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
    uint num_buckets = (1u << w) - 1u;
    uint total = num_windows * num_buckets;

    uint lane = idx & (SIMD_W - 1u);
    uint sg_base_idx = idx - lane;

    ulong AX[N_LIMBS], AY[N_LIMBS], AZ[N_LIMBS];
    zero_point(AX, AY, AZ);
    bool have_acc = false;

    ulong mask = (1ul << w) - 1ul;

    uint base_window = sg_base_idx / num_buckets;
    uint base_bucket_minus_1 = sg_base_idx - base_window * num_buckets;

    if (base_window < num_windows) {
        uint remaining = num_buckets - base_bucket_minus_1;
        uint active_count = (remaining < SIMD_W) ? remaining : SIMD_W;

        uint bit_offset = base_window * w;
        uint bucket_start = base_bucket_minus_1 + 1u;
        uint bucket_end_excl = bucket_start + active_count;

        for (uint base_pair = 0u; base_pair < n_pairs; base_pair += SIMD_W) {
            uint pair_i = base_pair + lane;
            bool src_valid = (pair_i < n_pairs);

            ulong s0 = src_valid ? scalars[pair_i * 4u] : 0ul;
            uint b = (uint)((s0 >> bit_offset) & mask);

            bool hit = src_valid &&
                       (b >= bucket_start) &&
                       (b < bucket_end_excl);

            uint target_lane = b - bucket_start;
            uint hit_u = hit ? 1u : 0u;

            if (simd_any(hit)) {
                for (uint src = 0u; src < SIMD_W; ++src) {
                    ushort src_lane = (ushort)src;
                    uint src_hit = simd_shuffle(hit_u, src_lane);

                    if (src_hit != 0u) {
                        uint tgt = simd_shuffle(target_lane, src_lane);

                        if (lane == tgt) {
                            uint src_pair_i = base_pair + src;
                            have_acc = accum_add_point(AX, AY, AZ,
                                                       have_acc,
                                                       points_in + src_pair_i * POINT_LIMBS,
                                                       q, q_inv_neg);
                        }
                    }
                }
            }
        }

        if (lane < active_count) {
            if (have_acc) {
                store_point(buckets + idx * POINT_LIMBS, AX, AY, AZ);
            }
            return;
        }

        // Lanes past active_count belong to the next window at a boundary.
        // They fall back to exact per-bucket scan below.
        zero_point(AX, AY, AZ);
        have_acc = false;
    }

    if (idx >= total) return;

    uint window_idx = idx / num_buckets;
    uint bucket_minus_1 = idx - window_idx * num_buckets;
    uint bucket_val = bucket_minus_1 + 1u;
    uint bit_offset = window_idx * w;

    for (uint i = 0u; i < n_pairs; ++i) {
        ulong s0 = scalars[i * 4u];
        uint b = (uint)((s0 >> bit_offset) & mask);

        if (b == bucket_val) {
            have_acc = accum_add_point(AX, AY, AZ,
                                       have_acc,
                                       points_in + i * POINT_LIMBS,
                                       q, q_inv_neg);
        }
    }

    if (have_acc) {
        store_point(buckets + idx * POINT_LIMBS, AX, AY, AZ);
    }
}
```

Incumbent result:
       uniform_N4K: correct, 7.06 ms, 0.8 GB/s (0.4% of 200 GB/s)
      uniform_N16K: correct, 38.03 ms, 0.6 GB/s (0.3% of 200 GB/s)
      uniform_N64K: correct, 321.53 ms, 0.3 GB/s (0.1% of 200 GB/s)
  score (gmean of fraction): 0.0025

## History

- iter  0: compile=OK | correct=True | score=0.00041459428754586365
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.0024614328135078383
- iter  3: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
