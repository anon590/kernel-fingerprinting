This fixes the unsupported acquire/release atomics by using Metal’s available relaxed atomics, while preserving the embedded bucket lock. The main optimization is switching from cooperative per-bucket full scans to a lock-protected scatter over pair/window jobs: first writers copy directly, and only true bucket collisions perform Jacobian additions. For the benchmark sizes this reduces scalar work from scanning all pairs for every bucket group to one scalar-window extraction per pair/window, with field arithmetic only on populated collision chains.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_LIMBS     = 6u;
constexpr constant uint POINT_LIMBS = 18u;

constexpr constant uint Z_BASE_LIMB = 12u;
constexpr constant uint Z5_LO_WORD  = 34u;
constexpr constant uint Z5_HI_WORD  = 35u;
constexpr constant uint DISPATCH_ALIGN = 64u;

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
                                                      memory_order_relaxed,
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
    atomic_store_explicit(lockp, z5_hi, memory_order_relaxed);
}

inline uint scalar_window(device const ulong *scalars,
                          uint pair_i,
                          uint window_idx,
                          uint w)
{
    device const ulong *sp = scalars + pair_i * 4u;

    // Fast path for the required runtime parameters: w = 16, num_windows = 4.
    if (w == 16u && window_idx < 4u) {
        return (uint)((sp[0] >> (window_idx << 4u)) & 0xFFFFul);
    }

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
// Lock-protected scatter.
// The dispatch is sized for bucket slots; with w=16,num_windows=4 the
// rounded dispatch has exactly enough lanes for up to 64K pair/window
// jobs. For larger n_pairs, lanes stride by the rounded dispatch size.
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

    if (num_windows == 0u || w == 0u || w >= 32u) return;

    uint num_buckets = (1u << w) - 1u;
    if (num_buckets == 0u) return;

    uint bucket_slots   = num_windows * num_buckets;
    uint dispatch_slots = (bucket_slots + (DISPATCH_ALIGN - 1u)) & ~(DISPATCH_ALIGN - 1u);
    if (dispatch_slots == 0u) return;

    uint total_jobs = n_pairs * num_windows;

    for (uint job = idx; job < total_jobs; job += dispatch_slots) {
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