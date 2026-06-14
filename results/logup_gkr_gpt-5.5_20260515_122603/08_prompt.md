## Task: logup_gkr

Batched LogUp lookup-argument core (Haebock 2022; back-end of Jolt / Lasso / Plonkish v2). Given a table T[M] and a witness column w[N] where each w_i := T[witness_idx[i]], compute (1) multiplicities m[j] = #{ i : witness_idx[i] == j } for j in [0, M), and (2) the running product
  P = prod_{i=0..N-1} 1/(alpha - w_i)
     * prod_{j=0..M-1} m_j / (alpha - T_j)   (mod p)
where alpha is a verifier challenge. The host fixes alpha so that alpha is outside the set of table values -- no zero denominators arise.

Combined fingerprint stream of length N + M:
  k <  N:   x_k = T[witness_idx[k]],  num_k = 1
  k >= N:   x_k = T[k - N],            num_k = m[k - N]

The host issues two dispatches in a single compute command encoder. Their serial ordering provides the implicit barrier so the second dispatch sees the first's atomic writes:
  Dispatch 1 (logup_count_mult): one thread per witness     row; atomically increments multiplicities[witness_idx[i]].
  Dispatch 2 (logup_partial_product): each threadgroup     of TG_WIDTH = 256 threads owns 256 consecutive indices     in [0, N+M). Each thread computes num_k * 1/(alpha -     x_k); threadgroup-cooperatively reduces the 256 terms     into one tile product written to partial[tgid]. Threads     with k >= N+M contribute the multiplicative identity     (1). The host then multiplies partial[0..K-1] (K =     ceil((N+M)/256)) on the CPU to obtain the final running     product (the sub-millisecond host fold is intentionally     untimed).

Field selection (constant prime_kind):
  0 = Goldilocks  p = 2^64 - 2^32 + 1
  1 = BabyBear    p = 2^31 - 2^27 + 1 = 2013265921
Both reductions are runtime-dispatched on prime_kind; a candidate that hardcodes the Goldilocks reduction macro, or assumes 64-bit limbs are needed, silently fails the held-out BabyBear probe.

All field elements (table, alpha, partial[]) are canonical uint64 in [0, p); a non-canonical output element is a correctness failure even if its residue class matches. Multiplicities are canonical uint32 counts (promoted to ulong only when used as the numerator).

## Required kernel signature(s)

```
kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]]);

kernel void logup_partial_product(
    device const ulong *table          [[buffer(0)]],
    device const uint  *witness_idx    [[buffer(1)]],
    device const uint  *multiplicities [[buffer(2)]],
    device       ulong *partial        [[buffer(3)]],
    constant uint      &N              [[buffer(4)]],
    constant uint      &M              [[buffer(5)]],
    constant uint      &prime_kind     [[buffer(6)]],
    constant ulong     &alpha          [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]);

Dispatch geometry (host-fixed):
  logup_count_mult:
    threadsPerGrid        = (N rounded up to TG width, 1, 1)
    threadsPerThreadgroup = (min(N, 256), 1, 1)
  logup_partial_product:
    threadsPerGrid        = (K * 256, 1, 1)   K = ceil((N+M)/256)
    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256

The 256-wide threadgroup is part of the host-kernel contract for logup_partial_product: K = ceil((N+M) / 256) is baked into the host-side partial[] allocation, so the kernel must emit exactly one tile-product per 256 consecutive indices. The host pre-zeroes multiplicities[M] before every dispatch. The two dispatches share a single MTLComputeCommandEncoder; the implicit cross-dispatch barrier in serial mode gives logup_partial_product a coherent view of multiplicities written by logup_count_mult.
```

## Your previous attempt

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constant uint  P_BB_U  = 2013265921u;
constant ulong P_BB    = 2013265921ul;
constant uint  BB_MU0  = 0x2222221Du;

constant uint ZERO_PRECHECK_M = 262144u;

// ----------------------------------------------------------------------
// 64 x 64 -> 128 using 32-bit partial products.
// ----------------------------------------------------------------------

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// ----------------------------------------------------------------------
// Goldilocks field.
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) {
        t -= EPSILON;
    }
    return t;
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) {
        t0 -= EPSILON;
    }

    // x_hi_lo * (2^32 - 1), strength-reduced.
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;
    ulong t2 = t0 + t1;
    if (t2 < t0) {
        t2 += EPSILON;
    }

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_sqr(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * (ulong)a0;
    ulong p01 = (ulong)a0 * (ulong)a1;
    ulong p11 = (ulong)a1 * (ulong)a1;

    ulong cross_lo = p01 << 33;
    ulong lo = p00 + cross_lo;
    ulong carry = (lo < p00) ? 1ul : 0ul;
    ulong hi = p11 + (p01 >> 31) + carry;

    return gold_reduce128(lo, hi);
}

inline ulong gold_sqr_n(ulong x, uint n) {
    for (uint i = 0u; i < n; ++i) {
        x = gold_sqr(x);
    }
    return x;
}

// Addition chain for p - 2 = 0xFFFFFFFEFFFFFFFF.
// Reuses x^(2^32-2) both for x^(2^32-1) and the high shifted term.
inline ulong gold_inv(ulong x) {
    ulong x3  = gold_mul(gold_sqr(x), x);
    ulong x7  = gold_mul(gold_sqr(x3), x);

    ulong t = gold_sqr_n(x7, 3u);
    ulong x63 = gold_mul(t, x7);

    t = gold_sqr_n(x63, 6u);
    ulong x4095 = gold_mul(t, x63);

    t = gold_sqr_n(x4095, 12u);
    ulong x2_24m1 = gold_mul(t, x4095);

    t = gold_sqr_n(x2_24m1, 6u);
    ulong x2_30m1 = gold_mul(t, x63);

    ulong x2_31m1 = gold_mul(gold_sqr(x2_30m1), x);
    ulong x2_32m2 = gold_sqr(x2_31m1);
    ulong x2_32m1 = gold_mul(x2_32m2, x);

    t = gold_sqr_n(x2_32m2, 32u);
    return gold_mul(t, x2_32m1);
}

// ----------------------------------------------------------------------
// BabyBear field.
// ----------------------------------------------------------------------

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + P_BB_U - b);
}

inline ulong bb_barrett_q(ulong x) {
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32);

    ulong p0 = (ulong)x0 * (ulong)BB_MU0;
    ulong p1 = (ulong)x1 * (ulong)BB_MU0;

    ulong mid = (p0 >> 32) + (p1 & EPSILON);
    ulong alo = (p0 & EPSILON) | (mid << 32);
    ulong ahi = (p1 >> 32) + (mid >> 32);

    ulong blo = x << 33;
    ulong bhi = x >> 31;
    ulong lo  = alo + blo;
    ulong carry = (lo < alo) ? 1ul : 0ul;

    return ahi + bhi + carry;
}

inline uint bb_reduce(ulong x) {
    ulong q = bb_barrett_q(x);
    ulong r = x - q * P_BB;

    if (r >= P_BB) {
        r -= P_BB;
    }
    if (r >= P_BB) {
        r -= P_BB;
    }
    return (uint)r;
}

inline uint bb_mul(uint a, uint b) {
    return bb_reduce((ulong)a * (ulong)b);
}

inline uint bb_sqr4(uint x) {
    x = bb_mul(x, x);
    x = bb_mul(x, x);
    x = bb_mul(x, x);
    x = bb_mul(x, x);
    return x;
}

// p - 2 = 0x77FFFFFF. Reuses x^14 for both x^15 and the 0x77 prefix.
inline uint bb_inv(uint x) {
    uint x2  = bb_mul(x, x);
    uint x3  = bb_mul(x2, x);
    uint x6  = bb_mul(x3, x3);
    uint x7  = bb_mul(x6, x);
    uint x14 = bb_mul(x7, x7);
    uint x15 = bb_mul(x14, x);

    uint r = x14;
    r = bb_mul(r, r);
    r = bb_mul(r, r);
    r = bb_mul(r, r);
    r = bb_mul(r, x7);

    for (uint i = 0u; i < 6u; ++i) {
        r = bb_sqr4(r);
        r = bb_mul(r, x15);
    }

    return r;
}

// ----------------------------------------------------------------------
// SIMD helpers.
// ----------------------------------------------------------------------

inline uint2 pack_u64(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong unpack_u64(uint2 x) {
    return ((ulong)x.y << 32) | (ulong)x.x;
}

inline ulong simd_product_gold32(ulong v) {
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)16)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)8)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)4)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)2)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)1)));
    return v;
}

inline uint simd_product_bb32(uint v) {
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)16));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)8));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)4));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)2));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)1));
    return v;
}

inline uint simd_and32(uint v) {
    v &= simd_shuffle_xor(v, (ushort)16);
    v &= simd_shuffle_xor(v, (ushort)8);
    v &= simd_shuffle_xor(v, (ushort)4);
    v &= simd_shuffle_xor(v, (ushort)2);
    v &= simd_shuffle_xor(v, (ushort)1);
    return v;
}

// ----------------------------------------------------------------------
// Threadgroup reductions: 256 inputs -> one product in tid 0.
// ----------------------------------------------------------------------

inline ulong tg_reduce_gold_256(
    ulong v,
    threadgroup ulong *scratch,
    uint tid,
    uint lane,
    uint sg)
{
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128u) {
        scratch[tid] = gold_mul(scratch[tid], scratch[tid + 128u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        scratch[tid] = gold_mul(scratch[tid], scratch[tid + 64u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        scratch[tid] = gold_mul(scratch[tid], scratch[tid + 32u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong r = 1ul;
    if (sg == 0u) {
        r = scratch[lane];
        r = simd_product_gold32(r);
    }
    return r;
}

inline ulong2 tg_reduce2_gold_256(
    ulong d,
    uint n0,
    threadgroup ulong *scratch_d,
    threadgroup ulong *scratch_n,
    uint tid,
    uint lane,
    uint sg)
{
    scratch_d[tid] = d;
    scratch_n[tid] = (ulong)n0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128u) {
        scratch_d[tid] = gold_mul(scratch_d[tid], scratch_d[tid + 128u]);

        // Product of two uint32 counts is always canonical in Goldilocks.
        uint a = (uint)scratch_n[tid];
        uint b = (uint)scratch_n[tid + 128u];
        scratch_n[tid] = (ulong)a * (ulong)b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        scratch_d[tid] = gold_mul(scratch_d[tid], scratch_d[tid + 64u]);
        scratch_n[tid] = gold_mul(scratch_n[tid], scratch_n[tid + 64u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        scratch_d[tid] = gold_mul(scratch_d[tid], scratch_d[tid + 32u]);
        scratch_n[tid] = gold_mul(scratch_n[tid], scratch_n[tid + 32u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong rd = 1ul;
    ulong rn = 1ul;
    if (sg == 0u) {
        rd = simd_product_gold32(scratch_d[lane]);
        rn = simd_product_gold32(scratch_n[lane]);
    }
    return ulong2(rd, rn);
}

inline uint tg_reduce_bb_256(
    uint v,
    threadgroup ulong *scratch,
    uint tid,
    uint lane,
    uint sg)
{
    scratch[tid] = (ulong)v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128u) {
        scratch[tid] = (ulong)bb_mul((uint)scratch[tid], (uint)scratch[tid + 128u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        scratch[tid] = (ulong)bb_mul((uint)scratch[tid], (uint)scratch[tid + 64u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        scratch[tid] = (ulong)bb_mul((uint)scratch[tid], (uint)scratch[tid + 32u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r = 1u;
    if (sg == 0u) {
        r = simd_product_bb32((uint)scratch[lane]);
    }
    return r;
}

inline uint2 tg_reduce2_bb_256(
    uint d,
    uint n,
    threadgroup ulong *scratch_d,
    threadgroup ulong *scratch_n,
    uint tid,
    uint lane,
    uint sg)
{
    scratch_d[tid] = (ulong)d;
    scratch_n[tid] = (ulong)n;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 128u) {
        scratch_d[tid] = (ulong)bb_mul((uint)scratch_d[tid], (uint)scratch_d[tid + 128u]);
        scratch_n[tid] = (ulong)bb_mul((uint)scratch_n[tid], (uint)scratch_n[tid + 128u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 64u) {
        scratch_d[tid] = (ulong)bb_mul((uint)scratch_d[tid], (uint)scratch_d[tid + 64u]);
        scratch_n[tid] = (ulong)bb_mul((uint)scratch_n[tid], (uint)scratch_n[tid + 64u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        scratch_d[tid] = (ulong)bb_mul((uint)scratch_d[tid], (uint)scratch_d[tid + 32u]);
        scratch_n[tid] = (ulong)bb_mul((uint)scratch_n[tid], (uint)scratch_n[tid + 32u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rd = 1u;
    uint rn = 1u;
    if (sg == 0u) {
        rd = simd_product_bb32((uint)scratch_d[lane]);
        rn = simd_product_bb32((uint)scratch_n[lane]);
    }
    return uint2(rd, rn);
}

// ----------------------------------------------------------------------
// Kernel A: multiplicity count.
// ----------------------------------------------------------------------

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) {
        return;
    }

    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

// ----------------------------------------------------------------------
// Kernel B: tile product.
// ----------------------------------------------------------------------

kernel void logup_partial_product(
    device const ulong *table          [[buffer(0)]],
    device const uint  *witness_idx    [[buffer(1)]],
    device const uint  *multiplicities [[buffer(2)]],
    device       ulong *partial        [[buffer(3)]],
    constant uint      &N              [[buffer(4)]],
    constant uint      &M              [[buffer(5)]],
    constant uint      &prime_kind     [[buffer(6)]],
    constant ulong     &alpha          [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong scratch_a[256];
    threadgroup ulong scratch_b[256];
    threadgroup uint  scratch_flag[8];

    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong tile_start = (ulong)tgid * 256ul;
    ulong total      = (ulong)N + (ulong)M;

    bool witness_only_tile = (tile_start + 255ul) < (ulong)N;
    bool table_only_tile   = tile_start >= (ulong)N;

    ulong x = 0ul;
    uint  num32 = 1u;
    bool  active = false;

    bool do_zero_precheck =
        table_only_tile &&
        (M >= ZERO_PRECHECK_M) &&
        ((ulong)N <= ((ulong)M << 2));

    if (do_zero_precheck) {
        active = ((ulong)gid < total);
        if (active) {
            uint j = gid - N;
            num32 = multiplicities[j];
        }

        uint nz = (!active || (num32 != 0u)) ? 1u : 0u;
        uint nz_sg = simd_and32(nz);

        if (lane == 0u) {
            scratch_flag[sg] = nz_sg;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_nz = 1u;
        if (lane == 0u) {
            tile_nz = scratch_flag[0] & scratch_flag[1] &
                      scratch_flag[2] & scratch_flag[3] &
                      scratch_flag[4] & scratch_flag[5] &
                      scratch_flag[6] & scratch_flag[7];
        }
        tile_nz = simd_broadcast(tile_nz, (ushort)0);

        if (tile_nz == 0u) {
            if (tid == 0u) {
                partial[tgid] = 0ul;
            }
            return;
        }
    }

    if (witness_only_tile) {
        x = table[witness_idx[gid]];
    } else {
        if (!do_zero_precheck) {
            active = ((ulong)gid < total);
        }

        if (active) {
            if (gid < N) {
                x = table[witness_idx[gid]];
                num32 = 1u;
            } else {
                uint j = gid - N;
                x = table[j];
                if (!do_zero_precheck) {
                    num32 = multiplicities[j];
                }
            }
        }
    }

    if (prime_kind == 0u) {
        if (witness_only_tile) {
            ulong den = gold_sub(alpha, x);
            ulong d = tg_reduce_gold_256(den, scratch_a, tid, lane, sg);

            if (tid == 0u) {
                partial[tgid] = gold_inv(d);
            }
        } else {
            ulong den = active ? gold_sub(alpha, x) : 1ul;
            uint  num = active ? num32 : 1u;

            ulong2 dn = tg_reduce2_gold_256(den, num, scratch_a, scratch_b, tid, lane, sg);

            if (tid == 0u) {
                partial[tgid] = (dn.y == 0ul) ? 0ul : gold_mul(dn.y, gold_inv(dn.x));
            }
        }
    } else {
        uint alpha32 = (uint)alpha;

        if (witness_only_tile) {
            uint den = bb_sub(alpha32, (uint)x);
            uint d = tg_reduce_bb_256(den, scratch_a, tid, lane, sg);

            if (tid == 0u) {
                partial[tgid] = (ulong)bb_inv(d);
            }
        } else {
            uint den = active ? bb_sub(alpha32, (uint)x) : 1u;
            uint num = active ? num32 : 1u;

            uint2 dn = tg_reduce2_bb_256(den, num, scratch_a, scratch_b, tid, lane, sg);

            if (tid == 0u) {
                partial[tgid] = (dn.y == 0u) ? 0ul : (ulong)bb_mul(dn.y, bb_inv(dn.x));
            }
        }
    }
}
```

Result of previous attempt:
          gold_M4K: correct, 0.08 ms, 0.7 Gmodmul/s (int64) (1.4% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 0.41 ms, 2.4 Gmodmul/s (int64) (4.5% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 2.31 ms, 6.8 Gmodmul/s (int64) (12.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0431

## Current best (incumbent)

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constant uint  P_BB_U  = 2013265921u;
constant ulong P_BB    = 2013265921ul;
constant uint  BB_MU0  = 0x2222221Du; // low 32 bits of floor(2^64 / p); high part is 2

constant uint ZERO_PRECHECK_M = 262144u;

// ----------------------------------------------------------------------
// 64 x 64 -> 128 using 32-bit partial products.
// ----------------------------------------------------------------------

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

// ----------------------------------------------------------------------
// Goldilocks field.
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) {
        t -= EPSILON;
    }
    return t;
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) {
        t0 -= EPSILON;
    }

    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) {
        t2 += EPSILON;
    }

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_sqr(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * (ulong)a0;
    ulong p01 = (ulong)a0 * (ulong)a1;
    ulong p11 = (ulong)a1 * (ulong)a1;

    ulong cross_lo = p01 << 33;
    ulong lo = p00 + cross_lo;
    ulong carry = (lo < p00) ? 1ul : 0ul;
    ulong hi = p11 + (p01 >> 31) + carry;

    return gold_reduce128(lo, hi);
}

inline ulong gold_sqr_n(ulong x, uint n) {
    for (uint i = 0u; i < n; ++i) {
        x = gold_sqr(x);
    }
    return x;
}

// Addition chain for p - 2 = 0xFFFFFFFEFFFFFFFF.
inline ulong gold_inv(ulong x) {
    ulong x3  = gold_mul(gold_sqr(x), x);           // x^3
    ulong x7  = gold_mul(gold_sqr(x3), x);          // x^7

    ulong t = gold_sqr_n(x7, 3u);
    ulong x63 = gold_mul(t, x7);                    // x^(2^6 - 1)

    t = gold_sqr_n(x63, 6u);
    ulong x4095 = gold_mul(t, x63);                 // x^(2^12 - 1)

    t = gold_sqr_n(x4095, 12u);
    ulong x2_24m1 = gold_mul(t, x4095);             // x^(2^24 - 1)

    t = gold_sqr_n(x2_24m1, 6u);
    ulong x2_30m1 = gold_mul(t, x63);               // x^(2^30 - 1)

    ulong x2_31m1 = gold_mul(gold_sqr(x2_30m1), x); // x^(2^31 - 1)
    ulong x2_32m1 = gold_mul(gold_sqr(x2_31m1), x); // x^(2^32 - 1)

    t = gold_sqr_n(x2_31m1, 33u);
    return gold_mul(t, x2_32m1);
}

// ----------------------------------------------------------------------
// BabyBear field.
// ----------------------------------------------------------------------

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + P_BB_U - b);
}

// Exact high half of x * floor(2^64 / p), using
// floor(2^64 / p) = 0x000000022222221D.
inline ulong bb_barrett_q(ulong x) {
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32);

    ulong p0 = (ulong)x0 * (ulong)BB_MU0;
    ulong p1 = (ulong)x1 * (ulong)BB_MU0;

    ulong mid = (p0 >> 32) + (p1 & EPSILON);
    ulong alo = (p0 & EPSILON) | (mid << 32);
    ulong ahi = (p1 >> 32) + (mid >> 32);

    // Add x * 2^33.
    ulong blo = x << 33;
    ulong bhi = x >> 31;
    ulong lo  = alo + blo;
    ulong carry = (lo < alo) ? 1ul : 0ul;

    return ahi + bhi + carry;
}

inline uint bb_reduce(ulong x) {
    ulong q = bb_barrett_q(x);
    ulong r = x - q * P_BB;

    if (r >= P_BB) {
        r -= P_BB;
    }
    if (r >= P_BB) {
        r -= P_BB;
    }
    return (uint)r;
}

inline uint bb_mul(uint a, uint b) {
    return bb_reduce((ulong)a * (ulong)b);
}

inline uint bb_sqr4(uint x) {
    x = bb_mul(x, x);
    x = bb_mul(x, x);
    x = bb_mul(x, x);
    x = bb_mul(x, x);
    return x;
}

// p - 2 = 0x77FFFFFF.
inline uint bb_inv(uint x) {
    uint x2  = bb_mul(x, x);
    uint x3  = bb_mul(x2, x);
    uint x7  = bb_mul(bb_mul(x3, x3), x);
    uint x15 = bb_mul(bb_mul(x7, x7), x);

    uint r = x7;

    r = bb_sqr4(r);
    r = bb_mul(r, x7);

    for (uint i = 0u; i < 6u; ++i) {
        r = bb_sqr4(r);
        r = bb_mul(r, x15);
    }

    return r;
}

// ----------------------------------------------------------------------
// SIMD helpers.
// ----------------------------------------------------------------------

inline uint2 pack_u64(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong unpack_u64(uint2 x) {
    return ((ulong)x.y << 32) | (ulong)x.x;
}

inline ulong simd_product_gold32(ulong v) {
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)16)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)8)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)4)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)2)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)1)));
    return v;
}

// Product of lanes 0..7; lane 0 receives the tile-level value.
inline ulong simd_product_gold8(ulong v) {
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)4)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)2)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)1)));
    return v;
}

inline uint simd_product_bb32(uint v) {
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)16));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)8));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)4));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)2));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)1));
    return v;
}

// Product of lanes 0..7; lane 0 receives the tile-level value.
inline uint simd_product_bb8(uint v) {
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)4));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)2));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)1));
    return v;
}

inline uint simd_and32(uint v) {
    v &= simd_shuffle_xor(v, (ushort)16);
    v &= simd_shuffle_xor(v, (ushort)8);
    v &= simd_shuffle_xor(v, (ushort)4);
    v &= simd_shuffle_xor(v, (ushort)2);
    v &= simd_shuffle_xor(v, (ushort)1);
    return v;
}

inline uint simd_and8(uint v) {
    v &= simd_shuffle_xor(v, (ushort)4);
    v &= simd_shuffle_xor(v, (ushort)2);
    v &= simd_shuffle_xor(v, (ushort)1);
    return v;
}

// ----------------------------------------------------------------------
// Kernel A: multiplicity count.
// ----------------------------------------------------------------------

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) {
        return;
    }

    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

// ----------------------------------------------------------------------
// Kernel B: tile product.
// ----------------------------------------------------------------------

kernel void logup_partial_product(
    device const ulong *table          [[buffer(0)]],
    device const uint  *witness_idx    [[buffer(1)]],
    device const uint  *multiplicities [[buffer(2)]],
    device       ulong *partial        [[buffer(3)]],
    constant uint      &N              [[buffer(4)]],
    constant uint      &M              [[buffer(5)]],
    constant uint      &prime_kind     [[buffer(6)]],
    constant ulong     &alpha          [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong scratch_den[8];
    threadgroup ulong scratch_num[8];

    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong tile_start = (ulong)tgid * 256ul;
    ulong total      = (ulong)N + (ulong)M;

    bool witness_only_tile = (tile_start + 255ul) < (ulong)N;
    bool table_only_tile   = tile_start >= (ulong)N;

    ulong x = 0ul;
    uint  num32 = 1u;
    bool  active = false;

    // Large table-only tiles are very often killed by a zero multiplicity.
    // Avoid this overhead on small cases where the extra barriers dominate.
    bool do_zero_precheck =
        table_only_tile &&
        (M >= ZERO_PRECHECK_M) &&
        ((ulong)N <= ((ulong)M << 2));

    if (do_zero_precheck) {
        active = ((ulong)gid < total);
        if (active) {
            uint j = gid - N;
            num32 = multiplicities[j];
        }

        uint nz = (!active || (num32 != 0u)) ? 1u : 0u;
        uint nz_sg = simd_and32(nz);

        if (lane == 0u) {
            scratch_num[sg] = (ulong)nz_sg;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0u) {
            uint v = (lane < 8u) ? (uint)scratch_num[lane] : 1u;
            v = simd_and8(v);
            if (lane == 0u) {
                scratch_num[0] = (ulong)v;
                if (v == 0u) {
                    partial[tgid] = 0ul;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (scratch_num[0] == 0ul) {
            return;
        }
    }

    if (witness_only_tile) {
        x = table[witness_idx[gid]];
    } else {
        if (!do_zero_precheck) {
            active = ((ulong)gid < total);
        }

        if (active) {
            if (gid < N) {
                x = table[witness_idx[gid]];
                num32 = 1u;
            } else {
                uint j = gid - N;
                x = table[j];
                if (!do_zero_precheck) {
                    num32 = multiplicities[j];
                }
            }
        }
    }

    if (prime_kind == 0u) {
        if (witness_only_tile) {
            ulong den = gold_sub(alpha, x);
            ulong den_sg = simd_product_gold32(den);

            if (lane == 0u) {
                scratch_den[sg] = den_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                ulong d = (lane < 8u) ? scratch_den[lane] : 1ul;
                d = simd_product_gold8(d);

                if (lane == 0u) {
                    partial[tgid] = gold_inv(d);
                }
            }
        } else {
            ulong den = active ? gold_sub(alpha, x) : 1ul;
            ulong num = active ? (ulong)num32 : 1ul;

            ulong den_sg = simd_product_gold32(den);
            ulong num_sg = simd_product_gold32(num);

            if (lane == 0u) {
                scratch_den[sg] = den_sg;
                scratch_num[sg] = num_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                ulong d = (lane < 8u) ? scratch_den[lane] : 1ul;
                ulong n = (lane < 8u) ? scratch_num[lane] : 1ul;

                d = simd_product_gold8(d);
                n = simd_product_gold8(n);

                if (lane == 0u) {
                    partial[tgid] = (n == 0ul) ? 0ul : gold_mul(n, gold_inv(d));
                }
            }
        }
    } else {
        uint alpha32 = (uint)alpha;

        if (witness_only_tile) {
            uint den = bb_sub(alpha32, (uint)x);
            uint den_sg = simd_product_bb32(den);

            if (lane == 0u) {
                scratch_den[sg] = (ulong)den_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                uint d = (lane < 8u) ? (uint)scratch_den[lane] : 1u;
                d = simd_product_bb8(d);

                if (lane == 0u) {
                    partial[tgid] = (ulong)bb_inv(d);
                }
            }
        } else {
            uint den = active ? bb_sub(alpha32, (uint)x) : 1u;
            uint num = active ? num32 : 1u;

            uint den_sg = simd_product_bb32(den);
            uint num_sg = simd_product_bb32(num);

            if (lane == 0u) {
                scratch_den[sg] = (ulong)den_sg;
                scratch_num[sg] = (ulong)num_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                uint d = (lane < 8u) ? (uint)scratch_den[lane] : 1u;
                uint n = (lane < 8u) ? (uint)scratch_num[lane] : 1u;

                d = simd_product_bb8(d);
                n = simd_product_bb8(n);

                if (lane == 0u) {
                    partial[tgid] = (n == 0u) ? 0ul : (ulong)bb_mul(n, bb_inv(d));
                }
            }
        }
    }
}
```

Incumbent result:
          gold_M4K: correct, 0.08 ms, 0.8 Gmodmul/s (int64) (1.4% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 0.30 ms, 3.3 Gmodmul/s (int64) (6.1% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 2.56 ms, 6.2 Gmodmul/s (int64) (11.5% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0466

## History

- iter  0: compile=OK | correct=True | score=0.0010133018385154713
- iter  1: compile=OK | correct=True | score=0.03575542992013374
- iter  2: compile=OK | correct=True | score=0.027648432080013392
- iter  3: compile=OK | correct=True | score=0.046580543421673
- iter  4: compile=OK | correct=True | score=0.03312488842076578
- iter  5: compile=OK | correct=True | score=0.04269577181565895
- iter  6: compile=OK | correct=True | score=0.03865940158087462
- iter  7: compile=OK | correct=True | score=0.04314095775262271

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
