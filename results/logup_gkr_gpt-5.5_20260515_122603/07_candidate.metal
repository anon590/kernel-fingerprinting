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