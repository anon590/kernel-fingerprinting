I lower the zero-multiplicity precheck threshold and make it cover boundary/table tiles, using a one-barrier threadgroup flag reduction to skip whole 256-term tiles that must be zero. I also replace several 64-bit partial products with explicit 32-bit `mulhi` recombination and unroll inversion square loops, which should reduce arithmetic overhead on Apple GPUs while preserving exact canonical field outputs.

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constant uint  P_BB_U  = 2013265921u;
constant ulong P_BB    = 2013265921ul;
constant uint  BB_MU0  = 0x2222221Du; // floor(2^64 / p) low limb; high limb is 2

constant uint ZERO_PRECHECK_M = 4096u;

// ----------------------------------------------------------------------
// 64 x 64 -> 128 from 32-bit limbs.
// ----------------------------------------------------------------------

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00l = a0 * b0;
    uint p00h = mulhi(a0, b0);
    uint p01l = a0 * b1;
    uint p01h = mulhi(a0, b1);
    uint p10l = a1 * b0;
    uint p10h = mulhi(a1, b0);
    uint p11l = a1 * b1;
    uint p11h = mulhi(a1, b1);

    ulong s  = (ulong)p00h + (ulong)p01l + (ulong)p10l;
    ulong lo = ((ulong)((uint)s) << 32) | (ulong)p00l;

    ulong hi = ((ulong)p11h << 32) + (ulong)p11l
             + (ulong)p01h + (ulong)p10h + (s >> 32);

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
    uint hi_lo = (uint)x_hi;
    uint hi_hi = (uint)(x_hi >> 32);

    ulong t = x_lo - (ulong)hi_hi;
    if (t > x_lo) {
        t -= EPSILON;
    }

    ulong u = ((ulong)hi_lo << 32) - (ulong)hi_lo;
    ulong r = t + u;
    if (r < t) {
        r += EPSILON;
    }

    return gold_canonical(r);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_sqr(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00l = a0 * a0;
    uint p00h = mulhi(a0, a0);
    uint p01l = a0 * a1;
    uint p01h = mulhi(a0, a1);
    uint p11l = a1 * a1;
    uint p11h = mulhi(a1, a1);

    uint dbl_l       = p01l << 1;
    uint dbl_h_low   = (p01h << 1) | (p01l >> 31);
    uint dbl_h_carry = p01h >> 31;

    ulong s  = (ulong)p00h + (ulong)dbl_l;
    ulong lo = ((ulong)((uint)s) << 32) | (ulong)p00l;

    ulong hi = ((ulong)p11h << 32) + (ulong)p11l
             + ((ulong)dbl_h_carry << 32)
             + (ulong)dbl_h_low + (s >> 32);

    return gold_reduce128(lo, hi);
}

inline ulong gold_sqr_3(ulong x) {
    x = gold_sqr(x);
    x = gold_sqr(x);
    x = gold_sqr(x);
    return x;
}

inline ulong gold_sqr_6(ulong x) {
    x = gold_sqr_3(x);
    x = gold_sqr_3(x);
    return x;
}

inline ulong gold_sqr_12(ulong x) {
    x = gold_sqr_6(x);
    x = gold_sqr_6(x);
    return x;
}

inline ulong gold_sqr_33(ulong x) {
    x = gold_sqr_12(x);
    x = gold_sqr_12(x);
    x = gold_sqr_6(x);
    x = gold_sqr_3(x);
    return x;
}

// Addition chain for p - 2 = 0xFFFFFFFEFFFFFFFF.
inline ulong gold_inv(ulong x) {
    ulong x3  = gold_mul(gold_sqr(x), x);           // x^3
    ulong x7  = gold_mul(gold_sqr(x3), x);          // x^7

    ulong t = gold_sqr_3(x7);
    ulong x63 = gold_mul(t, x7);                    // x^(2^6 - 1)

    t = gold_sqr_6(x63);
    ulong x4095 = gold_mul(t, x63);                 // x^(2^12 - 1)

    t = gold_sqr_12(x4095);
    ulong x2_24m1 = gold_mul(t, x4095);             // x^(2^24 - 1)

    t = gold_sqr_6(x2_24m1);
    ulong x2_30m1 = gold_mul(t, x63);               // x^(2^30 - 1)

    ulong x2_31m1 = gold_mul(gold_sqr(x2_30m1), x); // x^(2^31 - 1)
    ulong x2_32m1 = gold_mul(gold_sqr(x2_31m1), x); // x^(2^32 - 1)

    t = gold_sqr_33(x2_31m1);
    return gold_mul(t, x2_32m1);
}

// ----------------------------------------------------------------------
// BabyBear field.
// ----------------------------------------------------------------------

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + P_BB_U - b);
}

// Exact high half of x * floor(2^64 / p), where
// floor(2^64 / p) = 0x000000022222221D.
inline ulong bb_barrett_q(ulong x) {
    uint x0 = (uint)x;
    uint x1 = (uint)(x >> 32);

    uint p0l = x0 * BB_MU0;
    uint p0h = mulhi(x0, BB_MU0);
    uint p1l = x1 * BB_MU0;
    uint p1h = mulhi(x1, BB_MU0);

    ulong mid = (ulong)p0h + (ulong)p1l;
    ulong alo = ((ulong)((uint)mid) << 32) | (ulong)p0l;
    ulong ahi = (ulong)p1h + (mid >> 32);

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

    r = bb_sqr4(r);
    r = bb_mul(r, x15);
    r = bb_sqr4(r);
    r = bb_mul(r, x15);
    r = bb_sqr4(r);
    r = bb_mul(r, x15);
    r = bb_sqr4(r);
    r = bb_mul(r, x15);
    r = bb_sqr4(r);
    r = bb_mul(r, x15);
    r = bb_sqr4(r);
    r = bb_mul(r, x15);

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
    threadgroup uint  scratch_flag[8];

    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong tile_start = (ulong)tgid * 256ul;
    ulong total      = (ulong)N + (ulong)M;

    bool witness_only_tile = (tile_start + 255ul) < (ulong)N;

    ulong x = 0ul;
    uint  num32 = 1u;
    bool  active = false;

    // If a table-side numerator is zero, the whole tile product is zero.
    // This is common when N/M is small-to-moderate; do it before loading
    // table values or reducing denominators. Covers table-only and the
    // single boundary tile.
    bool do_zero_precheck =
        (!witness_only_tile) &&
        (M >= ZERO_PRECHECK_M) &&
        ((ulong)N <= ((ulong)M << 2));

    if (do_zero_precheck) {
        active = ((ulong)gid < total);

        uint nz = 1u;
        if (active && (gid >= N)) {
            uint j = gid - N;
            num32 = multiplicities[j];
            nz = (num32 != 0u) ? 1u : 0u;
        }

        uint nz_sg = simd_and32(nz);
        if (lane == 0u) {
            scratch_flag[sg] = nz_sg;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_nz = scratch_flag[0] & scratch_flag[1] &
                       scratch_flag[2] & scratch_flag[3] &
                       scratch_flag[4] & scratch_flag[5] &
                       scratch_flag[6] & scratch_flag[7];

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