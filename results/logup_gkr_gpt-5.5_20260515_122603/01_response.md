The main optimization is replacing 256 per-element inversions with one inversion per 256-element tile: each thread computes its denominator/numerator, then the threadgroup reduces numerator and denominator products and inverts only the denominator product. I also use SIMD-group reductions to minimize threadgroup barriers, and skip numerator reduction for tiles wholly in the witness region where all numerators are one. This should be substantially faster because field inversion via exponentiation dominated the seed kernel.

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constant uint  P_BB_U  = 2013265921u;
constant ulong P_BB    = 2013265921ul;
constant ulong BB_MU   = 9162596893ul; // floor(2^64 / 2013265921)

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

// Fixed exponent p - 2 = 0xFFFFFFFEFFFFFFFF.
// Bits after the top bit: 30 ones, one zero, 32 ones.
inline ulong gold_inv(ulong x) {
    ulong r = x;

    for (uint i = 0u; i < 30u; ++i) {
        r = gold_mul(r, r);
        r = gold_mul(r, x);
    }

    r = gold_mul(r, r); // zero bit

    for (uint i = 0u; i < 32u; ++i) {
        r = gold_mul(r, r);
        r = gold_mul(r, x);
    }

    return r;
}

// ----------------------------------------------------------------------
// BabyBear field.
// ----------------------------------------------------------------------

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + P_BB_U - b);
}

inline uint bb_reduce(ulong x) {
    ulong2 qprod = umul128(x, BB_MU);
    ulong q = qprod.y;
    ulong r = x - q * P_BB;
    if (r >= P_BB) {
        r -= P_BB;
    }
    return (uint)r;
}

inline uint bb_mul(uint a, uint b) {
    return bb_reduce((ulong)a * (ulong)b);
}

// Fixed exponent p - 2 = 0x77FFFFFF.
// Bits after the top bit: two ones, one zero, 27 ones.
inline uint bb_inv(uint x) {
    uint r = x;

    for (uint i = 0u; i < 2u; ++i) {
        r = bb_mul(r, r);
        r = bb_mul(r, x);
    }

    r = bb_mul(r, r); // zero bit

    for (uint i = 0u; i < 27u; ++i) {
        r = bb_mul(r, r);
        r = bb_mul(r, x);
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

inline ulong simd_product_gold(ulong v) {
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)16)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)8)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)4)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)2)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)1)));
    return v;
}

inline uint simd_product_bb(uint v) {
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)16));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)8));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)4));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)2));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)1));
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
// One inversion per 256-element tile:
//   prod(num_i / den_i) = prod(num_i) * inv(prod(den_i)).
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
    bool witness_only_tile = (tile_start + 255ul) < (ulong)N;

    ulong x = 0ul;
    uint  num32 = 1u;
    bool  active = false;

    if (witness_only_tile) {
        x = table[witness_idx[gid]];
    } else {
        ulong total = (ulong)N + (ulong)M;
        active = ((ulong)gid < total);

        if (active) {
            if (gid < N) {
                x = table[witness_idx[gid]];
                num32 = 1u;
            } else {
                uint j = gid - N;
                x = table[j];
                num32 = multiplicities[j];
            }
        }
    }

    if (prime_kind == 0u) {
        if (witness_only_tile) {
            ulong den = gold_sub(alpha, x);
            ulong den_sg = simd_product_gold(den);

            if (lane == 0u) {
                scratch_den[sg] = den_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                ulong d = (lane < 8u) ? scratch_den[lane] : 1ul;
                d = simd_product_gold(d);

                ulong out = gold_inv(d);
                if (lane == 0u) {
                    partial[tgid] = out;
                }
            }
        } else {
            ulong den = active ? gold_sub(alpha, x) : 1ul;
            ulong num = active ? (ulong)num32 : 1ul;

            ulong den_sg = simd_product_gold(den);
            ulong num_sg = simd_product_gold(num);

            if (lane == 0u) {
                scratch_den[sg] = den_sg;
                scratch_num[sg] = num_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                ulong d = (lane < 8u) ? scratch_den[lane] : 1ul;
                ulong n = (lane < 8u) ? scratch_num[lane] : 1ul;

                d = simd_product_gold(d);
                n = simd_product_gold(n);

                if (n == 0ul) {
                    if (lane == 0u) {
                        partial[tgid] = 0ul;
                    }
                } else {
                    ulong out = gold_mul(n, gold_inv(d));
                    if (lane == 0u) {
                        partial[tgid] = out;
                    }
                }
            }
        }
    } else {
        uint alpha32 = (uint)alpha;

        if (witness_only_tile) {
            uint den = bb_sub(alpha32, (uint)x);
            uint den_sg = simd_product_bb(den);

            if (lane == 0u) {
                scratch_den[sg] = (ulong)den_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                uint d = (lane < 8u) ? (uint)scratch_den[lane] : 1u;
                d = simd_product_bb(d);

                uint out = bb_inv(d);
                if (lane == 0u) {
                    partial[tgid] = (ulong)out;
                }
            }
        } else {
            uint den = active ? bb_sub(alpha32, (uint)x) : 1u;
            uint num = active ? num32 : 1u;

            uint den_sg = simd_product_bb(den);
            uint num_sg = simd_product_bb(num);

            if (lane == 0u) {
                scratch_den[sg] = (ulong)den_sg;
                scratch_num[sg] = (ulong)num_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                uint d = (lane < 8u) ? (uint)scratch_den[lane] : 1u;
                uint n = (lane < 8u) ? (uint)scratch_num[lane] : 1u;

                d = simd_product_bb(d);
                n = simd_product_bb(n);

                if (n == 0u) {
                    if (lane == 0u) {
                        partial[tgid] = 0ul;
                    }
                } else {
                    uint out = bb_mul(n, bb_inv(d));
                    if (lane == 0u) {
                        partial[tgid] = (ulong)out;
                    }
                }
            }
        }
    }
}
```