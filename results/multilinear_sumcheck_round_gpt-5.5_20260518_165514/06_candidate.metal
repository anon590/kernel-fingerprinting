#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD   = 0xFFFFFFFF00000001ul;
constant ulong EPSILON  = 0x00000000FFFFFFFFul;
constant ulong P_BB     = 2013265921ul;
constant ulong BB_MU    = 9162596893ul;

constant uint SG_COUNT   = 8u;
constant uint MAX_D      = 3u;
constant uint SCR_STRIDE = MAX_D + 1u;

// -----------------------------------------------------------------------------
// 64-bit multiply helpers
// -----------------------------------------------------------------------------

inline ulong2 umul128_u32(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);

    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);

    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);

    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    ulong mid = (ulong)p00_hi + (ulong)p01_lo + (ulong)p10_lo;
    ulong lo  = ((ulong)((uint)mid) << 32) | (ulong)p00_lo;
    ulong p11 = ((ulong)p11_hi << 32) | (ulong)p11_lo;
    ulong hi  = p11 + (ulong)p01_hi + (ulong)p10_hi + (mid >> 32);

    return ulong2(lo, hi);
}

inline ulong umulhi64_u32(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_hi = mulhi(a0, b0);

    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);

    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);

    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    ulong mid = (ulong)p00_hi + (ulong)p01_lo + (ulong)p10_lo;
    ulong p11 = ((ulong)p11_hi << 32) | (ulong)p11_lo;

    return p11 + (ulong)p01_hi + (ulong)p10_hi + (mid >> 32);
}

// -----------------------------------------------------------------------------
// Goldilocks field
// -----------------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_reduce128(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong t0 = lo - hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = (hi_lo << 32) - hi_lo;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128_u32(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_mul_add(ulong a, ulong b, ulong c) {
    ulong2 p = umul128_u32(a, b);
    ulong lo = p.x + c;
    ulong hi = p.y + ((lo < p.x) ? 1ul : 0ul);
    return gold_reduce128(lo, hi);
}

// -----------------------------------------------------------------------------
// BabyBear field
// -----------------------------------------------------------------------------

inline ulong bb_add(ulong a, ulong b) {
    ulong t = a + b;
    return (t >= P_BB) ? (t - P_BB) : t;
}

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_reduce_wide(ulong x) {
    ulong q = umulhi64_u32(x, BB_MU);
    ulong r = x - q * P_BB;
    if (r >= P_BB) r -= P_BB;
    if (r >= P_BB) r -= P_BB;
    return r;
}

inline ulong bb_mul(ulong a, ulong b) {
    return bb_reduce_wide((ulong)((uint)a) * (ulong)((uint)b));
}

inline ulong bb_mul_add(ulong a, ulong b, ulong c) {
    return bb_reduce_wide((ulong)((uint)a) * (ulong)((uint)b) + c);
}

// -----------------------------------------------------------------------------
// SIMD shuffle / reductions
// -----------------------------------------------------------------------------

inline ulong shuffle_xor_ulong(ulong x, ushort mask) {
    uint2 v = uint2((uint)x, (uint)(x >> 32));
    uint2 w = simd_shuffle_xor(v, mask);
    return ((ulong)w.y << 32) | (ulong)w.x;
}

inline ulong2 shuffle_xor_ulong2(ulong2 x, ushort mask) {
    uint4 v = uint4((uint)x.x, (uint)(x.x >> 32),
                    (uint)x.y, (uint)(x.y >> 32));
    uint4 w = simd_shuffle_xor(v, mask);
    return ulong2(((ulong)w.y << 32) | (ulong)w.x,
                  ((ulong)w.w << 32) | (ulong)w.z);
}

inline ulong2 gold_add2(ulong2 a, ulong2 b) {
    return ulong2(gold_add(a.x, b.x), gold_add(a.y, b.y));
}

inline ulong simd_sum_gold(ulong x) {
    x = gold_add(x, shuffle_xor_ulong(x, 16));
    x = gold_add(x, shuffle_xor_ulong(x,  8));
    x = gold_add(x, shuffle_xor_ulong(x,  4));
    x = gold_add(x, shuffle_xor_ulong(x,  2));
    x = gold_add(x, shuffle_xor_ulong(x,  1));
    return x;
}

inline ulong2 simd_sum_gold_pair(ulong2 x) {
    x = gold_add2(x, shuffle_xor_ulong2(x, 16));
    x = gold_add2(x, shuffle_xor_ulong2(x,  8));
    x = gold_add2(x, shuffle_xor_ulong2(x,  4));
    x = gold_add2(x, shuffle_xor_ulong2(x,  2));
    x = gold_add2(x, shuffle_xor_ulong2(x,  1));
    return x;
}

inline ulong simd_sum_bb(ulong x) {
    uint v = (uint)x;

    uint y = simd_shuffle_xor(v, (ushort)16);
    ulong s = (ulong)v + (ulong)y;
    v = (uint)((s >= P_BB) ? (s - P_BB) : s);

    y = simd_shuffle_xor(v, (ushort)8);
    s = (ulong)v + (ulong)y;
    v = (uint)((s >= P_BB) ? (s - P_BB) : s);

    y = simd_shuffle_xor(v, (ushort)4);
    s = (ulong)v + (ulong)y;
    v = (uint)((s >= P_BB) ? (s - P_BB) : s);

    y = simd_shuffle_xor(v, (ushort)2);
    s = (ulong)v + (ulong)y;
    v = (uint)((s >= P_BB) ? (s - P_BB) : s);

    y = simd_shuffle_xor(v, (ushort)1);
    s = (ulong)v + (ulong)y;
    v = (uint)((s >= P_BB) ? (s - P_BB) : s);

    return (ulong)v;
}

inline ulong2 simd_sum_bb_pair(ulong2 x) {
    uint2 v = uint2((uint)x.x, (uint)x.y);

    uint2 y = simd_shuffle_xor(v, (ushort)16);
    ulong sx = (ulong)v.x + (ulong)y.x;
    ulong sy = (ulong)v.y + (ulong)y.y;
    v.x = (uint)((sx >= P_BB) ? (sx - P_BB) : sx);
    v.y = (uint)((sy >= P_BB) ? (sy - P_BB) : sy);

    y = simd_shuffle_xor(v, (ushort)8);
    sx = (ulong)v.x + (ulong)y.x;
    sy = (ulong)v.y + (ulong)y.y;
    v.x = (uint)((sx >= P_BB) ? (sx - P_BB) : sx);
    v.y = (uint)((sy >= P_BB) ? (sy - P_BB) : sy);

    y = simd_shuffle_xor(v, (ushort)4);
    sx = (ulong)v.x + (ulong)y.x;
    sy = (ulong)v.y + (ulong)y.y;
    v.x = (uint)((sx >= P_BB) ? (sx - P_BB) : sx);
    v.y = (uint)((sy >= P_BB) ? (sy - P_BB) : sy);

    y = simd_shuffle_xor(v, (ushort)2);
    sx = (ulong)v.x + (ulong)y.x;
    sy = (ulong)v.y + (ulong)y.y;
    v.x = (uint)((sx >= P_BB) ? (sx - P_BB) : sx);
    v.y = (uint)((sy >= P_BB) ? (sy - P_BB) : sy);

    y = simd_shuffle_xor(v, (ushort)1);
    sx = (ulong)v.x + (ulong)y.x;
    sy = (ulong)v.y + (ulong)y.y;
    v.x = (uint)((sx >= P_BB) ? (sx - P_BB) : sx);
    v.y = (uint)((sy >= P_BB) ? (sy - P_BB) : sy);

    return ulong2((ulong)v.x, (ulong)v.y);
}

// -----------------------------------------------------------------------------
// Threadgroup reduction helpers
// -----------------------------------------------------------------------------

inline void reduce_store_gold_d2(
    ulong v0, ulong v1, ulong v2,
    device ulong *partial,
    uint tid,
    uint tgid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong2 v01 = simd_sum_gold_pair(ulong2(v0, v1));
    v2 = simd_sum_gold(v2);

    if (lane == 0u) {
        uint o = sg * SCR_STRIDE;
        scratch[o + 0u] = v01.x;
        scratch[o + 1u] = v01.y;
        scratch[o + 2u] = v2;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 3u) {
        ulong acc = scratch[0u * SCR_STRIDE + tid];
        acc = gold_add(acc, scratch[1u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[2u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[3u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[4u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[5u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[6u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[7u * SCR_STRIDE + tid]);

        partial[tgid * 3u + tid] = acc;
    }
}

inline void reduce_store_bb_d2(
    ulong v0, ulong v1, ulong v2,
    device ulong *partial,
    uint tid,
    uint tgid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong2 v01 = simd_sum_bb_pair(ulong2(v0, v1));
    v2 = simd_sum_bb(v2);

    if (lane == 0u) {
        uint o = sg * SCR_STRIDE;
        scratch[o + 0u] = v01.x;
        scratch[o + 1u] = v01.y;
        scratch[o + 2u] = v2;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 3u) {
        ulong acc = scratch[0u * SCR_STRIDE + tid];
        acc = bb_add(acc, scratch[1u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[2u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[3u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[4u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[5u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[6u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[7u * SCR_STRIDE + tid]);

        partial[tgid * 3u + tid] = acc;
    }
}

inline void reduce_store_gold(
    ulong v0, ulong v1, ulong v2, ulong v3,
    uint d,
    device ulong *partial,
    uint tid,
    uint tgid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong2 v01 = simd_sum_gold_pair(ulong2(v0, v1));
    ulong2 v23 = ulong2(0ul, 0ul);
    if (d >= 2u) {
        v23 = simd_sum_gold_pair(ulong2(v2, v3));
    }

    if (lane == 0u) {
        uint o = sg * SCR_STRIDE;
        scratch[o + 0u] = v01.x;
        if (d >= 1u) scratch[o + 1u] = v01.y;
        if (d >= 2u) scratch[o + 2u] = v23.x;
        if (d >= 3u) scratch[o + 3u] = v23.y;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid <= d) {
        ulong acc = scratch[0u * SCR_STRIDE + tid];
        acc = gold_add(acc, scratch[1u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[2u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[3u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[4u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[5u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[6u * SCR_STRIDE + tid]);
        acc = gold_add(acc, scratch[7u * SCR_STRIDE + tid]);

        partial[tgid * (d + 1u) + tid] = acc;
    }
}

inline void reduce_store_bb(
    ulong v0, ulong v1, ulong v2, ulong v3,
    uint d,
    device ulong *partial,
    uint tid,
    uint tgid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong2 v01 = simd_sum_bb_pair(ulong2(v0, v1));
    ulong2 v23 = ulong2(0ul, 0ul);
    if (d >= 2u) {
        v23 = simd_sum_bb_pair(ulong2(v2, v3));
    }

    if (lane == 0u) {
        uint o = sg * SCR_STRIDE;
        scratch[o + 0u] = v01.x;
        if (d >= 1u) scratch[o + 1u] = v01.y;
        if (d >= 2u) scratch[o + 2u] = v23.x;
        if (d >= 3u) scratch[o + 3u] = v23.y;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid <= d) {
        ulong acc = scratch[0u * SCR_STRIDE + tid];
        acc = bb_add(acc, scratch[1u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[2u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[3u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[4u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[5u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[6u * SCR_STRIDE + tid]);
        acc = bb_add(acc, scratch[7u * SCR_STRIDE + tid]);

        partial[tgid * (d + 1u) + tid] = acc;
    }
}

inline void reduce_store_one_gold(
    ulong v,
    device ulong *partial,
    uint out_idx,
    uint tid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    v = simd_sum_gold(v);

    if (lane == 0u) {
        scratch[sg] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        ulong acc = scratch[0u];
        acc = gold_add(acc, scratch[1u]);
        acc = gold_add(acc, scratch[2u]);
        acc = gold_add(acc, scratch[3u]);
        acc = gold_add(acc, scratch[4u]);
        acc = gold_add(acc, scratch[5u]);
        acc = gold_add(acc, scratch[6u]);
        acc = gold_add(acc, scratch[7u]);
        partial[out_idx] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void reduce_store_one_bb(
    ulong v,
    device ulong *partial,
    uint out_idx,
    uint tid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    v = simd_sum_bb(v);

    if (lane == 0u) {
        scratch[sg] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        ulong acc = scratch[0u];
        acc = bb_add(acc, scratch[1u]);
        acc = bb_add(acc, scratch[2u]);
        acc = bb_add(acc, scratch[3u]);
        acc = bb_add(acc, scratch[4u]);
        acc = bb_add(acc, scratch[5u]);
        acc = bb_add(acc, scratch[6u]);
        acc = bb_add(acc, scratch[7u]);
        partial[out_idx] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// -----------------------------------------------------------------------------
// Kernel A
// -----------------------------------------------------------------------------

kernel void sumcheck_round_h(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *partial    [[buffer(1)]],
    constant uint      &k_log      [[buffer(2)]],
    constant uint      &d_deg      [[buffer(3)]],
    constant uint      &prime_kind [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong scratch[SG_COUNT * SCR_STRIDE];

    uint d      = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base   = half_n << 1;

    ulong s0 = 0ul;
    ulong s1 = 0ul;
    ulong s2 = 0ul;
    ulong s3 = 0ul;

    if (d == 2u) {
        if (prime_kind == 0u) {
            if (k_log >= 18u) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];

                ulong da = gold_sub(a1, a0);
                ulong db = gold_sub(b1, b0);
                ulong a2 = gold_add(a1, da);
                ulong b2 = gold_add(b1, db);

                s0 = gold_mul(a0, b0);
                s1 = gold_mul(a1, b1);
                s2 = gold_mul(a2, b2);
            } else if (gid < half_n) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];

                ulong da = gold_sub(a1, a0);
                ulong db = gold_sub(b1, b0);
                ulong a2 = gold_add(a1, da);
                ulong b2 = gold_add(b1, db);

                s0 = gold_mul(a0, b0);
                s1 = gold_mul(a1, b1);
                s2 = gold_mul(a2, b2);
            }

            reduce_store_gold_d2(s0, s1, s2, partial, tid, tgid, scratch);
            return;
        } else {
            if (gid < half_n) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];

                ulong da = bb_sub(a1, a0);
                ulong db = bb_sub(b1, b0);
                ulong a2 = bb_add(a1, da);
                ulong b2 = bb_add(b1, db);

                s0 = bb_mul(a0, b0);
                s1 = bb_mul(a1, b1);
                s2 = bb_mul(a2, b2);
            }

            reduce_store_bb_d2(s0, s1, s2, partial, tid, tgid, scratch);
            return;
        }
    }

    if (d == 1u || d == 3u) {
        if (prime_kind == 0u) {
            if (gid < half_n) {
                if (d == 1u) {
                    s0 = f_in[gid];
                    s1 = f_in[gid + half_n];
                } else {
                    uint base2 = base + base;

                    ulong a0 = f_in[gid];
                    ulong a1 = f_in[gid + half_n];
                    ulong b0 = f_in[base + gid];
                    ulong b1 = f_in[base + gid + half_n];
                    ulong c0 = f_in[base2 + gid];
                    ulong c1 = f_in[base2 + gid + half_n];

                    ulong da = gold_sub(a1, a0);
                    ulong db = gold_sub(b1, b0);
                    ulong dc = gold_sub(c1, c0);

                    ulong a2 = gold_add(a1, da);
                    ulong b2 = gold_add(b1, db);
                    ulong c2 = gold_add(c1, dc);

                    ulong a3 = gold_add(a2, da);
                    ulong b3 = gold_add(b2, db);
                    ulong c3 = gold_add(c2, dc);

                    s0 = gold_mul(gold_mul(a0, b0), c0);
                    s1 = gold_mul(gold_mul(a1, b1), c1);
                    s2 = gold_mul(gold_mul(a2, b2), c2);
                    s3 = gold_mul(gold_mul(a3, b3), c3);
                }
            }

            reduce_store_gold(s0, s1, s2, s3, d, partial, tid, tgid, scratch);
        } else {
            if (gid < half_n) {
                if (d == 1u) {
                    s0 = f_in[gid];
                    s1 = f_in[gid + half_n];
                } else {
                    uint base2 = base + base;

                    ulong a0 = f_in[gid];
                    ulong a1 = f_in[gid + half_n];
                    ulong b0 = f_in[base + gid];
                    ulong b1 = f_in[base + gid + half_n];
                    ulong c0 = f_in[base2 + gid];
                    ulong c1 = f_in[base2 + gid + half_n];

                    ulong da = bb_sub(a1, a0);
                    ulong db = bb_sub(b1, b0);
                    ulong dc = bb_sub(c1, c0);

                    ulong a2 = bb_add(a1, da);
                    ulong b2 = bb_add(b1, db);
                    ulong c2 = bb_add(c1, dc);

                    ulong a3 = bb_add(a2, da);
                    ulong b3 = bb_add(b2, db);
                    ulong c3 = bb_add(c2, dc);

                    s0 = bb_mul(bb_mul(a0, b0), c0);
                    s1 = bb_mul(bb_mul(a1, b1), c1);
                    s2 = bb_mul(bb_mul(a2, b2), c2);
                    s3 = bb_mul(bb_mul(a3, b3), c3);
                }
            }

            reduce_store_bb(s0, s1, s2, s3, d, partial, tid, tgid, scratch);
        }

        return;
    }

    // Generic runtime-degree fallback.
    if (prime_kind == 0u) {
        for (uint t = 0u; t <= d; ++t) {
            ulong prod = 0ul;

            if (gid < half_n) {
                prod = 1ul;
                uint off = gid;

                for (uint i = 0u; i < d; ++i) {
                    ulong f0 = f_in[off];
                    ulong f1 = f_in[off + half_n];

                    ulong v;
                    if (t == 0u) {
                        v = f0;
                    } else if (t == 1u) {
                        v = f1;
                    } else {
                        ulong delta = gold_sub(f1, f0);
                        v = gold_mul_add((ulong)t, delta, f0);
                    }

                    prod = gold_mul(prod, v);
                    off += base;
                }
            }

            reduce_store_one_gold(prod, partial, tgid * (d + 1u) + t, tid, scratch);
        }
    } else {
        for (uint t = 0u; t <= d; ++t) {
            ulong prod = 0ul;

            if (gid < half_n) {
                prod = 1ul;
                uint off = gid;

                for (uint i = 0u; i < d; ++i) {
                    ulong f0 = f_in[off];
                    ulong f1 = f_in[off + half_n];

                    ulong v;
                    if (t == 0u) {
                        v = f0;
                    } else if (t == 1u) {
                        v = f1;
                    } else {
                        ulong delta = bb_sub(f1, f0);
                        v = bb_mul_add((ulong)t, delta, f0);
                    }

                    prod = bb_mul(prod, v);
                    off += base;
                }
            }

            reduce_store_one_bb(prod, partial, tgid * (d + 1u) + t, tid, scratch);
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel B
// -----------------------------------------------------------------------------

kernel void sumcheck_fold(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *f_out      [[buffer(1)]],
    constant ulong     &r          [[buffer(2)]],
    constant uint      &k_log      [[buffer(3)]],
    constant uint      &d_deg      [[buffer(4)]],
    constant uint      &prime_kind [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint shift  = k_log - 1u;
    uint half_n = 1u << shift;

    if (d_deg == 2u) {
        if (k_log >= 18u) {
            uint off = gid + (gid & half_n);

            ulong f0 = f_in[off];
            ulong f1 = f_in[off + half_n];

            if (prime_kind == 0u) {
                ulong delta = gold_sub(f1, f0);
                f_out[gid] = gold_mul_add(r, delta, f0);
            } else {
                ulong delta = bb_sub(f1, f0);
                f_out[gid] = bb_mul_add(r, delta, f0);
            }
            return;
        }

        uint total = half_n << 1;
        if (gid >= total) return;

        uint off = gid + ((gid >= half_n) ? half_n : 0u);

        ulong f0 = f_in[off];
        ulong f1 = f_in[off + half_n];

        if (prime_kind == 0u) {
            ulong delta = gold_sub(f1, f0);
            f_out[gid] = gold_add(f0, gold_mul(r, delta));
        } else {
            ulong delta = bb_sub(f1, f0);
            f_out[gid] = bb_add(f0, bb_mul(r, delta));
        }
        return;
    }

    uint total = d_deg * half_n;
    if (gid >= total) return;

    uint poly_i = gid >> shift;
    uint j      = gid & (half_n - 1u);
    uint base   = half_n << 1;
    uint off    = poly_i * base + j;

    ulong f0 = f_in[off];
    ulong f1 = f_in[off + half_n];

    if (prime_kind == 0u) {
        ulong delta = gold_sub(f1, f0);
        f_out[gid] = gold_add(f0, gold_mul(r, delta));
    } else {
        ulong delta = bb_sub(f1, f0);
        f_out[gid] = bb_add(f0, bb_mul(r, delta));
    }
}