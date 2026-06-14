#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;

constant uint TG_WIDTH = 256u;
constant uint MAX_D    = 8u;

// ---------------------- Goldilocks helpers ----------------------------

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

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
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

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

// 2*a mod p for Goldilocks: standard add
inline ulong gold_dbl(ulong a) {
    return gold_add(a, a);
}

// ---------------------- BabyBear helpers ------------------------------

inline ulong bb_add(ulong a, ulong b) {
    ulong t = a + b;
    return (t >= P_BB) ? (t - P_BB) : t;
}

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

inline ulong bb_dbl(ulong a) {
    return bb_add(a, a);
}

// ---- ulong simd shuffle via uint halves -----------------------------

inline ulong simd_shfl_xor_ulong(ulong v, ushort mask) {
    uint lo = (uint)(v);
    uint hi = (uint)(v >> 32);
    lo = simd_shuffle_xor(lo, mask);
    hi = simd_shuffle_xor(hi, mask);
    return ((ulong)hi << 32) | (ulong)lo;
}

inline ulong simd_reduce_add_gold(ulong v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        ulong other = simd_shfl_xor_ulong(v, off);
        v = gold_add(v, other);
    }
    return v;
}

inline ulong simd_reduce_add_bb(ulong v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        ulong other = simd_shfl_xor_ulong(v, off);
        v = bb_add(v, other);
    }
    return v;
}

// ----------------------------------------------------------------------
// Kernel A
// ----------------------------------------------------------------------

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
    threadgroup ulong scratch[8u * (MAX_D + 1u)];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    ulong term[MAX_D + 1u];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

    bool active = (gid < half_n);

    // Fast path: d == 2. Compute h(0), h(1) directly from f^(0), f^(1)
    // and h(2) = prod_i (2*f1 - f0). Avoids delta-loop bookkeeping.
    if (d == 2u) {
        if (active) {
            if (prime_kind == 0u) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];
                // h(0) = a0 * b0
                term[0] = gold_mul(a0, b0);
                // h(1) = a1 * b1
                term[1] = gold_mul(a1, b1);
                // h(2) = (2*a1 - a0) * (2*b1 - b0)
                ulong a2 = gold_sub(gold_dbl(a1), a0);
                ulong b2 = gold_sub(gold_dbl(b1), b0);
                term[2] = gold_mul(a2, b2);
            } else {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];
                term[0] = bb_mul(a0, b0);
                term[1] = bb_mul(a1, b1);
                ulong a2 = bb_sub(bb_dbl(a1), a0);
                ulong b2 = bb_sub(bb_dbl(b1), b0);
                term[2] = bb_mul(a2, b2);
            }
        }
    } else {
        // General path: affine-step recurrence over t.
        if (prime_kind == 0u) {
            if (active) {
                ulong f0 = f_in[gid];
                ulong f1 = f_in[gid + half_n];
                ulong delta = gold_sub(f1, f0);
                ulong ft = f0;
                term[0] = ft;
                for (uint t = 1u; t <= d; ++t) {
                    ft = gold_add(ft, delta);
                    term[t] = ft;
                }
                for (uint i = 1u; i < d; ++i) {
                    ulong g0 = f_in[i * base + gid];
                    ulong g1 = f_in[i * base + gid + half_n];
                    ulong gd = gold_sub(g1, g0);
                    ulong gt = g0;
                    term[0] = gold_mul(term[0], gt);
                    for (uint t = 1u; t <= d; ++t) {
                        gt = gold_add(gt, gd);
                        term[t] = gold_mul(term[t], gt);
                    }
                }
            }
        } else {
            if (active) {
                ulong f0 = f_in[gid];
                ulong f1 = f_in[gid + half_n];
                ulong delta = bb_sub(f1, f0);
                ulong ft = f0;
                term[0] = ft;
                for (uint t = 1u; t <= d; ++t) {
                    ft = bb_add(ft, delta);
                    term[t] = ft;
                }
                for (uint i = 1u; i < d; ++i) {
                    ulong g0 = f_in[i * base + gid];
                    ulong g1 = f_in[i * base + gid + half_n];
                    ulong gd = bb_sub(g1, g0);
                    ulong gt = g0;
                    term[0] = bb_mul(term[0], gt);
                    for (uint t = 1u; t <= d; ++t) {
                        gt = bb_add(gt, gd);
                        term[t] = bb_mul(term[t], gt);
                    }
                }
            }
        }
    }

    uint lane    = tid & 31u;
    uint warp_id = tid >> 5;
    uint stride  = d + 1u;

    // Stage 1: SIMD-reduce within each warp, write 8 warp-sums per t.
    if (prime_kind == 0u) {
        for (uint t = 0u; t <= d; ++t) {
            ulong v = simd_reduce_add_gold(term[t]);
            if (lane == 0u) scratch[warp_id * stride + t] = v;
        }
    } else {
        for (uint t = 0u; t <= d; ++t) {
            ulong v = simd_reduce_add_bb(term[t]);
            if (lane == 0u) scratch[warp_id * stride + t] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: warp 0 reduces 8 warp-sums per t (serial sum of 8 is cheap).
    if (warp_id == 0u) {
        if (lane <= d) {
            uint t = lane;
            ulong acc = scratch[t];
            if (prime_kind == 0u) {
                for (uint w = 1u; w < 8u; ++w) {
                    acc = gold_add(acc, scratch[w * stride + t]);
                }
            } else {
                for (uint w = 1u; w < 8u; ++w) {
                    acc = bb_add(acc, scratch[w * stride + t]);
                }
            }
            partial[tgid * stride + t] = acc;
        }
    }
}

// ----------------------------------------------------------------------
// Kernel B
// ----------------------------------------------------------------------

kernel void sumcheck_fold(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *f_out      [[buffer(1)]],
    constant ulong     &r          [[buffer(2)]],
    constant uint      &k_log      [[buffer(3)]],
    constant uint      &d_deg      [[buffer(4)]],
    constant uint      &prime_kind [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    uint total = d * half_n;
    if (gid >= total) return;

    uint poly_i = gid / half_n;
    uint j      = gid - poly_i * half_n;

    ulong f0 = f_in[poly_i * base + j];
    ulong f1 = f_in[poly_i * base + j + half_n];

    if (prime_kind == 0u) {
        ulong delta = gold_sub(f1, f0);
        ulong fold  = gold_add(f0, gold_mul(r, delta));
        f_out[poly_i * half_n + j] = fold;
    } else {
        ulong delta = bb_sub(f1, f0);
        ulong fold  = bb_add(f0, bb_mul(r, delta));
        f_out[poly_i * half_n + j] = fold;
    }
}