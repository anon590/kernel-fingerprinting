**Optimization:** The benchmark is `d=2` Goldilocks-dominated. Currently each thread handles 1 pair index, doing 3 mulhi-style Goldilocks multiplies plus 5 simd_reduce passes per t. The bottleneck is the reduction overhead (5 shuffles × 3 ts = 15 shuffle rounds, plus per-warp scratch). I'll restructure so each thread still owns ONE pair index (contract), but the per-t simd reductions are interleaved so the compiler can hide latency, AND I fold the t=0,1,2 reductions into a single loop where each lane carries a `ulong3` and we shuffle the triple together in lockstep — cutting shuffle instructions per pair from 30 (3×5×2 uint shuffles) by overlapping. I also specialize the d=2 path to avoid the generic interpolation overhead and add an `if (gid >= half_n)` early-skip after zero-init so inactive threads don't do reads. Memory: Goldilocks input dominates; reads stay coalesced.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;

constant uint MAX_D    = 8u;

// ---------------------- Goldilocks helpers ----------------------------

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong carry = (t < a) ? EPSILON : 0ul;
    t += carry;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    ulong borrow = (t > a) ? EPSILON : 0ul;
    return t - borrow;
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (x_lo < x_hi_hi) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
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
    return gold_reduce128(lo, hi);
}

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

// Reduce three Goldilocks values in parallel within a simdgroup;
// shares the simd_shuffle_xor cost across the three accumulators
// by interleaving them so the compiler can issue back-to-back shuffles.
inline void simd_reduce3_gold(thread ulong &v0, thread ulong &v1, thread ulong &v2) {
    ulong o0, o1, o2;
    o0 = simd_shfl_xor_ulong(v0, 16); o1 = simd_shfl_xor_ulong(v1, 16); o2 = simd_shfl_xor_ulong(v2, 16);
    v0 = gold_add(v0, o0); v1 = gold_add(v1, o1); v2 = gold_add(v2, o2);
    o0 = simd_shfl_xor_ulong(v0,  8); o1 = simd_shfl_xor_ulong(v1,  8); o2 = simd_shfl_xor_ulong(v2,  8);
    v0 = gold_add(v0, o0); v1 = gold_add(v1, o1); v2 = gold_add(v2, o2);
    o0 = simd_shfl_xor_ulong(v0,  4); o1 = simd_shfl_xor_ulong(v1,  4); o2 = simd_shfl_xor_ulong(v2,  4);
    v0 = gold_add(v0, o0); v1 = gold_add(v1, o1); v2 = gold_add(v2, o2);
    o0 = simd_shfl_xor_ulong(v0,  2); o1 = simd_shfl_xor_ulong(v1,  2); o2 = simd_shfl_xor_ulong(v2,  2);
    v0 = gold_add(v0, o0); v1 = gold_add(v1, o1); v2 = gold_add(v2, o2);
    o0 = simd_shfl_xor_ulong(v0,  1); o1 = simd_shfl_xor_ulong(v1,  1); o2 = simd_shfl_xor_ulong(v2,  1);
    v0 = gold_add(v0, o0); v1 = gold_add(v1, o1); v2 = gold_add(v2, o2);
}

inline ulong simd_reduce_add_gold(ulong v) {
    ulong o;
    o = simd_shfl_xor_ulong(v, 16); v = gold_add(v, o);
    o = simd_shfl_xor_ulong(v,  8); v = gold_add(v, o);
    o = simd_shfl_xor_ulong(v,  4); v = gold_add(v, o);
    o = simd_shfl_xor_ulong(v,  2); v = gold_add(v, o);
    o = simd_shfl_xor_ulong(v,  1); v = gold_add(v, o);
    return v;
}

inline ulong simd_reduce_add_bb(ulong v) {
    ulong o;
    o = simd_shfl_xor_ulong(v, 16); v = bb_add(v, o);
    o = simd_shfl_xor_ulong(v,  8); v = bb_add(v, o);
    o = simd_shfl_xor_ulong(v,  4); v = bb_add(v, o);
    o = simd_shfl_xor_ulong(v,  2); v = bb_add(v, o);
    o = simd_shfl_xor_ulong(v,  1); v = bb_add(v, o);
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
    // 8 warps * (d+1)  -- only used when d != 2
    threadgroup ulong scratch[8u * (MAX_D + 1u)];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    uint lane    = tid & 31u;
    uint warp_id = tid >> 5;
    uint stride  = d + 1u;

    bool active = (gid < half_n);

    if (d == 2u) {
        // Fast path: d=2, three round-poly evaluations.
        ulong t0 = 0ul, t1 = 0ul, t2 = 0ul;

        if (active) {
            if (prime_kind == 0u) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];
                ulong a2 = gold_sub(gold_dbl(a1), a0);
                ulong b2 = gold_sub(gold_dbl(b1), b0);
                t0 = gold_mul(a0, b0);
                t1 = gold_mul(a1, b1);
                t2 = gold_mul(a2, b2);
            } else {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];
                ulong a2 = bb_sub(bb_dbl(a1), a0);
                ulong b2 = bb_sub(bb_dbl(b1), b0);
                t0 = bb_mul(a0, b0);
                t1 = bb_mul(a1, b1);
                t2 = bb_mul(a2, b2);
            }
        }

        // Interleaved 3-way simd reduce within the warp.
        if (prime_kind == 0u) {
            simd_reduce3_gold(t0, t1, t2);
        } else {
            // BB: cheaper add, just do 3 separate reductions
            t0 = simd_reduce_add_bb(t0);
            t1 = simd_reduce_add_bb(t1);
            t2 = simd_reduce_add_bb(t2);
        }

        // Stage 1: lane 0 of each warp writes the three warp sums.
        // Layout: scratch[warp_id*3 + t]
        if (lane == 0u) {
            scratch[warp_id * 3u + 0u] = t0;
            scratch[warp_id * 3u + 1u] = t1;
            scratch[warp_id * 3u + 2u] = t2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2: warp 0 lanes 0..2 finalize and write.
        if (warp_id == 0u && lane < 3u) {
            uint t = lane;
            ulong acc = scratch[t];
            if (prime_kind == 0u) {
                acc = gold_add(acc, scratch[1u * 3u + t]);
                acc = gold_add(acc, scratch[2u * 3u + t]);
                acc = gold_add(acc, scratch[3u * 3u + t]);
                acc = gold_add(acc, scratch[4u * 3u + t]);
                acc = gold_add(acc, scratch[5u * 3u + t]);
                acc = gold_add(acc, scratch[6u * 3u + t]);
                acc = gold_add(acc, scratch[7u * 3u + t]);
            } else {
                acc = bb_add(acc, scratch[1u * 3u + t]);
                acc = bb_add(acc, scratch[2u * 3u + t]);
                acc = bb_add(acc, scratch[3u * 3u + t]);
                acc = bb_add(acc, scratch[4u * 3u + t]);
                acc = bb_add(acc, scratch[5u * 3u + t]);
                acc = bb_add(acc, scratch[6u * 3u + t]);
                acc = bb_add(acc, scratch[7u * 3u + t]);
            }
            partial[tgid * 3u + t] = acc;
        }
        return;
    }

    // ---------------- General path: arbitrary d ----------------
    ulong term[MAX_D + 1u];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

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

    if (warp_id == 0u && lane <= d) {
        uint t = lane;
        ulong acc = scratch[t];
        if (prime_kind == 0u) {
            acc = gold_add(acc, scratch[1u * stride + t]);
            acc = gold_add(acc, scratch[2u * stride + t]);
            acc = gold_add(acc, scratch[3u * stride + t]);
            acc = gold_add(acc, scratch[4u * stride + t]);
            acc = gold_add(acc, scratch[5u * stride + t]);
            acc = gold_add(acc, scratch[6u * stride + t]);
            acc = gold_add(acc, scratch[7u * stride + t]);
        } else {
            acc = bb_add(acc, scratch[1u * stride + t]);
            acc = bb_add(acc, scratch[2u * stride + t]);
            acc = bb_add(acc, scratch[3u * stride + t]);
            acc = bb_add(acc, scratch[4u * stride + t]);
            acc = bb_add(acc, scratch[5u * stride + t]);
            acc = bb_add(acc, scratch[6u * stride + t]);
            acc = bb_add(acc, scratch[7u * stride + t]);
        }
        partial[tgid * stride + t] = acc;
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
```