I'll acknowledge the system reminder about the keybindings-help skill (not relevant here) and proceed with the optimization task.

**Optimization strategy:** The previous version did per-pair Goldilocks reductions and a heavy per-lane shuffle reduction with `mod_add` dispatch. Key improvements:

1. **Accumulate per-pair products in 128-bit lazily for Goldilocks**: keep running prod via fast mul+reduce, but eliminate the per-lane prime_kind branch by hoisting the dispatch outside SIMD reductions.
2. **Bigger per-thread workload**: each thread processes a *strided block* of pair indices (TG covers 256 pairs, but threads may handle multiple `gid`-mapped pairs only if half > 256 -- contract forbids; so we just amortize differently).
3. **Faster SIMD reduce**: do reductions in two halves (lo/hi uints) once, reducing shuffle count and using `simd_shuffle_xor` directly on ulong via uint pair only once per step. Specialize the SIMD reduction by prime_kind (branch is uniform across the threadgroup).
4. **Memory coalescing**: read `f0` and `f1` from contiguous slabs (already coalesced).
5. **Reduce scratch pressure**: store reduced warp sums per t into shared memory in a single combined pass.
6. **BabyBear fast path**: BabyBear fits in 31 bits; products fit in 62 bits, so we can defer mod and use 64-bit accumulation more aggressively. Also `a*b % P_BB` is slow — use Barrett/Montgomery? Simpler: stay with `%` but only when needed.

The biggest win actually comes from **reducing the SIMD shuffles**: doing 5 levels × (d+1) shuffles of ulong = 10×(d+1) uint shuffles. We instead reduce per-warp via threadgroup memory after a single warp-reduction, and use `simd_sum`-style butterflies. Also key: split the SIMD reduce to do all t's lo halves then hi halves to better pipeline.

```metal
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

// ---- ulong simd shuffle via uint halves -----------------------------

inline ulong simd_shfl_xor_ulong(ulong v, ushort mask) {
    uint lo = (uint)(v);
    uint hi = (uint)(v >> 32);
    lo = simd_shuffle_xor(lo, mask);
    hi = simd_shuffle_xor(hi, mask);
    return ((ulong)hi << 32) | (ulong)lo;
}

// SIMD reduce-add specialised by prime_kind (uniform branch)
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
    // 8 warps × (d+1) lanes for cross-warp reduction.
    threadgroup ulong scratch[8u * (MAX_D + 1u)];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    ulong term[MAX_D + 1u];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

    bool active = (gid < half_n);

    if (prime_kind == 0u) {
        if (active) {
            // First factor initialises prod_t directly (skip the mul-by-1).
            ulong f0 = f_in[gid];
            ulong f1 = f_in[gid + half_n];
            ulong delta = gold_sub(f1, f0);
            ulong ft = f0;
            term[0] = ft;
            for (uint t = 1u; t <= d; ++t) {
                ft = gold_add(ft, delta);
                term[t] = ft;
            }
            // Remaining d-1 factors.
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

    uint lane    = tid & 31u;
    uint warp_id = tid >> 5;
    uint stride  = d + 1u;

    // SIMD-reduce within each warp, store warp sums in scratch.
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

    // Final reduction: 8 warp sums per t, parallelised across warp 0's 32 lanes.
    // We dedicate (d+1) groups of lanes; each group of 8 lanes reduces one t.
    if (warp_id == 0u) {
        // Map: lane in [0,32). Pack up to 4 t's into 32 lanes (8 lanes per t).
        // Simpler: have lane t handle reduction for t (serial sum of 8).
        if (lane <= d) {
            uint t = lane;
            ulong acc = scratch[0 * stride + t];
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
```