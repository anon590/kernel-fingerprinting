## Task: multilinear_sumcheck_round

One degree-d sumcheck round on a product polynomial g(x) = f_0(x) * f_1(x) * ... * f_{d-1}(x), where each f_i: {0,1}^k -> F_p is multilinear, stored as a length 2^k_log table of evaluations on the Boolean hypercube. The kernel folds the FIRST variable: it emits (A) the univariate round polynomial h(X) = sum_{x' in {0,1}^(k-1)} prod_i f_i(X, x'), represented by its d+1 evaluations h(0), h(1), ..., h(d); and (B) the folded factor tables f_i_new[j] = f_i(r, j) for j in [0, 2^(k-1)), where r is the verifier-supplied round challenge in [0, p).

Layout convention. The variable being folded is the most significant bit of the hypercube index, so for j in [0, 2^(k-1)) the X = 0 and X = 1 slices are
  f_i^(0)[j] = f_in[i * 2^k_log + j]
  f_i^(1)[j] = f_in[i * 2^k_log + j + 2^(k-1)]
The multilinear extension along the first variable, evaluated at any X in F_p, is the unique affine interpolant
  f_i(X, j) = f_i^(0)[j] + X * (f_i^(1)[j] - f_i^(0)[j])   (mod p)
so the kernel must produce, in one round,
  h(t)       = sum_{j in [0, 2^(k-1))} prod_i f_i(t, j)
               for t in {0, 1, ..., d}
  f_i_new[j] = f_i(r, j)
               for i in [0, d) and j in [0, 2^(k-1)).

Two-kernel pipeline (host issues both in ONE compute command encoder; the serial encoder gives an implicit barrier so kernel B observes A's writes):
  Dispatch 1 (sumcheck_round_h): each threadgroup owns     256 consecutive pair indices in [0, half) where     half = 2^(k_log - 1). For each pair index j the     thread contributes the d+1 per-pair products     prod_i f_i(t, j); the threadgroup cooperatively     reduces 256 contributions per t into one tile sum     and writes d+1 contiguous ulongs to     partial[tgid * (d+1) + t]. Threads with gid >=     half contribute 0 (additive identity for the sum).
  Dispatch 2 (sumcheck_fold): one thread per output     (poly_i, j); writes one folded coefficient to     f_out[poly_i * half + j]. Guard against gid >= d *     half (the grid is rounded up to a multiple of the TG     width).

The host then sums partial[0..K-1] per t on the CPU (K = ceil(half / 256), ~1 KB total -- intentionally untimed) to obtain h_evals[0..d+1], and cross-checks the sumcheck consistency identity h(0) + h(1) == sum_x prod_i f_i(x). A candidate whose h_evals matches a same-buggy reference but indexes the linear extension the wrong way silently fails this identity.

Field selection (constant prime_kind):
  0 = Goldilocks   p = 2^64 - 2^32 + 1
  1 = BabyBear     p = 2^31 - 2^27 + 1 = 2013265921
Both reductions, the per-pair t-loop, and the threadgroup geometry must dispatch on the RUNTIME values of prime_kind, d_deg, and k_log. Baking any of them in as a compile-time constant -- a specific reduction macro, a fixed unroll over t, a hardcoded buffer stride, ... -- violates the kernel contract.

All field elements (f_in, partial, f_out, r) are canonical uint64 in [0, p); a non-canonical output is treated as a correctness failure even if its residue class matches the reference.

## Required kernel signature(s)

```
kernel void sumcheck_round_h(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *partial    [[buffer(1)]],
    constant uint      &k_log      [[buffer(2)]],
    constant uint      &d_deg      [[buffer(3)]],
    constant uint      &prime_kind [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]);

kernel void sumcheck_fold(
    device const ulong *f_in       [[buffer(0)]],
    device       ulong *f_out      [[buffer(1)]],
    constant ulong     &r          [[buffer(2)]],
    constant uint      &k_log      [[buffer(3)]],
    constant uint      &d_deg      [[buffer(4)]],
    constant uint      &prime_kind [[buffer(5)]],
    uint gid [[thread_position_in_grid]]);

Dispatch geometry (host-fixed):
  sumcheck_round_h:
    threadsPerGrid        = (K * 256, 1, 1)   K = ceil(half / 256)
    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256
  sumcheck_fold:
    threadsPerGrid        = (d * half rounded up to TG width, 1, 1)
    threadsPerThreadgroup = (min(d * half, 256), 1, 1)

The 256-wide threadgroup is part of the host-kernel contract for sumcheck_round_h: K = ceil(half / 256) is baked into the host-side partial[] allocation, so the kernel must emit exactly one (d+1)-element tile sum per 256 consecutive pair indices.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;

constant uint TG_WIDTH = 256u;
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

// ---- ulong simd shuffle via uint2 -----------------------------------

inline ulong simd_shfl_xor_ulong(ulong v, ushort mask) {
    uint2 pair = uint2((uint)v, (uint)(v >> 32));
    pair = simd_shuffle_xor(pair, mask);
    return ((ulong)pair.y << 32) | (ulong)pair.x;
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
    // 8 warps, one slot each, reused per t.
    threadgroup ulong warp_sums[8];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    ulong term[MAX_D + 1u];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

    bool active = (gid < half_n);

    if (d == 2u) {
        if (active) {
            if (prime_kind == 0u) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];
                term[0] = gold_mul(a0, b0);
                term[1] = gold_mul(a1, b1);
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

    // For each t: simd-reduce within warp, write warp sum, barrier,
    // warp 0 lane 0..7 reads 8 warp sums and finalizes.
    // Single threadgroup slot reused per t.
    for (uint t = 0u; t <= d; ++t) {
        ulong v;
        if (prime_kind == 0u) {
            v = simd_reduce_add_gold(term[t]);
        } else {
            v = simd_reduce_add_bb(term[t]);
        }
        if (lane == 0u) warp_sums[warp_id] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp_id == 0u && lane == 0u) {
            ulong acc = warp_sums[0];
            if (prime_kind == 0u) {
                acc = gold_add(acc, warp_sums[1]);
                acc = gold_add(acc, warp_sums[2]);
                acc = gold_add(acc, warp_sums[3]);
                acc = gold_add(acc, warp_sums[4]);
                acc = gold_add(acc, warp_sums[5]);
                acc = gold_add(acc, warp_sums[6]);
                acc = gold_add(acc, warp_sums[7]);
            } else {
                acc = bb_add(acc, warp_sums[1]);
                acc = bb_add(acc, warp_sums[2]);
                acc = bb_add(acc, warp_sums[3]);
                acc = bb_add(acc, warp_sums[4]);
                acc = bb_add(acc, warp_sums[5]);
                acc = bb_add(acc, warp_sums[6]);
                acc = bb_add(acc, warp_sums[7]);
            }
            partial[tgid * stride + t] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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

Result of previous attempt:
       gold_k14_d2: correct, 0.04 ms, 9.3 GB/s (4.6% of 200 GB/s)
       gold_k16_d2: correct, 0.09 ms, 17.6 GB/s (8.8% of 200 GB/s)
       gold_k18_d2: correct, 0.26 ms, 24.5 GB/s (12.2% of 200 GB/s)
  score (gmean of fraction): 0.0794

## Current best (incumbent)

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
    // x = x_lo + x_hi * 2^64
    // 2^64 = 2^32 - 1 (mod p)  =>  x_hi * 2^64 = x_hi * 2^32 - x_hi
    // Split x_hi = x_hi_hi * 2^32 + x_hi_lo
    // x_hi * 2^32 = x_hi_lo * 2^32 + x_hi_hi * 2^64
    //             = x_hi_lo * 2^32 + x_hi_hi * (2^32 - 1)
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (x_lo < x_hi_hi) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON; // (2^32 - 1) * x_hi_lo (no overflow)
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
    // scratch: 8 warps * (d+1) entries
    threadgroup ulong scratch[8u * (MAX_D + 1u)];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    ulong term[MAX_D + 1u];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

    bool active = (gid < half_n);

    if (d == 2u) {
        if (active) {
            if (prime_kind == 0u) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];
                term[0] = gold_mul(a0, b0);
                term[1] = gold_mul(a1, b1);
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

    // Stage 1: SIMD-reduce within each warp; lane 0 writes per-t warp sum.
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

    // Stage 2: warp 0 finishes — each of (d+1) lanes owns one t.
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

Incumbent result:
       gold_k14_d2: correct, 0.03 ms, 15.6 GB/s (7.8% of 200 GB/s)
       gold_k16_d2: correct, 0.05 ms, 28.9 GB/s (14.5% of 200 GB/s)
       gold_k18_d2: correct, 0.19 ms, 33.7 GB/s (16.8% of 200 GB/s)
  score (gmean of fraction): 0.1239

## History

- iter  0: compile=OK | correct=True | score=0.015217166986741219
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.04015551516102348
- iter  3: compile=OK | correct=True | score=0.07473168161436831
- iter  4: compile=OK | correct=True | score=0.08405390012410388
- iter  5: compile=OK | correct=True | score=0.12392677434660494
- iter  6: compile=OK | correct=True | score=0.07943028526611749

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
