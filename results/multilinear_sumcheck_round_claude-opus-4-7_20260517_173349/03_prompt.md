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

// ---------------------- Generic dispatch ------------------------------

inline ulong mod_add(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_add(a, b) : bb_add(a, b);
}

inline ulong mod_sub(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

inline ulong mod_mul(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

// ---- ulong simd shuffle via uint halves (Apple GPU workaround) -------

inline ulong simd_shfl_xor_ulong(ulong v, ushort mask) {
    uint lo = (uint)(v);
    uint hi = (uint)(v >> 32);
    lo = simd_shuffle_xor(lo, mask);
    hi = simd_shuffle_xor(hi, mask);
    return ((ulong)hi << 32) | (ulong)lo;
}

// ----------------------------------------------------------------------
// Kernel A: per-tile partial sums of h(t) for t = 0..d.
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
    // 8 warps * (d+1 <= MAX_D+1) lanes.
    threadgroup ulong scratch[8u * (MAX_D + 1u)];

    uint d = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base = 1u << k_log;

    ulong term[MAX_D + 1u];
    for (uint t = 0u; t <= MAX_D; ++t) term[t] = 0ul;

    if (gid < half_n) {
        ulong prod_t[MAX_D + 1u];
        for (uint t = 0u; t <= d; ++t) prod_t[t] = 1ul;

        if (prime_kind == 0u) {
            // Goldilocks fast path
            for (uint i = 0u; i < d; ++i) {
                ulong f0 = f_in[i * base + gid];
                ulong f1 = f_in[i * base + gid + half_n];
                ulong delta = gold_sub(f1, f0);
                ulong ft = f0;
                prod_t[0] = gold_mul(prod_t[0], ft);
                for (uint t = 1u; t <= d; ++t) {
                    ft = gold_add(ft, delta);
                    prod_t[t] = gold_mul(prod_t[t], ft);
                }
            }
        } else {
            // BabyBear fast path
            for (uint i = 0u; i < d; ++i) {
                ulong f0 = f_in[i * base + gid];
                ulong f1 = f_in[i * base + gid + half_n];
                ulong delta = bb_sub(f1, f0);
                ulong ft = f0;
                prod_t[0] = bb_mul(prod_t[0], ft);
                for (uint t = 1u; t <= d; ++t) {
                    ft = bb_add(ft, delta);
                    prod_t[t] = bb_mul(prod_t[t], ft);
                }
            }
        }
        for (uint t = 0u; t <= d; ++t) term[t] = prod_t[t];
    }

    uint lane    = tid & 31u;
    uint warp_id = tid >> 5;

    // SIMD reduce 32 lanes via uint-pair shuffles.
    for (uint t = 0u; t <= d; ++t) {
        ulong v = term[t];
        for (ushort off = 16; off > 0; off >>= 1) {
            ulong other = simd_shfl_xor_ulong(v, off);
            v = mod_add(v, other, prime_kind);
        }
        if (lane == 0u) {
            scratch[warp_id * (d + 1u) + t] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp reduces the 8 warp sums.
    if (warp_id == 0u) {
        for (uint t = 0u; t <= d; ++t) {
            ulong v = (lane < 8u) ? scratch[lane * (d + 1u) + t] : 0ul;
            for (ushort off = 4; off > 0; off >>= 1) {
                ulong other = simd_shfl_xor_ulong(v, off);
                v = mod_add(v, other, prime_kind);
            }
            if (lane == 0u) {
                partial[tgid * (d + 1u) + t] = v;
            }
        }
    }
}

// ----------------------------------------------------------------------
// Kernel B: fold each factor table along the first variable.
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

    ulong f0    = f_in[poly_i * base + j];
    ulong f1    = f_in[poly_i * base + j + half_n];
    ulong delta = mod_sub(f1, f0, prime_kind);
    ulong fold  = mod_add(f0,
                          mod_mul(r, delta, prime_kind),
                          prime_kind);
    f_out[poly_i * half_n + j] = fold;
}
```

Result of previous attempt:
       gold_k14_d2: correct, 0.06 ms, 7.0 GB/s (3.5% of 200 GB/s)
       gold_k16_d2: correct, 0.22 ms, 7.1 GB/s (3.5% of 200 GB/s)
       gold_k18_d2: correct, 0.60 ms, 10.6 GB/s (5.3% of 200 GB/s)
  score (gmean of fraction): 0.0402

## History

- iter  0: compile=OK | correct=True | score=0.015217166986741219
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.04015551516102348

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
