The previous attempt failed because `simd_shuffle_xor` does not support `ulong` on Apple GPUs. I'll fix this by splitting each `ulong` into two `uint` halves and shuffling them separately (a known Apple GPU workaround). 

**Optimizations vs incumbent:**
1. **SIMD-level reduction via uint-pair shuffles**: Avoid the expensive 256-wide threadgroup-memory tree reduction. Reduce 32 lanes within a simdgroup using `simd_shuffle_xor` on `uint` halves, then do a tiny 8-way reduction across warps via threadgroup memory.
2. **Incremental f_i(t) computation**: Replace `mod_mul((ulong)t, delta, …)` per t with `ft += delta` recurrence (saves one mul per (i,t)).
3. **Lift `prime_kind` into a compile-time branch** *at runtime* via an early `if` so the hot inner loop avoids per-op ternary overhead — keeping runtime dispatch but reducing branch frequency.

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