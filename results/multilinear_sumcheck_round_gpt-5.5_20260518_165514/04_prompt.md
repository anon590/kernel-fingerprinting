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

constant ulong P_GOLD   = 0xFFFFFFFF00000001ul;
constant ulong EPSILON  = 0x00000000FFFFFFFFul;
constant ulong P_BB     = 2013265921ul;
constant ulong BB_MU    = 9162596893ul;          // floor(2^64 / P_BB)
constant ulong BB_R64   = 1172168163ul;          // 2^64 mod P_BB

constant uint TG_WIDTH      = 256u;
constant uint SG_WIDTH      = 32u;
constant uint SG_COUNT      = 8u;
constant uint MAX_D         = 3u;
constant uint SCR_STRIDE    = MAX_D + 1u;
constant uint RAW_D2_STRIDE = 9u;

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
// SIMD shuffle / field reductions
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
// Raw wide-sum reductions for d=2 h(X)
// -----------------------------------------------------------------------------

struct Raw128 {
    ulong lo;
    ulong hi;
    uint  top;
};

struct Raw64 {
    ulong lo;
    uint  top;
};

inline Raw128 make_raw128(ulong lo, ulong hi, uint top) {
    Raw128 r;
    r.lo = lo;
    r.hi = hi;
    r.top = top;
    return r;
}

inline Raw64 make_raw64(ulong lo, uint top) {
    Raw64 r;
    r.lo = lo;
    r.top = top;
    return r;
}

inline Raw128 raw128_add(Raw128 a, Raw128 b) {
    ulong lo = a.lo + b.lo;
    uint c0 = (lo < a.lo) ? 1u : 0u;

    ulong hi0 = a.hi + b.hi;
    uint c1 = (hi0 < a.hi) ? 1u : 0u;

    ulong hi = hi0 + (ulong)c0;
    uint c2 = (hi < hi0) ? 1u : 0u;

    a.lo = lo;
    a.hi = hi;
    a.top = a.top + b.top + c1 + c2;
    return a;
}

inline Raw64 raw64_add(Raw64 a, Raw64 b) {
    ulong lo = a.lo + b.lo;
    uint c = (lo < a.lo) ? 1u : 0u;
    a.lo = lo;
    a.top = a.top + b.top + c;
    return a;
}

inline Raw128 raw128_shuffle_xor(Raw128 x, ushort mask) {
    Raw128 y;
    y.lo = shuffle_xor_ulong(x.lo, mask);
    y.hi = shuffle_xor_ulong(x.hi, mask);
    y.top = simd_shuffle_xor(x.top, mask);
    return y;
}

inline Raw64 raw64_shuffle_xor(Raw64 x, ushort mask) {
    Raw64 y;
    y.lo = shuffle_xor_ulong(x.lo, mask);
    y.top = simd_shuffle_xor(x.top, mask);
    return y;
}

inline Raw128 simd_sum_raw128(Raw128 x) {
    x = raw128_add(x, raw128_shuffle_xor(x, 16));
    x = raw128_add(x, raw128_shuffle_xor(x,  8));
    x = raw128_add(x, raw128_shuffle_xor(x,  4));
    x = raw128_add(x, raw128_shuffle_xor(x,  2));
    x = raw128_add(x, raw128_shuffle_xor(x,  1));
    return x;
}

inline Raw64 simd_sum_raw64(Raw64 x) {
    x = raw64_add(x, raw64_shuffle_xor(x, 16));
    x = raw64_add(x, raw64_shuffle_xor(x,  8));
    x = raw64_add(x, raw64_shuffle_xor(x,  4));
    x = raw64_add(x, raw64_shuffle_xor(x,  2));
    x = raw64_add(x, raw64_shuffle_xor(x,  1));
    return x;
}

inline ulong gold_reduce_raw128(Raw128 x) {
    ulong r = gold_reduce128(x.lo, x.hi);
    if (x.top != 0u) {
        r = gold_sub(r, ((ulong)x.top) << 32);
    }
    return r;
}

inline ulong bb_reduce_raw64(Raw64 x) {
    ulong r = bb_reduce_wide(x.lo);
    if (x.top != 0u) {
        ulong h = bb_reduce_wide((ulong)x.top * BB_R64);
        r = bb_add(r, h);
    }
    return r;
}

inline void reduce_store_gold_d2_raw(
    ulong l0, ulong h0,
    ulong l1, ulong h1,
    ulong l2, ulong h2,
    device ulong *partial,
    uint tid,
    uint tgid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    Raw128 r0 = simd_sum_raw128(make_raw128(l0, h0, 0u));
    Raw128 r1 = simd_sum_raw128(make_raw128(l1, h1, 0u));
    Raw128 r2 = simd_sum_raw128(make_raw128(l2, h2, 0u));

    if (lane == 0u) {
        uint o = sg * RAW_D2_STRIDE;

        scratch[o + 0u] = r0.lo;
        scratch[o + 1u] = r0.hi;
        scratch[o + 2u] = (ulong)r0.top;

        scratch[o + 3u] = r1.lo;
        scratch[o + 4u] = r1.hi;
        scratch[o + 5u] = (ulong)r1.top;

        scratch[o + 6u] = r2.lo;
        scratch[o + 7u] = r2.hi;
        scratch[o + 8u] = (ulong)r2.top;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 3u) {
        uint tt = tid * 3u;

        Raw128 acc = make_raw128(0ul, 0ul, 0u);

        uint o0 = 0u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o0 + 0u], scratch[o0 + 1u], (uint)scratch[o0 + 2u]));

        uint o1 = 1u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o1 + 0u], scratch[o1 + 1u], (uint)scratch[o1 + 2u]));

        uint o2 = 2u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o2 + 0u], scratch[o2 + 1u], (uint)scratch[o2 + 2u]));

        uint o3 = 3u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o3 + 0u], scratch[o3 + 1u], (uint)scratch[o3 + 2u]));

        uint o4 = 4u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o4 + 0u], scratch[o4 + 1u], (uint)scratch[o4 + 2u]));

        uint o5 = 5u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o5 + 0u], scratch[o5 + 1u], (uint)scratch[o5 + 2u]));

        uint o6 = 6u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o6 + 0u], scratch[o6 + 1u], (uint)scratch[o6 + 2u]));

        uint o7 = 7u * RAW_D2_STRIDE + tt;
        acc = raw128_add(acc, make_raw128(scratch[o7 + 0u], scratch[o7 + 1u], (uint)scratch[o7 + 2u]));

        partial[tgid * 3u + tid] = gold_reduce_raw128(acc);
    }
}

inline void reduce_store_bb_d2_raw(
    ulong v0, ulong v1, ulong v2,
    device ulong *partial,
    uint tid,
    uint tgid,
    threadgroup ulong *scratch)
{
    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    Raw64 r0 = simd_sum_raw64(make_raw64(v0, 0u));
    Raw64 r1 = simd_sum_raw64(make_raw64(v1, 0u));
    Raw64 r2 = simd_sum_raw64(make_raw64(v2, 0u));

    if (lane == 0u) {
        uint o = sg * 6u;

        scratch[o + 0u] = r0.lo;
        scratch[o + 1u] = (ulong)r0.top;

        scratch[o + 2u] = r1.lo;
        scratch[o + 3u] = (ulong)r1.top;

        scratch[o + 4u] = r2.lo;
        scratch[o + 5u] = (ulong)r2.top;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 3u) {
        uint tt = tid * 2u;

        Raw64 acc = make_raw64(0ul, 0u);

        uint o0 = 0u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o0 + 0u], (uint)scratch[o0 + 1u]));

        uint o1 = 1u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o1 + 0u], (uint)scratch[o1 + 1u]));

        uint o2 = 2u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o2 + 0u], (uint)scratch[o2 + 1u]));

        uint o3 = 3u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o3 + 0u], (uint)scratch[o3 + 1u]));

        uint o4 = 4u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o4 + 0u], (uint)scratch[o4 + 1u]));

        uint o5 = 5u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o5 + 0u], (uint)scratch[o5 + 1u]));

        uint o6 = 6u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o6 + 0u], (uint)scratch[o6 + 1u]));

        uint o7 = 7u * 6u + tt;
        acc = raw64_add(acc, make_raw64(scratch[o7 + 0u], (uint)scratch[o7 + 1u]));

        partial[tgid * 3u + tid] = bb_reduce_raw64(acc);
    }
}

// -----------------------------------------------------------------------------
// Threadgroup reduction helpers for d=1 / d=3 fast paths
// -----------------------------------------------------------------------------

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

// Generic correctness fallback for d outside the small fast paths.
inline void reduce_store_one_gold_loop(
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

inline void reduce_store_one_bb_loop(
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
    threadgroup ulong scratch[SG_COUNT * RAW_D2_STRIDE];

    uint d      = d_deg;
    uint half_n = 1u << (k_log - 1u);
    uint base   = half_n << 1;

    // Fast d=2 path: wide raw product accumulation, reduce once per tile/eval.
    if (d == 2u) {
        if (prime_kind == 0u) {
            ulong l0 = 0ul, h0 = 0ul;
            ulong l1 = 0ul, h1 = 0ul;
            ulong l2 = 0ul, h2 = 0ul;

            if (gid < half_n) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];

                ulong da = gold_sub(a1, a0);
                ulong db = gold_sub(b1, b0);
                ulong a2 = gold_add(a1, da);
                ulong b2 = gold_add(b1, db);

                ulong2 p0 = umul128_u32(a0, b0);
                ulong2 p1 = umul128_u32(a1, b1);
                ulong2 p2 = umul128_u32(a2, b2);

                l0 = p0.x; h0 = p0.y;
                l1 = p1.x; h1 = p1.y;
                l2 = p2.x; h2 = p2.y;
            }

            reduce_store_gold_d2_raw(l0, h0, l1, h1, l2, h2, partial, tid, tgid, scratch);
            return;
        } else {
            ulong s0 = 0ul;
            ulong s1 = 0ul;
            ulong s2 = 0ul;

            if (gid < half_n) {
                ulong a0 = f_in[gid];
                ulong a1 = f_in[gid + half_n];
                ulong b0 = f_in[base + gid];
                ulong b1 = f_in[base + gid + half_n];

                ulong da = bb_sub(a1, a0);
                ulong db = bb_sub(b1, b0);
                ulong a2 = bb_add(a1, da);
                ulong b2 = bb_add(b1, db);

                s0 = (ulong)((uint)a0) * (ulong)((uint)b0);
                s1 = (ulong)((uint)a1) * (ulong)((uint)b1);
                s2 = (ulong)((uint)a2) * (ulong)((uint)b2);
            }

            reduce_store_bb_d2_raw(s0, s1, s2, partial, tid, tgid, scratch);
            return;
        }
    }

    // Fast d=1 / d=3 paths.
    if (d == 1u || d == 3u) {
        ulong s0 = 0ul;
        ulong s1 = 0ul;
        ulong s2 = 0ul;
        ulong s3 = 0ul;

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

    // Generic fallback for other runtime d values.
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

            reduce_store_one_gold_loop(prod, partial, tgid * (d + 1u) + t, tid, scratch);
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

            reduce_store_one_bb_loop(prod, partial, tgid * (d + 1u) + t, tid, scratch);
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
    uint total  = d_deg * half_n;

    if (gid >= total) return;

    uint poly_i = gid >> shift;
    uint j      = gid & (half_n - 1u);
    uint off    = (poly_i << k_log) + j;

    ulong f0 = f_in[off];
    ulong f1 = f_in[off + half_n];

    if (prime_kind == 0u) {
        ulong delta = gold_sub(f1, f0);
        f_out[gid] = gold_mul_add(r, delta, f0);
    } else {
        ulong delta = bb_sub(f1, f0);
        f_out[gid] = bb_mul_add(r, delta, f0);
    }
}
```

Result of previous attempt:
       gold_k14_d2: correct, 0.04 ms, 9.4 GB/s (4.7% of 200 GB/s)
       gold_k16_d2: correct, 0.09 ms, 18.5 GB/s (9.2% of 200 GB/s)
       gold_k18_d2: correct, 0.26 ms, 24.2 GB/s (12.1% of 200 GB/s)
  score (gmean of fraction): 0.0807

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD   = 0xFFFFFFFF00000001ul;
constant ulong EPSILON  = 0x00000000FFFFFFFFul;
constant ulong P_BB     = 2013265921ul;
constant ulong BB_MU    = 9162596893ul;          // floor(2^64 / P_BB)

constant uint TG_WIDTH   = 256u;
constant uint SG_WIDTH   = 32u;
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
            if (gid < half_n) {
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
```

Incumbent result:
       gold_k14_d2: correct, 0.02 ms, 20.4 GB/s (10.2% of 200 GB/s)
       gold_k16_d2: correct, 0.05 ms, 28.7 GB/s (14.3% of 200 GB/s)
       gold_k18_d2: correct, 0.22 ms, 28.1 GB/s (14.1% of 200 GB/s)
  score (gmean of fraction): 0.1272

## History

- iter  0: compile=OK | correct=True | score=0.012594084791284247
- iter  1: compile=OK | correct=True | score=0.08690235591881525
- iter  2: compile=OK | correct=True | score=0.12720307086961388
- iter  3: compile=OK | correct=True | score=0.080718344118207

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
