## Task: kyber_ntt

Batched forward Cooley-Tukey NTT over a small prime field for a negacyclic polynomial ring Z_q[X] / (X^n + 1). Each polynomial has length n; the modulus q, the polynomial length n, the number of NTT stages n_levels, and the precomputed twiddle table are all bound at runtime so a single kernel runs every parameter set the host supplies.

Convention (matches the FIPS 203 / FIPS 204 / pqclean reference C implementations):
  k = 1
  for level = 0..n_levels:
      len = n >> (level + 1)
      for start = 0, 2*len, ..., n - 2*len:
          z = zetas[k++]
          for j = start..start + len - 1:
              t          = (z * a[j+len]) mod q
              a[j+len]   = (a[j] - t)       mod q
              a[j]       = (a[j] + t)       mod q
Equivalent per-thread mapping (ltid in [0, n/2) owns one butterfly per level):
  group_idx   = ltid / len
  j_in_group  = ltid - group_idx * len   (= ltid mod len)
  j           = (group_idx << 1) * len + j_in_group
  zeta_index  = (1 << level) + group_idx

Zetas table (host-precomputed, length 1 << n_levels):
  zetas[k] = zeta^bit_reverse(k, n_levels)  mod q
where zeta is a primitive 2^(n_levels+1)-th root of unity in F_q. The forward NTT consumes zetas[1..(1 << n_levels) - 1] in increasing index order; zetas[0] = 1 is the unread identity element.

Bounds for kernel design: q fits in a 32-bit unsigned integer; n is a power of two with n <= 256; n_levels <= 8 (so the zetas table has at most 256 entries). The kernel MUST read q, n, and n_levels from their bound buffers and load every twiddle from the zetas buffer at runtime; compile-time constants for any of these values are incorrect.

Storage: ``uint32`` per coefficient, in-place. The host writes the input coefficients into ``coeffs`` and reads the result back from the same buffer; ``coeffs`` is (batch * n) uint values in row-major order (polynomial p's coefficients live at offsets p*n .. p*n + n - 1).

All inputs are canonical: a[i] in [0, q). Outputs MUST also be canonical -- a value in [q, 2^32) with the same residue class still counts as a mismatch on the bit-exact reference comparison.

## Required kernel signature(s)

```
kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]]);

Dispatch (host-provided):
  threadsPerGrid        = (batch * (n/2), 1, 1)
  threadsPerThreadgroup = (n/2, 1, 1)
Each threadgroup owns ONE polynomial; tgid in [0, batch) selects the polynomial, ltid in [0, n/2) owns one butterfly per level. Every test size uses n = 256, so n/2 = 128 threads per threadgroup is sufficient; a static threadgroup scratch of size 256 covers every case. Threadgroup-cooperative and simdgroup schemes are valid as long as the buffer layout and the canonical-output contract are preserved.
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_reduce_generic(ulong t, uint q) {
    return (uint)(t % (ulong)q);
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return mod_reduce_generic((ulong)a * (ulong)b, q);
}

inline uint mod_add_generic(uint a, uint b, uint q) {
    uint s = a + b;
    return ((s < a) || (s >= q)) ? (s - q) : s;
}

inline uint mod_sub_generic(uint a, uint b, uint q) {
    uint d = a - b;
    return (a >= b) ? d : (d + q);
}

inline uint mod_mul_3329(uint a, uint b) {
    uint v    = a * b;
    uint qhat = (v * 315u) >> 20;
    uint r    = v - qhat * 3329u;
    return (r >= 3329u) ? (r + 3329u) : r;
}

inline uint mod_add_3329(uint a, uint b) {
    uint s = a + b;
    uint r = s - 3329u;
    return (s >= 3329u) ? r : s;
}

inline uint mod_sub_3329(uint a, uint b) {
    uint d = a - b;
    return (a >= b) ? d : (d + 3329u);
}

inline void bfly_tg_generic(threadgroup uint *s, uint j, uint len, uint zeta, uint q) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_generic(zeta, y, q);
    s[j]       = mod_add_generic(x, t, q);
    s[j + len] = mod_sub_generic(x, t, q);
}

inline void finish_kyber_3329_from_stage2(
    device       uint *poly,
    device const uint *zetas,
    uint ltid,
    uint s2lo,
    uint s2hi)
{
    uint lane  = ltid & 31u;
    uint r     = lane & 15u;
    uint chunk = ltid >> 4u;
    uint base  = chunk << 5u;

    uint2 got = simd_shuffle_xor(uint2(s2lo, s2hi), (ushort)16);
    uint upper = lane >> 4u;
    uint x = (upper != 0u) ? got.y : s2lo;
    uint y = (upper != 0u) ? s2hi : got.x;
    uint t = mod_mul_3329(zetas[8u + chunk], y);
    uint v0 = mod_add_3329(x, t);
    uint v1 = mod_sub_3329(x, t);

    got = simd_shuffle_xor(uint2(v0, v1), (ushort)8);
    upper = r >> 3u;
    x = (upper != 0u) ? got.y : v0;
    y = (upper != 0u) ? v1 : got.x;
    t = mod_mul_3329(zetas[16u + (chunk << 1u) + upper], y);
    v0 = mod_add_3329(x, t);
    v1 = mod_sub_3329(x, t);

    got = simd_shuffle_xor(uint2(v0, v1), (ushort)4);
    upper = (r >> 2u) & 1u;
    x = (upper != 0u) ? got.y : v0;
    y = (upper != 0u) ? v1 : got.x;
    t = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], y);
    v0 = mod_add_3329(x, t);
    v1 = mod_sub_3329(x, t);

    got = simd_shuffle_xor(uint2(v0, v1), (ushort)2);
    upper = (r >> 1u) & 1u;
    x = (upper != 0u) ? got.y : v0;
    y = (upper != 0u) ? v1 : got.x;
    t = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], y);

    uint p = ((r >> 1u) << 2u) | (r & 1u);
    poly[base + p]      = mod_add_3329(x, t);
    poly[base + p + 2u] = mod_sub_3329(x, t);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        if (batch < 16u) {
            uint x0 = poly[ltid];
            uint y0 = poly[ltid + 128u];
            uint m0 = mod_mul_3329(zetas[1u], y0);

            a[ltid]        = mod_add_3329(x0, m0);
            a[ltid + 128u] = mod_sub_3329(x0, m0);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint group32 = ltid >> 5u;
            uint r32     = ltid & 31u;
            uint half128 = group32 >> 1u;
            uint upper64 = group32 & 1u;
            uint base128 = half128 << 7u;
            uint z1      = zetas[2u + half128];

            uint u0 = a[base128 + r32];
            uint v0 = a[base128 + r32 + 64u];
            uint t0 = mod_mul_3329(z1, v0);
            uint sx = (upper64 != 0u) ? mod_sub_3329(u0, t0)
                                       : mod_add_3329(u0, t0);

            uint u1 = a[base128 + r32 + 32u];
            uint v1 = a[base128 + r32 + 96u];
            uint t1 = mod_mul_3329(z1, v1);
            uint sy = (upper64 != 0u) ? mod_sub_3329(u1, t1)
                                       : mod_add_3329(u1, t1);

            uint t2   = mod_mul_3329(zetas[4u + group32], sy);
            uint s2lo = mod_add_3329(sx, t2);
            uint s2hi = mod_sub_3329(sx, t2);

            finish_kyber_3329_from_stage2(poly, zetas, ltid, s2lo, s2hi);
            return;
        }

        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(zetas[1u], y);
        uint lo = mod_add_3329(x, t);
        uint hi = mod_sub_3329(x, t);

        uint upper64 = ltid >> 6u;
        a[ltid] = (upper64 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ex = a[ltid ^ 64u];
        x = (upper64 != 0u) ? ex : lo;
        y = (upper64 != 0u) ? hi : ex;
        t = mod_mul_3329(zetas[2u + upper64], y);
        lo = mod_add_3329(x, t);
        hi = mod_sub_3329(x, t);

        uint upper32 = (ltid >> 5u) & 1u;
        a[128u + ltid] = (upper32 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ex = a[128u + (ltid ^ 32u)];
        x = (upper32 != 0u) ? ex : lo;
        y = (upper32 != 0u) ? hi : ex;

        uint group32 = ltid >> 5u;
        t = mod_mul_3329(zetas[4u + group32], y);
        uint s2lo = mod_add_3329(x, t);
        uint s2hi = mod_sub_3329(x, t);

        finish_kyber_3329_from_stage2(poly, zetas, ltid, s2lo, s2hi);
        return;
    }

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint len_log = 31u - clz(half_n);
    uint k_start = 1u;

    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid >> len_log;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (len_log + 1u)) | j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        bfly_tg_generic(a, j, length, zeta, q);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        len_log -= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.01 ms, 0.2 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.7 GB/s (1.4% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 25.6 GB/s (12.8% of 200 GB/s)
  score (gmean of fraction): 0.0117

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_reduce_generic(ulong t, uint q) {
    return (uint)(t % (ulong)q);
}

inline uint mod_mul_generic(uint a, uint b, uint q) {
    return mod_reduce_generic((ulong)a * (ulong)b, q);
}

inline uint mod_add_generic(uint a, uint b, uint q) {
    uint s = a + b;
    return ((s < a) || (s >= q)) ? (s - q) : s;
}

inline uint mod_sub_generic(uint a, uint b, uint q) {
    uint d = a - b;
    return (a >= b) ? d : (d + q);
}

// Exact for canonical inputs modulo 3329.
// v < 3329^2, and qhat=(v*315)>>20 is never too small and is at most
// one too large.  The one-too-large case wraps negative and is corrected
// by adding q.
inline uint mod_mul_3329(uint a, uint b) {
    uint v    = a * b;
    uint qhat = (v * 315u) >> 20;
    uint r    = v - qhat * 3329u;
    return (r >= 3329u) ? (r + 3329u) : r;
}

inline uint mod_add_3329(uint a, uint b) {
    uint s = a + b;
    uint r = s - 3329u;
    return (s >= 3329u) ? r : s;
}

inline uint mod_sub_3329(uint a, uint b) {
    uint d = a - b;
    return (a >= b) ? d : (d + 3329u);
}

inline void bfly_tg_generic(threadgroup uint *s, uint j, uint len, uint zeta, uint q) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_generic(zeta, y, q);
    s[j]       = mod_add_generic(x, t, q);
    s[j + len] = mod_sub_generic(x, t, q);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    // Runtime-selected Kyber fast path: q=3329, n=256, stages len=128..2.
    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + ((size_t)tgid << 8);

        // Stage 0, len = 128.  Keep both outputs in registers.
        uint x = poly[ltid];
        uint y = poly[ltid + 128u];
        uint t = mod_mul_3329(zetas[1u], y);
        uint lo = mod_add_3329(x, t);     // position ltid
        uint hi = mod_sub_3329(x, t);     // position ltid + 128

        // Stage 1, len = 64.  Threads paired by xor 64 exchange only the
        // value the partner needs: lower half publishes hi, upper publishes lo.
        uint upper64 = ltid >> 6u;
        a[ltid] = (upper64 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ex = a[ltid ^ 64u];
        x = (upper64 != 0u) ? ex : lo;
        y = (upper64 != 0u) ? hi : ex;
        t = mod_mul_3329(zetas[2u + upper64], y);
        lo = mod_add_3329(x, t);
        hi = mod_sub_3329(x, t);

        // Stage 2, len = 32.  Same register/scratch exchange pattern, now
        // paired by xor 32 inside each 64-coefficient region.  Use the upper
        // half of scratch to avoid a read/write hazard with the previous slot.
        uint upper32 = (ltid >> 5u) & 1u;
        a[128u + ltid] = (upper32 != 0u) ? lo : hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ex = a[128u + (ltid ^ 32u)];
        x = (upper32 != 0u) ? ex : lo;
        y = (upper32 != 0u) ? hi : ex;

        uint group32 = ltid >> 5u;
        t = mod_mul_3329(zetas[4u + group32], y);
        uint s2lo = mod_add_3329(x, t);
        uint s2hi = mod_sub_3329(x, t);

        // Stages 3..6 are entirely inside one SIMD group.  Each SIMD group
        // owns a 64-coefficient region; each half-SIMD owns one 32-coeff chunk.
        uint lane  = ltid & 31u;
        uint r     = lane & 15u;
        uint chunk = ltid >> 4u;          // 0..7
        uint base  = chunk << 5u;

        // Stage 3, len = 16: exchange across half-SIMDs with xor 16.
        uint2 got = simd_shuffle_xor(uint2(s2lo, s2hi), (ushort)16);
        uint upper = lane >> 4u;
        x = (upper != 0u) ? got.y : s2lo;
        y = (upper != 0u) ? s2hi  : got.x;
        t = mod_mul_3329(zetas[8u + chunk], y);
        uint v0 = mod_add_3329(x, t);
        uint v1 = mod_sub_3329(x, t);

        // Stage 4, len = 8.
        got = simd_shuffle_xor(uint2(v0, v1), (ushort)8);
        upper = r >> 3u;
        x = (upper != 0u) ? got.y : v0;
        y = (upper != 0u) ? v1    : got.x;
        t = mod_mul_3329(zetas[16u + (chunk << 1u) + upper], y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 5, len = 4.
        got = simd_shuffle_xor(uint2(v0, v1), (ushort)4);
        upper = (r >> 2u) & 1u;
        x = (upper != 0u) ? got.y : v0;
        y = (upper != 0u) ? v1    : got.x;
        t = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], y);
        v0 = mod_add_3329(x, t);
        v1 = mod_sub_3329(x, t);

        // Stage 6, len = 2.
        got = simd_shuffle_xor(uint2(v0, v1), (ushort)2);
        upper = (r >> 1u) & 1u;
        x = (upper != 0u) ? got.y : v0;
        y = (upper != 0u) ? v1    : got.x;
        t = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], y);

        uint p = ((r >> 1u) << 2u) | (r & 1u);
        poly[base + p]      = mod_add_3329(x, t);
        poly[base + p + 2u] = mod_sub_3329(x, t);
        return;
    }

    // Fully generic fallback for all runtime parameter sets.
    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * (size_t)n;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint len_log = 31u - clz(half_n);
    uint k_start = 1u;

    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid >> len_log;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (len_log + 1u)) | j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        bfly_tg_generic(a, j, length, zeta, q);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        len_log -= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.01 ms, 0.2 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.01 ms, 2.9 GB/s (1.5% of 200 GB/s)
          kyb_B256: correct, 0.02 ms, 27.0 GB/s (13.5% of 200 GB/s)
  score (gmean of fraction): 0.0122

## History

- iter  3: compile=OK | correct=True | score=0.009464537722598837
- iter  4: compile=OK | correct=True | score=0.012218838527923995
- iter  5: compile=OK | correct=True | score=0.011587969732483315
- iter  6: compile=OK | correct=True | score=0.011985789122542364
- iter  7: compile=OK | correct=True | score=0.011951749068987608
- iter  8: compile=OK | correct=True | score=0.011658683949925334
- iter  9: compile=OK | correct=True | score=0.01135762577754541
- iter 10: compile=OK | correct=True | score=0.011731266130164458

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel and
propose a STRUCTURALLY different approach.

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
