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
    // Correct even when q > 2^31 and the 32-bit add wraps.
    return ((s < a) || (s >= q)) ? (s - q) : s;
}

inline uint mod_sub_generic(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Exact for canonical inputs modulo 3329.
// floor(2^32 / 3329) = 1290167.  For v < 3329^2, one correction suffices.
inline uint mod_mul_3329(uint a, uint b) {
    uint v = a * b;
    uint qhat = (uint)(((ulong)v * (ulong)1290167u) >> 32);
    uint r = v - qhat * 3329u;
    return (r >= 3329u) ? (r - 3329u) : r;
}

inline uint mod_add_3329(uint a, uint b) {
    uint s = a + b;
    return (s >= 3329u) ? (s - 3329u) : s;
}

inline uint mod_sub_3329(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + 3329u - b);
}

inline void bfly_tg_3329(threadgroup uint *s, uint j, uint len, uint zeta) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_3329(zeta, y);
    s[j]       = mod_add_3329(x, t);
    s[j + len] = mod_sub_3329(x, t);
}

inline void bfly_tg_generic(threadgroup uint *s, uint j, uint len, uint zeta, uint q) {
    uint x = s[j];
    uint y = s[j + len];
    uint t = mod_mul_generic(zeta, y, q);
    s[j]       = mod_add_generic(x, t, q);
    s[j + len] = mod_sub_generic(x, t, q);
}

// Fetch a coefficient from the previous register-resident CT stage.
// vals.x/vals.y are the low/high outputs owned by each lane in a 16-lane
// half-simdgroup.  pos is local to the 32-coefficient chunk.
inline uint fetch_reg_pair(uint2 vals, uint pos, uint prev_log, uint half_base) {
    uint prev_len = 1u << prev_log;
    uint span_log = prev_log + 1u;
    uint offset   = pos & ((1u << span_log) - 1u);
    uint owner    = ((pos >> span_log) << prev_log) | (offset & (prev_len - 1u));
    uint2 got     = simd_shuffle(vals, (ushort)(half_base + owner));
    return ((offset & prev_len) != 0u) ? got.y : got.x;
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

    // Fast path for the Kyber parameter set, selected from runtime buffers.
    if (q == 3329u && n == 256u && n_levels == 7u) {
        device uint *poly = coeffs + (size_t)tgid * (size_t)256u;

        // Stage 0, len = 128.  Load directly from global and produce scratch.
        uint x0 = poly[ltid];
        uint y0 = poly[ltid + 128u];
        uint t0 = mod_mul_3329(zetas[1], y0);
        a[ltid]        = mod_add_3329(x0, t0);
        a[ltid + 128u] = mod_sub_3329(x0, t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 1, len = 64.
        uint g = ltid >> 6u;
        uint j = (g << 7u) | (ltid & 63u);
        bfly_tg_3329(a, j, 64u, zetas[2u + g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2, len = 32.
        g = ltid >> 5u;
        j = (g << 6u) | (ltid & 31u);
        bfly_tg_3329(a, j, 32u, zetas[4u + g]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Remaining stages are independent within 32-coefficient chunks.
        // Each simdgroup handles two chunks; each lane in a half-simdgroup
        // carries the two outputs of its current butterfly in registers.
        uint lane      = ltid & 31u;
        uint half_base = lane & 16u;
        uint r         = lane & 15u;
        uint chunk     = ltid >> 4u;      // 0..7
        uint base      = chunk << 5u;     // coefficient base of 32-point chunk

        // Stage 3, len = 16: load from scratch into registers.
        uint rx = a[base + r];
        uint ry = a[base + r + 16u];
        uint rt = mod_mul_3329(zetas[8u + chunk], ry);
        uint v0 = mod_add_3329(rx, rt);
        uint v1 = mod_sub_3329(rx, rt);

        // Stage 4, len = 8.
        uint p = ((r >> 3u) << 4u) | (r & 7u);
        uint2 vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      4u, half_base);
        ry = fetch_reg_pair(vals, p + 8u, 4u, half_base);
        rt = mod_mul_3329(zetas[16u + (chunk << 1u) + (r >> 3u)], ry);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 5, len = 4.
        p = ((r >> 2u) << 3u) | (r & 3u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      3u, half_base);
        ry = fetch_reg_pair(vals, p + 4u, 3u, half_base);
        rt = mod_mul_3329(zetas[32u + (chunk << 2u) + (r >> 2u)], ry);
        v0 = mod_add_3329(rx, rt);
        v1 = mod_sub_3329(rx, rt);

        // Stage 6, len = 2.  Store final butterfly outputs directly.
        p = ((r >> 1u) << 2u) | (r & 1u);
        vals = uint2(v0, v1);
        rx = fetch_reg_pair(vals, p,      2u, half_base);
        ry = fetch_reg_pair(vals, p + 2u, 2u, half_base);
        rt = mod_mul_3329(zetas[64u + (chunk << 3u) + (r >> 1u)], ry);

        poly[base + p]      = mod_add_3329(rx, rt);
        poly[base + p + 2u] = mod_sub_3329(rx, rt);
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

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.2 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.4 GB/s (1.2% of 200 GB/s)
          kyb_B256: correct, 0.04 ms, 17.6 GB/s (8.8% of 200 GB/s)
  score (gmean of fraction): 0.0093

## History

- iter  0: compile=OK | correct=True | score=0.004143036155423717
- iter  1: compile=OK | correct=True | score=0.009293009205202641

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
