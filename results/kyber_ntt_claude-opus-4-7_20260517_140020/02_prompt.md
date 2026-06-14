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

// Compute floor(2^64 / q) using Metal's 64-bit unsigned division.
// Called once per threadgroup; q is small so this is cheap relative
// to the savings during the butterfly inner loop.
inline ulong barrett_m(uint q) {
    // 2^64 / q  =  ((2^64 - 1) / q) with a possible +1 correction.
    // Use: m = (~0ULL) / q; if ((~0ULL) - m*q == q-1) m += 1; but
    // simpler: compute via two halves. We just use ulong division.
    ulong qq = (ulong)q;
    ulong m  = (~(ulong)0) / qq;
    // Check if 2^64 is exactly divisible; if (~0ULL) % q == q-1 then add 1.
    ulong r  = (~(ulong)0) - m * qq;
    if (r == qq - 1) m += 1;
    return m;
}

// Reduce a 64-bit t into [0, q) given m = floor(2^64 / q).
// h = floor(t * m / 2^64); r = t - h*q; r in [0, 2q); subtract q if needed.
inline uint barrett_reduce(ulong t, uint q, ulong m) {
    // Compute high 64 bits of t * m using 32-bit halves.
    ulong t_lo = t & 0xFFFFFFFFULL;
    ulong t_hi = t >> 32;
    ulong m_lo = m & 0xFFFFFFFFULL;
    ulong m_hi = m >> 32;

    ulong ll = t_lo * m_lo;
    ulong lh = t_lo * m_hi;
    ulong hl = t_hi * m_lo;
    ulong hh = t_hi * m_hi;

    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    ulong h   = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);

    ulong r = t - h * (ulong)q;
    // r is in [0, 2q) (typically), correct with up to 2 subtractions to be safe.
    uint ru = (uint)r;
    if (ru >= q) ru -= q;
    if (ru >= q) ru -= q;
    return ru;
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

kernel void kyber_ntt(
    device       uint *coeffs     [[buffer(0)]],
    device const uint *zetas      [[buffer(1)]],
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    uint qv      = q;
    ulong m      = barrett_m(qv);
    uint half_n  = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load two coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    uint nlv     = n_levels;
    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = barrett_reduce((ulong)zeta * (ulong)y, qv, m);

        a[j]          = mod_add(x, t, qv);
        a[j + length] = mod_sub(x, t, qv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.4 Gmodmul/s (u32) (0.8% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.16 ms, 1.5 Gmodmul/s (u32) (2.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0049

## History

- iter  0: compile=OK | correct=True | score=0.0036622358479690953
- iter  1: compile=OK | correct=True | score=0.004894578014846105

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
