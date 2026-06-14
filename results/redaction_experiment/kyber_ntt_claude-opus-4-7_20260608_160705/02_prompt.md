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
constexpr constant uint Z_MAX = 256u;

// Barrett-style reduction: given t < q^2 < 2^46, compute t mod q
// using precomputed m = floor(2^k / q). We use k = 46 so m fits in 32 bits
// for q up to ~2^23. Actually use a generic 64-bit approach.
// Since q < 2^23, t = a*b < 2^46. We compute:
//   approx = (t * m) >> shift; r = t - approx * q; if (r >= q) r -= q;
// where m = floor(2^shift / q) and shift = 46.
// But simpler and correct: just use (t - (t/q)*q). The division by a
// runtime q is slow. Use Barrett with shift=46.

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Barrett reduction. m_barrett = floor(2^48 / q). Then for t < 2^46:
//   q_est = (t * m) >> 48;  r = t - q_est * q;
//   r may be in [0, 2*q); subtract q if needed.
inline uint mod_mul_barrett(uint a, uint b, uint q, ulong m_barrett) {
    ulong t = (ulong)a * (ulong)b;
    // multiply t (up to 2^46) by m (up to 2^25 for q ~ 2^23) -> up to 2^71. Overflow!
    // Use mulhi style: compute high 64 bits of t * m.
    // Easier: shift t down first. Since t < 2^46 and we want (t*m) >> 48,
    // and m = floor(2^48/q), use the high part.
    // Actually we can use: q_est = mulhi(t, m) >> (48 - 64) ... messy.
    // Simplest correct path: fall back to t % q. The host's q is small;
    // compiler may optimize. But that's what we want to avoid.
    //
    // Use 128-bit emulation via two 32-bit pieces:
    // t = t_hi * 2^32 + t_lo
    // t * m = t_hi*m * 2^32 + t_lo*m
    uint t_lo = (uint)t;
    uint t_hi = (uint)(t >> 32);
    ulong prod_lo = (ulong)t_lo * m_barrett;
    ulong prod_hi = (ulong)t_hi * m_barrett;
    // (t * m) >> 48 = ((prod_hi << 32) + prod_lo) >> 48
    //               = (prod_hi >> 16) + (prod_lo >> 48)  [with carry from low bits]
    // Combine: total = prod_hi << 32 | ... but we just need >>48.
    // result = (prod_hi >> 16) + (prod_lo >> 48)
    // We need to include carry from (prod_hi_low_16 << 32 + prod_lo) overflow into bit 48.
    // top 16 bits of prod_lo combine with low 16 bits of prod_hi:
    ulong mid = (prod_hi << 16) + (prod_lo >> 32); // careful: this can overflow
    // safer: do it carefully
    ulong q_est = (prod_hi >> 16) + ((prod_hi & 0xFFFFul) << 16) + (prod_lo >> 32);
    q_est >>= 16;
    // Hmm, let me redo cleanly.
    // total_high64 = prod_hi + (prod_lo >> 32) + carry_from_low_part
    // Actually: t*m as 128-bit = prod_hi * 2^32 + prod_lo (treating both as 64-bit).
    // Let H = prod_hi, L = prod_lo.
    // Then total = H * 2^32 + L. >> 48 = (H >> 16) + (L >> 48) provided we
    // account for the low 16 bits of H lining up with bits 32..47 of total.
    // bits [48..) of total = bits [16..) of H + carry from (bits [0..16) of H << 32 + L) into bit 48.
    ulong low_part = ((H_placeholder()) << 32); // placeholder removed below
    return 0u;
}

// Given the complexity, use a cleaner approach: precompute m = 2^32 / q approx
// using a different strategy. Or just trust the compiler on uint % uint.

inline uint mod_mul(uint a, uint b, uint q) {
    ulong t = (ulong)a * (ulong)b;
    return (uint)(t % (ulong)q);
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
    threadgroup uint zs[Z_MAX];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load of coefficients
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperative load of zetas into threadgroup memory
    uint z_count = 1u << n_levels;
    if (ltid < z_count) {
        zs[ltid] = zetas[ltid];
    }
    if (ltid + half_n < z_count) {
        zs[ltid + half_n] = zetas[ltid + half_n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zs[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, q);

        uint sum  = mod_add(x, t, q);
        uint diff = mod_sub(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum;
        a[j + length] = diff;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
  COMPILE FAILED: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:62:24: error: use of undeclared identifier 'H_placeholder'
    ulong low_part = ((H_placeholder()) << 32); // placeholder removed below
                       ^
" UserInfo={NSLocalizedDescription=program_source:62:24: error: use of undeclared identifier 'H_placeholder'
    ulong low_part = ((H_placeholder()) << 32); // placeholder removed below
                       ^
}

## Current best (incumbent)

```metal
// Naive seed for a batched negacyclic NTT (Z6, forward).
//
// One threadgroup per polynomial: each threadgroup runs all n_levels
// Cooley-Tukey butterfly stages in threadgroup memory and writes back.
// Per-stage barrier flushes the in-place updates so the next stage
// reads consistent values.
//
// Convention (matches the FIPS 203 / FIPS 204 / pqclean references):
//
//   k = 1
//   for level = 0..n_levels:
//       len = n >> (level + 1)
//       for start = 0, 2*len, ..., n - 2*len:
//           zeta = zetas[k++]
//           for j = start..start + len - 1:
//               t          = (zeta * a[j+len]) mod q
//               a[j+len]   = (a[j] - t)        mod q
//               a[j]       = (a[j] + t)        mod q
//
// Equivalent per-thread mapping (ltid in [0, n/2) owns one butterfly
// at every level):
//
//   group_idx   = ltid / len
//   j_in_group  = ltid - group_idx * len      // ltid % len
//   j           = (group_idx << 1) * len + j_in_group
//   zeta_index  = k_start + group_idx          // k_start = 1 << level
//
// Zetas table (host-precomputed, length 2^n_levels):
//   zetas[k] = zeta^bit_reverse(k, n_levels)   mod q
// where zeta is a primitive 2^(n_levels+1)-th root of unity in F_q.
// The concrete (q, n_levels, zeta) values are bound at runtime through
// the q and n_levels constant buffers and the zetas device buffer;
// the kernel does not need to know which parameter set is in play.
// Entry zetas[0] = 1 is the unread identity element (k starts at 1).
//
// Buffer layout (host-fixed; must be preserved by candidate kernels):
//   buffer 0: device       uint *coeffs       (length batch * n;
//             read+written in place)
//   buffer 1: device const uint *zetas        (length 1 << n_levels)
//   buffer 2: constant uint     &q            (modulus; 3329 or 8380417)
//   buffer 3: constant uint     &n            (polynomial length; 256)
//   buffer 4: constant uint     &n_levels     (number of NTT stages)
//   buffer 5: constant uint     &batch
//
// Dispatch (host-provided):
//   threadsPerGrid        = (batch * (n/2), 1, 1)
//   threadsPerThreadgroup = (n/2, 1, 1)
// Each threadgroup owns ONE polynomial; ltid in [0, n/2) owns one
// butterfly per stage.
//
// Outputs MUST be canonical ([0, q)); a non-canonical value with the
// same residue class still counts as a mismatch on the host-side
// reference comparison. n is a power of two with n <= 256 and
// n_levels <= 8 across every test size, so n/2 <= 128 threads per
// threadgroup and a static scratch of size 256 cover every case.

#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_reduce(ulong t, uint q) {
    return (uint)(t % (ulong)q);
}

inline uint mod_mul(uint a, uint b, uint q) {
    return mod_reduce((ulong)a * (ulong)b, q);
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

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Each thread loads two coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;          // halves each stage
    uint k_start = 1u;              // index into zetas for the first group
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, q);

        a[j]          = mod_add(x, t, q);
        a[j + length] = mod_sub(x, t, q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    // Writeback.
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.04 ms, 0.1 GB/s (0.0% of 200 GB/s)
           kyb_B16: correct, 0.04 ms, 1.0 GB/s (0.5% of 200 GB/s)
          kyb_B256: correct, 0.07 ms, 9.3 GB/s (4.6% of 200 GB/s)
  score (gmean of fraction): 0.0041

## History

- iter  0: compile=OK | correct=True | score=0.0041359315746271105
- iter  1: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
