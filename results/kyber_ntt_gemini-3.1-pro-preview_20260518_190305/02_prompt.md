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

// Fast 32-bit Barrett reduction for Y < 2^32.
// Requires precomputed mu = floor(2^32 / q).
inline uint mod32(uint Y, uint q, uint mu) {
    uint q_est = mulhi(Y, mu);
    uint rem = Y - q_est * q;
    // Mathematical guarantee: error in q_est is at most 1, so rem < 2q.
    return (rem >= q) ? (rem - q) : rem;
}

// Computes (a * b) % q efficiently without 64-bit division.
inline uint mod_mul(uint a, uint b, uint q, uint mu, uint R) {
    ulong P = (ulong)a * b;
    uint P_hi = (uint)(P >> 32);
    uint P_lo = (uint)P;

    // Y = P_hi * R. Decompose Y into x_hi * 2^32 + x_lo
    uint x_hi = mulhi(P_hi, R);
    uint x_lo = P_hi * R;

    // Sum the lower parts and catch the carry (each carry represents 2^32 = R mod q)
    uint sum_lo = x_lo + P_lo;
    uint carry = (sum_lo < x_lo) ? 1u : 0u;

    uint sum_hi_R = (x_hi + carry) * R;

    uint sum_lo_mod = mod32(sum_lo, q, mu);
    uint total_sum = sum_hi_R + sum_lo_mod;

    return mod32(total_sum, q, mu);
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
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    // Load kernel-wide scalars into fast registers
    uint local_q        = q;
    uint local_n        = n;
    uint local_n_levels = n_levels;

    // Precompute constants for the fast 64-bit to 32-bit reduction
    uint mu = (uint)(0x100000000ull / local_q);
    uint R  = (uint)(0x100000000ull % local_q);

    threadgroup uint a[N_MAX];
    threadgroup uint shared_zetas[256];

    uint half_n = local_n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * local_n;

    // Each thread loads two coefficients into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperatively load zetas twiddles into threadgroup memory
    uint zetas_count = 1u << local_n_levels;
    for (uint i = ltid; i < zetas_count; i += half_n) {
        shared_zetas[i] = zetas[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log_len = ctz(half_n);
    uint k_start = 1u;

    // Batched Cooley-Tukey NTT
    for (uint level = 0u; level < local_n_levels; ++level) {
        uint length     = 1u << log_len;
        
        // Fast power-of-two arithmetic (replaces hardware division and modulo)
        uint group_idx  = ltid >> log_len;
        uint j          = (group_idx << (log_len + 1u)) | (ltid & (length - 1u));
        uint zeta       = shared_zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul(zeta, y, local_q, mu, R);

        a[j]          = mod_add(x, t, local_q);
        a[j + length] = mod_sub(x, t, local_q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        log_len--;
    }

    // Write-back out of place to VRAM
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.13 ms, 1.7 Gmodmul/s (u32) (3.3% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0055

## History

- iter  0: compile=OK | correct=True | score=0.0043181043832901085
- iter  1: compile=OK | correct=True | score=0.005482531848977972

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
