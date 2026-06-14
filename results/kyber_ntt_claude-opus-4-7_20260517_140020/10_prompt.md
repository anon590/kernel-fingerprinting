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
    threadgroup uint a[N_MAX];
    threadgroup uint zcache[N_MAX];
    threadgroup uint zshoup[N_MAX];

    const uint qv     = q;
    const uint nv     = n;
    const uint nlv    = n_levels;
    const uint half_n = nv >> 1u;
    const uint zlen   = 1u << nlv;

    device uint *poly = coeffs + (size_t)tgid * (size_t)nv;

    // Load polynomial coefficients into threadgroup memory.
    if (ltid < half_n) {
        a[ltid]          = poly[ltid];
        a[ltid + half_n] = poly[ltid + half_n];
    }

    // Load zetas and compute Shoup constants: zsh = floor(zeta * 2^32 / q).
    // zlen <= 256, half_n <= 128, so each of the half_n threads handles up to
    // ceil(zlen / half_n) entries. We loop with stride half_n to cover all of zlen.
    for (uint idx = ltid; idx < zlen; idx += half_n) {
        uint zv = zetas[idx];
        zcache[idx] = zv;
        // zsh = floor((zv << 32) / q). Since zv < q <= 2^32-1, the numerator fits.
        zshoup[idx] = (uint)(((ulong)zv << 32) / (ulong)qv);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length   = half_n;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));
    uint k_start  = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;

        uint zeta = zcache[k_start + group_idx];
        uint zsh  = zshoup[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Shoup multiplication: returns t = zeta*y mod q in [0, 2q), then reduce.
        // q_hat = mulhi(zsh, y);  r = zeta*y - q_hat*q  (both mod 2^32)
        uint q_hat = mulhi(zsh, y);
        uint r     = zeta * y - q_hat * qv;
        if (r >= qv) r -= qv;

        // Canonical butterfly.
        uint sum = x + r;
        sum = (sum >= qv) ? (sum - qv) : sum;
        uint dif = x + qv - r;
        dif = (dif >= qv) ? (dif - qv) : dif;

        a[j]          = sum;
        a[j + length] = dif;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        log2_len = (log2_len == 0u) ? 0u : (log2_len - 1u);
    }

    if (ltid < half_n) {
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
    }
}
```

Result of previous attempt:
            kyb_B1: correct, 0.01 ms, 0.1 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.02 ms, 1.0 Gmodmul/s (u32) (1.8% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.05 ms, 4.3 Gmodmul/s (u32) (8.0% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0120

## History

- iter  2: compile=OK | correct=True | score=0.007616842172810286
- iter  3: compile=OK | correct=True | score=0.006342152440686771
- iter  4: compile=OK | correct=True | score=0.007407634760317044
- iter  5: compile=OK | correct=False | score=N/A
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.009309802474392625
- iter  8: compile=OK | correct=False | score=N/A
- iter  9: compile=OK | correct=True | score=0.012048695643084698

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
