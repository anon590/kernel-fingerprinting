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
    if (tgid >= batch) return;

    threadgroup uint a[N_MAX];

    const uint half_n = n >> 1u;
    const uint qv     = q;
    const uint nl     = n_levels;

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load.
    uint x0 = poly[ltid];
    uint y0 = poly[ltid + half_n];

    // Stage 0 fused with the load: zeta_index = 1, length = half_n,
    // j = ltid, j+length = ltid+half_n.
    {
        uint zeta = zetas[1];
        // q small (Kyber q=3329), so zeta*y0 fits well in 32 bits.
        uint prod = zeta * y0;
        uint t = prod % qv;

        uint sum_  = x0 + t;
        if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x0 + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        a[ltid]          = sum_;
        a[ltid + half_n] = diff_;
    }

    // log2(length) starts at log2(half_n) - 1, since stage 0 already done.
    // length for level L is n >> (L+1). We track log2_len.
    // Compute log2(half_n): nl-1 because n = 1 << nl is the typical setup,
    // but n could be smaller; derive via clz.
    uint log2_len = (uint)(31 - clz(half_n)) - 1u; // log2(half_n/2)
    uint k_start  = 2u;

    for (uint level = 1u; level < nl; ++level) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint length     = 1u << log2_len;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint prod = zeta * y;
        uint t = prod % qv;

        uint sum_  = x + t;
        if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        // Each thread's writes target the same (j, j+length) it just read.
        // The next level's reads happen after the barrier at the top.
        a[j]          = sum_;
        a[j + length] = diff_;

        k_start <<= 1u;
        if (log2_len > 0u) log2_len -= 1u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.2 GB/s (1.1% of 200 GB/s)
          kyb_B256: correct, 0.04 ms, 17.0 GB/s (8.5% of 200 GB/s)
  score (gmean of fraction): 0.0087

## History

- iter  4: compile=OK | correct=True | score=0.008081877454414984
- iter  5: compile=OK | correct=True | score=0.0074012100634580765
- iter  6: compile=OK | correct=True | score=0.007831845156251705
- iter  7: compile=OK | correct=True | score=0.006255263608148074
- iter  8: compile=OK | correct=True | score=0.005071305134467653
- iter  9: compile=OK | correct=True | score=0.005245034950234042
- iter 10: compile=OK | correct=True | score=0.002774373282585404
- iter 11: compile=OK | correct=True | score=0.008664550541168709

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
