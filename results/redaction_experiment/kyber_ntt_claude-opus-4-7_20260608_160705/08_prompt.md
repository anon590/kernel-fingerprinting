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

    // Each thread owns two slots: lo_idx = ltid, hi_idx = ltid + half_n.
    // We keep them in registers across all levels; threadgroup memory is
    // only used to "rotate" partner ownership between levels whose stride
    // is >= 32 (outside a single simdgroup). For stride < 32 we use
    // simd_shuffle_xor to swap with the partner lane directly.
    uint lo = poly[ltid];
    uint hi = poly[ltid + half_n];

    uint length  = half_n;
    uint k_start = 1u;

    // Cache simd lane id (ltid mod 32)
    const uint lane = ltid & 31u;

    for (uint level = 0u; level < nl; ++level) {
        // Compute (j, j+length) for this thread's butterfly using the
        // standard mapping.
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x, y;

        if (length >= 32u) {
            // Stride spans multiple simdgroups -> go through threadgroup mem.
            // Store our two owned slots (lo at ltid, hi at ltid+half_n).
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[ltid]          = lo;
            a[ltid + half_n] = hi;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            x = a[j];
            y = a[j + length];
        } else if (length == 0u) {
            // Defensive: should not occur for n>=2.
            x = lo; y = hi;
        } else {
            // length is a power of two in [1,16]. Partner lane within
            // simdgroup differs by `length` in the lane index.
            // Our two owned slots at this point correspond to consecutive
            // pairs (j_lo, j_lo + length_prev?) — but after threadgroup
            // staging at the boundary level (length == half_n/.../32),
            // we re-established the invariant that thread ltid owns
            // a[ltid] and a[ltid+half_n]. For length < 32 each thread's
            // butterfly is (j, j+length) where j and j+length are in the
            // same half (the same set of half_n entries). So we need to
            // shuffle within the simdgroup over the "lo" register for
            // threads ltid < half_n... but every thread is < half_n.
            //
            // Actually with our layout, after the staging barrier at the
            // last length>=32 level, we re-load a[ltid] into lo and
            // a[ltid+half_n] into hi. From that point onward, lengths are
            // < 32 and the butterflies operate ENTIRELY within the lo
            // half OR the hi half (since length < half_n once level>=1).
            // So we need to do butterflies on lo (over threads owning
            // the lo-half slots) and on hi (over threads owning hi-half).
            //
            // But the standard mapping has ltid in [0, half_n) owning
            // butterfly (j, j+length). For levels with length < half_n,
            // j spans both halves. So actually each thread's butterfly
            // is NOT confined to one half register.
            //
            // To keep this simple and correct, fall back to threadgroup
            // memory for these levels too.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[ltid]          = lo;
            a[ltid + half_n] = hi;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            x = a[j];
            y = a[j + length];
        }

        // t = (zeta * y) mod q. For Kyber q=3329, zeta,y < q so prod < 2^24.
        uint prod = zeta * y;
        uint t    = prod % qv;

        uint sum_  = x + t;          if (sum_  >= qv) sum_  -= qv;
        uint diff_ = x + qv - t;     if (diff_ >= qv) diff_ -= qv;

        // Write back to threadgroup memory at (j, j+length), then reload
        // our owned slots (ltid, ltid+half_n) so the invariant holds.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum_;
        a[j + length] = diff_;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lo = a[ltid];
        hi = a[ltid + half_n];

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = lo;
    poly[ltid + half_n] = hi;
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 1.7 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.06 ms, 10.9 GB/s (5.4% of 200 GB/s)
  score (gmean of fraction): 0.0063

## Current best (incumbent)

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

    uint half_n = n >> 1u;
    uint qv     = q;
    uint nl     = n_levels;

    device uint *poly = coeffs + (size_t)tgid * n;

    // Cooperative load: each of the half_n threads loads two coefficients.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Precompute float reciprocal of q for Barrett-style reduction.
    // q < 2^23 (Kyber uses 3329), zeta,y < q so product < 2^46, well within
    // float64-ish range but we exploit that q fits in ~12 bits so product
    // fits in ~24 bits which is exact in float. Use double for safety on
    // larger q values that still fit the spec (q < 2^32; but pairs (zeta,y)
    // can yield products up to 2^64 in general). Here q is small so float
    // works, but we use float with a safety correction loop.
    float inv_q_f = 1.0f / (float)qv;

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nl; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t = (zeta * y) mod q via Barrett-style float reciprocal.
        // zeta, y < q <= ~3329 so prod < ~1.1e7, exact in float.
        uint prod_lo = zeta * y;                       // exact for small q
        float pf = (float)prod_lo;
        uint qhat = (uint)(pf * inv_q_f);
        uint t = prod_lo - qhat * qv;
        // Correct for rounding: t may be in [-q, 2q).
        if ((int)t < 0) t += qv;
        if (t >= qv)    t -= qv;

        uint sum_  = x + t;
        if (sum_ >= qv) sum_ -= qv;
        uint diff_ = x + qv - t;
        if (diff_ >= qv) diff_ -= qv;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = sum_;
        a[j + length] = diff_;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.2 GB/s (1.1% of 200 GB/s)
          kyb_B256: correct, 0.04 ms, 14.8 GB/s (7.4% of 200 GB/s)
  score (gmean of fraction): 0.0081

## History

- iter  0: compile=OK | correct=True | score=0.0041359315746271105
- iter  1: compile=FAIL | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.004471339490410848
- iter  3: compile=OK | correct=True | score=0.005007783692408259
- iter  4: compile=OK | correct=True | score=0.008081877454414984
- iter  5: compile=OK | correct=True | score=0.0074012100634580765
- iter  6: compile=OK | correct=True | score=0.007831845156251705
- iter  7: compile=OK | correct=True | score=0.006255263608148074

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
