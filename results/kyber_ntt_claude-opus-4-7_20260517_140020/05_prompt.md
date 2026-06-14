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

// Fast mulhi for 64-bit using metal's built-in mulhi on uint halves.
// Computes the high 64 bits of (a * b) where a,b are 64-bit.
inline ulong mulhi64(ulong a, ulong b) {
    uint a_lo = (uint)a;
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)b;
    uint b_hi = (uint)(b >> 32);

    // 32x32 -> 64 partial products via metal mulhi (high) + plain mul (low).
    ulong ll = (ulong)a_lo * (ulong)b_lo;
    ulong lh = (ulong)a_lo * (ulong)b_hi;
    ulong hl = (ulong)a_hi * (ulong)b_lo;
    ulong hh = (ulong)a_hi * (ulong)b_hi;

    ulong mid = (ll >> 32) + (uint)lh + (uint)hl;
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
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
    threadgroup uint a[N_MAX];
    threadgroup uint zcache[N_MAX];

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    // Barrett reciprocal: m = floor(2^64 / q). For t < q*2^32 <= q*q*... we get
    // q_est = mulhi64(t, m) in [t/q - 1, t/q], so r = t - q_est*q in [0, 2q).
    ulong mbar = (~(ulong)0) / (ulong)qv;
    ulong rem_chk = (~(ulong)0) - mbar * (ulong)qv;
    if (rem_chk == (ulong)qv - 1u) mbar += 1u;

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients.
    uint c0 = poly[ltid];
    uint c1 = poly[ltid + half_n];
    a[ltid]          = c0;
    a[ltid + half_n] = c1;

    // Cache zetas (up to n entries; only (1<<nlv) are used).
    zcache[ltid]          = zetas[ltid];
    zcache[ltid + half_n] = zetas[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Track length as power of 2 with explicit log2_len to avoid ctz each level.
    // half_n = n/2; for n=256, half_n=128, log2(half_n)=7.
    uint log2_len = 0u;
    {
        uint v = half_n;
        while (v > 1u) { v >>= 1u; log2_len++; }
    }
    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zcache[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // Barrett reduction of zeta*y mod q.
        ulong tfull = (ulong)zeta * (ulong)y;
        ulong qest  = mulhi64(tfull, mbar);
        uint  r     = (uint)(tfull - qest * (ulong)qv);
        r = (r >= qv) ? (r - qv) : r;

        // Canonical add/sub without branches.
        uint s = x + r;
        s = (s >= qv) ? (s - qv) : s;
        uint d = x + qv - r;
        d = (d >= qv) ? (d - qv) : d;

        a[j]          = s;
        a[j + length] = d;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        if (log2_len > 0u) log2_len--;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.05 ms, 4.2 Gmodmul/s (u32) (8.0% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0074

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

// Reduce a 64-bit value t (with t < q * 2^32, i.e. zeta*y where zeta,y < q < 2^32)
// to canonical [0, q) using Barrett with a 32-bit reciprocal.
//
// We need m and shift such that for all t in [0, q*2^32):
//   floor(t / q) == floor(t * m / 2^(32+shift))   approximately, with <=2 correction.
//
// Use: shift = ceil(log2(q)); m = floor(2^(32+shift) / q).
// Then q_est = mulhi(t_lo_or_full * m), corrected.
//
// Simpler: since q < 2^24 in all real ZK params (Kyber 3329, Dilithium 8380417),
// just use 64-bit mul by m where m = floor(2^k / q), and shift right by k.
// We pick k = 32 + ceil(log2(q)) so quotient fits.

// Compute mulhi for 64-bit: high 64 bits of a*b, where a,b are 64-bit.
inline ulong mulhi64(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFULL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFULL;
    ulong b_hi = b >> 32;
    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;
    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
    return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    int d = (int)a - (int)b;
    return (d < 0) ? (uint)(d + (int)q) : (uint)d;
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
    threadgroup uint a[N_MAX];
    threadgroup uint zcache[N_MAX]; // cache zetas[1..n-1]

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    // Precompute Barrett constants for 64-bit dividend:
    //   shift = bit_width(q)  (smallest s with 2^s >= q, but we use ceil-log2 robustly)
    //   m = floor(2^(64) / q)
    // Then for t < q * 2^32 <= 2^(32+shift):
    //   q_est = mulhi64(t, m); remainder = t - q_est * q in [0, 2q); correct.
    ulong mbar = (~(ulong)0) / (ulong)qv;
    // possible +1 correction so that mbar = floor(2^64/q) exactly when 2^64 % q == 0
    ulong rem  = (~(ulong)0) - mbar * (ulong)qv;
    if (rem == (ulong)qv - 1) mbar += 1;

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cache zetas[0..n-1] (at most 256 entries). Two loads per thread covers n/2 threads.
    // Total zetas used: (1 << nlv) entries, which is <= n. We just load n entries; extras unused.
    zcache[ltid]          = zetas[ltid];
    zcache[ltid + half_n] = zetas[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = (length == 0u) ? 0u : (ltid >> (uint)ctz(length));
        // ltid / length where length is power of two
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zcache[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t_full = zeta * y, fits in 64 bits since both < 2^32.
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett: q_est = mulhi64(tfull, mbar); r = tfull - q_est * q
        ulong qest = mulhi64(tfull, mbar);
        uint  r    = (uint)(tfull - qest * (ulong)qv);
        // r in [0, 2q); single correction.
        if (r >= qv) r -= qv;

        a[j]          = mod_add(x, r, qv);
        a[j + length] = mod_sub(x, r, qv);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.05 ms, 4.4 Gmodmul/s (u32) (8.2% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0076

## History

- iter  0: compile=OK | correct=True | score=0.0036622358479690953
- iter  1: compile=OK | correct=True | score=0.004894578014846105
- iter  2: compile=OK | correct=True | score=0.007616842172810286
- iter  3: compile=OK | correct=True | score=0.006342152440686771
- iter  4: compile=OK | correct=True | score=0.007407634760317044

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
