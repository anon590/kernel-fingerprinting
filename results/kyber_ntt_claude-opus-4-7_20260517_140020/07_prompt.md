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

    uint qv     = q;
    uint nv     = n;
    uint nlv    = n_levels;
    uint half_n = nv >> 1u;

    // Barrett constants.
    // s = bit_width(q): smallest integer with 2^s >= q+1 (i.e., q < 2^s).
    // For q in [1, 2^31], s in [1, 31].
    // mbar = floor(2^(32 + s) / q).
    //   2^(32+s)/q < 2^(32+s)/2^(s-1) = 2^33, so mbar can be up to 33 bits.
    //   But if q > 2^(s-1), tighter: mbar < 2^(32+s)/q.
    // To keep mbar in 32 bits we'd need q >= 2^s, contradiction. So mbar can be 33 bits.
    // We store mbar as ulong but split into (mbar_lo, mbar_hi in {0,1}).
    //
    // For tfull < q^2 < 2^(2s) <= 2^(32+s) (since s <= 32),
    //   thi = tfull >> s   < 2^s <= 2^32, fits in u32 (when s<32; for s=32 we'd need care).
    //   q_est = (thi * mbar) >> 32 = mulhi(thi, mbar_lo) + thi * mbar_hi
    //         in {floor(tfull/q) - e, floor(tfull/q)} for small e (<=1).
    //   r = (u32)tfull - q_est * q, in [0, 2q); one correction.
    uint s = 32u - clz(qv);                 // ceil(log2(q+1)); 0 only if q==0
    if (qv == 0u) s = 0u;
    // (qv is a real prime, never 0; guard kept for safety)
    ulong mbar = ((ulong)1 << (32u + s)) / (ulong)qv;
    uint  mbar_lo = (uint)mbar;
    uint  mbar_hi = (uint)(mbar >> 32);     // 0 or 1

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // tfull = zeta * y, < q^2 < 2^(2s).
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett reduction.
        uint  thi   = (uint)(tfull >> s);
        // q_est = high-32 of (thi * mbar). Since mbar = mbar_hi*2^32 + mbar_lo:
        //   thi * mbar = thi*mbar_lo + thi*mbar_hi * 2^32
        //   high 32 of thi*mbar_lo = mulhi(thi, mbar_lo)
        //   thi*mbar_hi contributes thi*mbar_hi (mbar_hi in {0,1})
        uint  qest  = mulhi(thi, mbar_lo) + thi * mbar_hi;

        uint  r     = (uint)tfull - qest * qv;
        // r in [0, 2q); one correction (Barrett may under-estimate by 1).
        if (r >= qv) r -= qv;
        // Extra safety: in rare edge cases r could be 2q (if estimate under by 2).
        // Bound: |q_est - floor(t/q)| <= 1 with this Barrett, so r in [0, 2q). One sub suffices.

        // Butterfly.
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

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: INCORRECT (bit_exact=37)
  fail_reason: correctness failed at size kyb_B1: bit_exact=37

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
- iter  5: compile=OK | correct=False | score=N/A
- iter  6: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
