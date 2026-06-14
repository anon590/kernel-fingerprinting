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

    const uint qv     = q;
    const uint nv     = n;
    const uint nlv    = n_levels;
    const uint half_n = nv >> 1u;

    // Barrett with 32-bit reciprocal.
    // L = bit_width(q): smallest s with 2^s >= q  ->  L = 32 - clz(q-1) (for q>=2)
    // Equivalently L = 32 - clz(q) when q is not a power of two;
    // we use L = 32 - clz(q) which gives L such that q <= 2^L, and 2^(L-1) <= q (for q>=2).
    // For zeta, y in [0, q), t = zeta*y < q^2 <= 2^(2L), but more tightly t < q^2.
    // thi = t >> (L-1):  thi < q^2 / 2^(L-1) <= q * 2^(L+1) / 2^(L-1) ... we need a cleaner bound.
    //
    // Use 64-bit Barrett: shift k = 2L; mbar = floor(2^k / q).
    // q_est = (t * mbar) >> k. We have t < q^2 <= 2^(2L), mbar < 2^(2L)/q + 1 <= 2^(L+1).
    // So t*mbar < 2^(2L) * 2^(L+1) = 2^(3L+1) -- may overflow 64 bits if L>21.
    //
    // Safer scheme that handles q up to 2^31:
    //   Compute thi = t >> 32 (high 32 bits of t).
    //   mbar = floor(2^64 / q); mbar fits in 64 bits.
    //   q_est = mulhi32(thi, mbar_hi) approx... too coarse.
    //
    // We fall back to: 32-bit Barrett where mbar = floor(2^(2L) / q), shift = 2L,
    // but split product. Since L <= 32, mbar <= 2^(L+1) fits in 33 bits.
    //
    // Cleanest correct fast path: split t into (t_hi, t_lo) 32-bit halves.
    // floor(t / q) = floor((t_hi * 2^32 + t_lo) / q).
    // Precompute: M = floor(2^64 / q) as ulong (one-time, cheap).
    // q_est_hi = mulhi(t_hi, M_hi) + ... -- this is the mulhi64 path (4 muls).
    //
    // FASTER: For our regime (q < 2^L, L = bit_width(q)),
    //   t = zeta * y < q^2.  Let t32 = t >> (L-1).  Then t32 < q^2 / 2^(L-1).
    //   Since q < 2^L: t32 < 2^(2L) / 2^(L-1) = 2^(L+1) <= 2^32 (when L<=31).
    //   Set mbar = floor(2^(L+1) * 2^31 / q) = floor(2^(L+32) / q), this fits in <= 33 bits.
    //   But simpler: use mbar32 = floor(2^32 * 2^(L-1) / q) so q_est = mulhi(t32, mbar32).
    //   Need mbar32 in 32 bits: mbar32 = floor(2^(L+31) / q). Since q >= 2^(L-1),
    //   mbar32 <= 2^(L+31)/2^(L-1) = 2^32. So mbar32 fits in 32 bits (saturates at 2^32-1 ok).
    //
    // q_est = mulhi(t32, mbar32) gives approximately floor(t / q) with error <= 2.
    // r = (uint)t - q_est * q, then correct down by q up to twice.

    uint L;
    if (qv <= 1u) { L = 1u; }
    else { L = 32u - clz(qv - 1u); }   // smallest L s.t. q <= 2^L; q >= 2 => L>=1
    // Ensure 2^(L-1) <= q (true when L = ceil(log2 q) and q >= 2)
    uint sh = (L == 0u) ? 0u : (L - 1u);

    // mbar32 = floor(2^(32+sh) / q), saturated to 2^32-1 if overflow.
    ulong num  = (sh >= 32u) ? (~(ulong)0) : ((ulong)1 << (32u + sh));
    ulong mbig = num / (ulong)qv;
    uint  mbar32 = (mbig > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : (uint)mbig;

    device uint *poly = coeffs + (size_t)tgid * nv;

    // Load coefficients & zetas into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    zcache[ltid]          = zetas[ltid];
    zcache[ltid + half_n] = zetas[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length   = half_n;
    uint log2_len = (half_n == 0u) ? 0u : (31u - clz(half_n));
    uint k_start  = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint mask       = length - 1u;
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & mask;
        uint j          = (group_idx << (log2_len + 1u)) + j_in_group;
        uint zeta       = zcache[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        // t = zeta * y < q^2; full 64-bit product.
        ulong tfull = (ulong)zeta * (ulong)y;

        // Barrett: t32 = tfull >> sh, fits in 32 bits.
        uint t32  = (uint)(tfull >> sh);
        uint qest = mulhi(t32, mbar32);
        uint r    = (uint)tfull - qest * qv;
        // r in [0, ~3q); correct down up to 2 times.
        if (r >= qv) r -= qv;
        if (r >= qv) r -= qv;

        // Butterfly (canonical).
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
            kyb_B1: correct, 0.02 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.02 ms, 0.6 Gmodmul/s (u32) (1.1% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.04 ms, 5.5 Gmodmul/s (u32) (10.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0093

## History

- iter  0: compile=OK | correct=True | score=0.0036622358479690953
- iter  1: compile=OK | correct=True | score=0.004894578014846105
- iter  2: compile=OK | correct=True | score=0.007616842172810286
- iter  3: compile=OK | correct=True | score=0.006342152440686771
- iter  4: compile=OK | correct=True | score=0.007407634760317044
- iter  5: compile=OK | correct=False | score=N/A
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.009309802474392625

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
