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

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// 64-bit high multiply: high 64 bits of a*b.
inline ulong mulhi64(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFFul;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFul;
    ulong b_hi = b >> 32;

    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;

    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFul) + (hl & 0xFFFFFFFFul);
    ulong hi  = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    return hi;
}

// Barrett-ish: given T = zeta*y with T < q^2 < 2^62 (q<2^31), reduce mod q.
// m64 = floor(2^64 / q). Then q_est = mulhi64(T, m64). r = T - q_est*q.
// r in [0, 2q); correct with one conditional subtract.
inline uint barrett_reduce(ulong T, uint q, ulong m64) {
    ulong q_est = mulhi64(T, m64);
    ulong r = T - q_est * (ulong)q;
    // r < 2q (in fact r < q+something small)
    uint ru = (uint)r;
    if (ru >= q) ru -= q;
    return ru;
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
    threadgroup uint zlocal[Z_MAX];

    uint qreg   = q;
    uint nreg   = n;
    uint nlv    = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;

    // Precompute Barrett constant: m64 = floor(2^64 / q).
    // 2^64 / q computed as ((2^64 - 1) / q) plus possible +1 correction.
    // Use unsigned long division directly.
    ulong m64;
    {
        // Compute floor(2^64 / q). Since 2^64 overflows, use:
        // m = (0 - q) / q + 1  ==  floor((2^64 - q)/q) + 1 = floor(2^64/q)  when q | 2^64? no.
        // Safer: m = floor((2^64 - 1)/q); if ((m+1)*q == 0 in 64-bit) m += 1 (when q divides 2^64, impossible for odd q>1; for q=power of two could matter but q is prime).
        // We use: m64 = (~0ul - (~0ul % q)) / q ... but simplest:
        ulong qq = (ulong)qreg;
        ulong max64 = ~0ul;
        ulong r = max64 % qq;
        m64 = max64 / qq;
        if (r + 1ul == qq) m64 += 1ul; // then 2^64 is exactly divisible
    }

    // Cooperatively load zetas into threadgroup memory (raw, not Montgomery).
    {
        uint i1 = ltid;
        if (i1 < zcount) zlocal[i1] = zetas[i1];
        uint i2 = ltid + half_n;
        if (i2 < zcount) zlocal[i2] = zetas[i2];
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load polynomial.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zv         = zlocal[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = barrett_reduce((ulong)zv * (ulong)y, qreg, m64);

        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 1.8 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.14 ms, 4.6 GB/s (2.3% of 200 GB/s)
  score (gmean of fraction): 0.0049

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;
constexpr constant uint Z_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Compute q^{-1} mod 2^32 via Newton iteration (q must be odd).
inline uint compute_qinv(uint q) {
    uint x = q;                       // 3-bit inverse: q ≡ q^{-1} mod 8
    x = x * (2u - q * x);             // 6 bits
    x = x * (2u - q * x);             // 12 bits
    x = x * (2u - q * x);             // 24 bits
    x = x * (2u - q * x);             // 48 -> 32 bits
    x = x * (2u - q * x);             // safety
    return x;
}

// Montgomery reduction of T < q*2^32. Returns T*2^{-32} mod q in [0, q).
// Requires q < 2^31.
inline uint mont_reduce(ulong T, uint q, uint qinv_neg) {
    uint T_lo = (uint)T;
    uint T_hi = (uint)(T >> 32);
    uint m = T_lo * qinv_neg;
    uint mq_hi = mulhi(m, q);
    uint mq_lo = m * q;
    uint sum_lo = T_lo + mq_lo;
    uint carry = (sum_lo < T_lo) ? 1u : 0u;
    uint t = T_hi + mq_hi + carry;
    if (t >= q) t -= q;
    return t;
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
    threadgroup uint zeta_mont[Z_MAX];

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;

    // Each thread computes Montgomery constants locally (cheap, no barrier needed).
    uint qinv = compute_qinv(qreg);
    uint qinv_neg = 0u - qinv;

    // R = 2^32 mod q.
    uint Rmod;
    {
        ulong r = ((ulong)0xFFFFFFFFul) % (ulong)qreg;
        r += 1ul;
        if (r >= (ulong)qreg) r -= (ulong)qreg;
        Rmod = (uint)r;
    }

    // Cooperatively precompute zeta_mont[k] = (zetas[k] * R) mod q,
    // i.e. Montgomery form of zetas[k]. zcount <= 256, threads = half_n (<=128).
    {
        ulong Q = (ulong)qreg;
        uint i1 = ltid;
        if (i1 < zcount) {
            ulong v = (ulong)zetas[i1] * (ulong)Rmod;
            zeta_mont[i1] = (uint)(v % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            ulong v = (ulong)zetas[i2] * (ulong)Rmod;
            zeta_mont[i2] = (uint)(v % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load polynomial into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        // Read both operands into registers first.
        uint x = a[j];
        uint y = a[j + length];

        // t = (zeta * y) mod q via Montgomery: mont_reduce(y * zeta_mont) = y*zeta mod q.
        uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);

        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        // Full barrier before writing so all reads of this level are done.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
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
           kyb_B16: correct, 0.02 ms, 1.7 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.08 ms, 8.6 GB/s (4.3% of 200 GB/s)
  score (gmean of fraction): 0.0058

## History

- iter  1: compile=OK | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.0027103400175003674
- iter  3: compile=OK | correct=True | score=0.002765701390740252
- iter  4: compile=OK | correct=True | score=0.003552312891167097
- iter  5: compile=OK | correct=False | score=N/A
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.00581370010912664
- iter  8: compile=OK | correct=True | score=0.004895909002653772

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
