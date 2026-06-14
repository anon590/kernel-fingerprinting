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

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Montgomery reduction: given T < q * 2^32, returns T * 2^{-32} mod q in [0, q).
// Requires qinv_neg = (-q^{-1}) mod 2^32, i.e. q * qinv_neg = -1 mod 2^32.
inline uint mont_reduce(ulong T, uint q, uint qinv_neg) {
    uint T_lo = (uint)T;
    uint m = T_lo * qinv_neg;             // m = (T mod 2^32) * (-q^{-1}) mod 2^32
    // t = (T + m*q) / 2^32; since T + m*q is divisible by 2^32.
    // High 32 bits of (T + m*q): high(T) + high(m*q) + carry.
    uint T_hi = (uint)(T >> 32);
    uint mq_hi = mulhi(m, q);
    uint mq_lo = m * q;
    // carry from T_lo + mq_lo
    uint sum_lo = T_lo + mq_lo;
    uint carry = (sum_lo < T_lo) ? 1u : 0u;
    uint t = T_hi + mq_hi + carry;
    // t in [0, 2q); reduce.
    if (t >= q) t -= q;
    return t;
}

// Modular multiply: a * b mod q, using Montgomery on b that is ALREADY in Mont form.
// result = a * b * 2^{-32} mod q. If b_mont = b * 2^32 mod q, this yields a*b mod q.
inline uint mod_mul_mont(uint a, uint b_mont, uint q, uint qinv_neg) {
    return mont_reduce((ulong)a * (ulong)b_mont, q, qinv_neg);
}

// Compute (-q^{-1}) mod 2^32 via Newton iteration (5 iters give 32 valid bits for odd q).
inline uint compute_qinv_neg(uint q) {
    // Start: x = q; x*q == 1 mod 2^3 since q odd? Actually q*q == 1 mod 8 for odd q.
    uint x = q;                  // q*q == 1 mod 8
    x = x * (2u - q * x);        // mod 16
    x = x * (2u - q * x);        // mod 256
    x = x * (2u - q * x);        // mod 2^16
    x = x * (2u - q * x);        // mod 2^32
    // Now x = q^{-1} mod 2^32; we want -q^{-1}.
    return 0u - x;
}

// Compute R = 2^32 mod q.
inline uint compute_R(uint q) {
    // 2^32 mod q = ((2^32 - 1) mod q) + 1, then reduce.
    ulong r = ((ulong)0xFFFFFFFFul) % (ulong)q;
    r += 1ul;
    if (r >= (ulong)q) r -= (ulong)q;
    return (uint)r;
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
    threadgroup uint zeta_mont_tg[N_MAX];   // precomputed Montgomery-form zetas
    threadgroup uint q_share;
    threadgroup uint qinv_share;
    threadgroup uint R_share;

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;
    uint zcount = 1u << nlv;   // number of zetas

    // Thread 0 computes Montgomery constants once per threadgroup.
    if (ltid == 0u) {
        q_share = qreg;
        qinv_share = compute_qinv_neg(qreg);
        R_share = compute_R(qreg);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qinv_neg = qinv_share;
    uint Rmod = R_share;

    // Cooperatively convert zetas to Montgomery form into threadgroup memory.
    // zcount <= 256, threads = n/2 = up to 128. Two slots each suffices.
    {
        uint i = ltid;
        if (i < zcount) {
            uint z = zetas[i];
            zeta_mont_tg[i] = mod_mul_mont(z, Rmod, qreg, qinv_neg) == 0u && z != 0u
                ? mod_mul_mont(z, Rmod, qreg, qinv_neg)
                : mod_mul_mont(z, Rmod, qreg, qinv_neg);
            // Simpler: zeta_mont = z * R mod q = mont_reduce(z * R * 2^32)? No.
            // mod_mul_mont(z, R, q, qinv_neg) = z * R * 2^{-32} mod q = z mod q. Wrong!
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            uint z = zetas[i2];
            (void)z;
        }
    }
    // The above is wrong; recompute properly below.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Correct conversion: zeta_mont = (z * R) mod q. Use 64-bit multiply.
    {
        ulong Q = (ulong)qreg;
        uint i = ltid;
        if (i < zcount) {
            ulong v = (ulong)zetas[i] * (ulong)Rmod;
            zeta_mont_tg[i] = (uint)(v % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            ulong v = (ulong)zetas[i2] * (ulong)Rmod;
            zeta_mont_tg[i2] = (uint)(v % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta_m     = zeta_mont_tg[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        // t = (zeta * y) mod q via Montgomery: mont_reduce(y * zeta_mont).
        uint t = mod_mul_mont(y, zeta_m, qreg, qinv_neg);

        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        if (length > 16u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            a[j]          = xpt;
            a[j + length] = xmt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            a[j]          = xpt;
            a[j + length] = xmt;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.1 GB/s (0.0% of 200 GB/s)
           kyb_B16: correct, 0.03 ms, 1.6 GB/s (0.8% of 200 GB/s)
          kyb_B256: INCORRECT (bit_exact=5789)
  fail_reason: correctness failed at size kyb_B256: bit_exact=5789

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

inline uint mod_add(uint a, uint b, uint q) {
    uint t = a + b;
    return (t >= q) ? (t - q) : t;
}

inline uint mod_sub(uint a, uint b, uint q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// Barrett-like reduction using a precomputed 64-bit reciprocal m = floor(2^64 / q).
// Given x = a*b with a,b < q < 2^32, x < 2^64.
// Estimate quotient: qh = mulhi64(x, m); then r = x - qh*q; correct r if r >= q.
inline uint mod_mul_barrett(uint a, uint b, uint q, ulong m) {
    ulong x = (ulong)a * (ulong)b;
    // mulhi for 64x64: use Metal's mulhi only exists for 32-bit. Compute via split.
    // x_hi*m + x_lo*m high half.
    ulong x_hi = x >> 32;
    ulong x_lo = x & 0xFFFFFFFFul;
    ulong m_hi = m >> 32;
    ulong m_lo = m & 0xFFFFFFFFul;
    ulong ll = x_lo * m_lo;
    ulong lh = x_lo * m_hi;
    ulong hl = x_hi * m_lo;
    ulong hh = x_hi * m_hi;
    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFul) + (hl & 0xFFFFFFFFul);
    ulong qh = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    ulong r = x - qh * (ulong)q;
    // r may be in [0, 2q); correct twice to be safe (Barrett can be off by up to ~2).
    if (r >= (ulong)q) r -= (ulong)q;
    if (r >= (ulong)q) r -= (ulong)q;
    return (uint)r;
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

    uint qreg = q;
    uint nreg = n;
    uint nlv  = n_levels;
    uint half_n = nreg >> 1u;

    // Precomputed Barrett reciprocal m = floor(2^64 / q).
    // For q != 1, compute as: m = (~0ull) / q  (gives floor((2^64 - 1)/q));
    // adjust: if (~0ull) - m*q == q-1 then floor(2^64/q) = m+1, else m.
    ulong neg1 = (ulong)0xFFFFFFFFFFFFFFFFul;
    ulong m = neg1 / (ulong)qreg;
    ulong rem = neg1 - m * (ulong)qreg;
    if (rem == (ulong)(qreg - 1u)) m = m + 1ul;

    device uint *poly = coeffs + (size_t)tgid * nreg;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    for (uint level = 0u; level < nlv; ++level) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_barrett(zeta, y, qreg, m);

        a[j]          = mod_add(x, t, qreg);
        a[j + length] = mod_sub(x, t, qreg);

        // Once length <= 16, the pairs (j, j+length) for every active
        // thread all lie within a single 32-lane simdgroup span of
        // threadgroup memory, so a simd-level barrier suffices.
        if (length > 16u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        k_start <<= 1u;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.03 ms, 0.1 GB/s (0.0% of 200 GB/s)
           kyb_B16: correct, 0.03 ms, 1.2 GB/s (0.6% of 200 GB/s)
          kyb_B256: correct, 0.17 ms, 4.0 GB/s (2.0% of 200 GB/s)
  score (gmean of fraction): 0.0036

## History

- iter  0: compile=OK | correct=True | score=0.002793329448886116
- iter  1: compile=OK | correct=False | score=N/A
- iter  2: compile=OK | correct=True | score=0.0027103400175003674
- iter  3: compile=OK | correct=True | score=0.002765701390740252
- iter  4: compile=OK | correct=True | score=0.003552312891167097
- iter  5: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
