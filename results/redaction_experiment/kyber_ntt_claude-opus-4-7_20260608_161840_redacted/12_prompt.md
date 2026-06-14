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

inline uint compute_qinv(uint q) {
    uint x = q;
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    x = x * (2u - q * x);
    return x;
}

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

// Butterfly given two register values x,y and a montgomery-form zeta zm.
inline void bfly(thread uint &x, thread uint &y, uint zm, uint q, uint qinv_neg) {
    uint t = mont_reduce((ulong)y * (ulong)zm, q, qinv_neg);
    uint xpt = mod_add(x, t, q);
    uint xmt = mod_sub(x, t, q);
    x = xpt;
    y = xmt;
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

    // Cooperatively precompute zetas in Montgomery form.
    {
        ulong Q = (ulong)qreg;
        uint i1 = ltid;
        if (i1 < zcount) {
            zeta_mont[i1] = (uint)(((ulong)zetas[i1] * (ulong)Rmod) % Q);
        }
        uint i2 = ltid + half_n;
        if (i2 < zcount) {
            zeta_mont[i2] = (uint)(((ulong)zetas[i2] * (ulong)Rmod) % Q);
        }
    }

    device uint *poly = coeffs + (size_t)tgid * nreg;

    // Load polynomial into threadgroup memory.
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint k_start = 1u;

    // ---- Phase 1: threadgroup-memory levels (stride > 16).
    // For n=256, half_n=128. Levels with length > 16 are levels where
    // length in {128, 64, 32}. After those, length = 16, and stride 16
    // butterflies map onto simdgroups of width 32 (each simd owns 32 consecutive
    // ltids, which cover 32 consecutive coeffs in our layout).
    //
    // We'll do threadgroup-mem levels while length > 16.
    while (length > 16u) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
    }

    // ---- Phase 2: simdgroup levels (length <= 16).
    // Each thread now holds two coefficients in registers, packed so that:
    //   thread ltid (with ltid in [0, half_n)) handles butterfly between
    //   coeff index j and j+length where length<=16.
    //
    // Strategy: load both coefficients into registers, then do remaining
    // levels using simd_shuffle_xor to exchange the partner coefficient.
    //
    // We re-index: each ltid corresponds to a butterfly pair at the current
    // level. After loading, we transition to a "one coefficient per thread"
    // model for the simd phase. There are n coefficients = half_n*2.
    // Use ltid for low half and ltid+half_n for high half... but for simd
    // shuffle we need partners within a 32-lane simdgroup.
    //
    // Simpler: continue with the "two coeffs per thread" model. At each
    // remaining level, thread ltid still computes its butterfly. The values
    // a[j] and a[j+length] are already exchanged via threadgroup memory in
    // the last phase. We continue exactly that pattern but skip barriers
    // where the partner lies within the same simdgroup.
    //
    // For length <= 16: each butterfly group (2*length coeffs) fits in a
    // contiguous block of 'length' threads (one simdgroup if length<=16).
    // The partner read a[j+length] is from a different ltid, but within
    // the same simdgroup. Threadgroup memory writes/reads are coherent
    // within a simdgroup without an explicit barrier IF we use
    // simdgroup_barrier(mem_threadgroup).
    while (length > 0u) {
        uint group_idx  = ltid / length;
        uint j_in_group = ltid - group_idx * length;
        uint j          = (group_idx << 1u) * length + j_in_group;
        uint zm         = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];

        uint t = mont_reduce((ulong)y * (ulong)zm, qreg, qinv_neg);
        uint xpt = mod_add(x, t, qreg);
        uint xmt = mod_sub(x, t, qreg);

        simdgroup_barrier(mem_flags::mem_threadgroup);
        a[j]          = xpt;
        a[j + length] = xmt;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        if (length == 1u) break;
        length  >>= 1u;
    }

    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: INCORRECT (bit_exact=256)
  fail_reason: correctness failed at size kyb_B1: bit_exact=256

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

- iter  4: compile=OK | correct=True | score=0.003552312891167097
- iter  5: compile=OK | correct=False | score=N/A
- iter  6: compile=OK | correct=False | score=N/A
- iter  7: compile=OK | correct=True | score=0.00581370010912664
- iter  8: compile=OK | correct=True | score=0.004895909002653772
- iter  9: compile=OK | correct=True | score=0.005171652562248
- iter 10: compile=OK | correct=True | score=0.005720496741663505
- iter 11: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
