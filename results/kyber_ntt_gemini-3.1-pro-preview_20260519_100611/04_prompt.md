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

struct ZetaW {
    uint zeta;
    uint W;
};

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

    threadgroup uint a[256];
    threadgroup ZetaW tg_ZW[256];
    threadgroup uint tg_M_lo;
    threadgroup uint tg_M_hi;

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Thread 0 computes the 64-bit reciprocal for the runtime modulus
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        tg_M_lo = (uint)M;
        tg_M_hi = (uint)(M >> 32);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint M_lo = tg_M_lo;
    uint M_hi = tg_M_hi;

    // Precompute Barrett fractional multipliers (W) for all utilized zetas
    uint num_zetas = 1u << n_levels;
    for (uint i = ltid; i < num_zetas; i += half_n) {
        uint z = zetas[i];
        tg_ZW[i].zeta = z;
        // Exactly computes the upper 32 bits of (z * M)
        tg_ZW[i].W    = z * M_hi + mulhi(z, M_lo);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        
        ZetaW zw  = tg_ZW[k_start + group_idx];
        uint zeta = zw.zeta;
        uint W    = zw.W;

        uint x = a[j];
        uint y = a[j + length];
        
        // Fast fractional Barrett reduction
        uint k = mulhi(y, W);
        ulong rem64 = (ulong)y * zeta - (ulong)k * q;
        
        // The fractional error bounds guarantee the remainder is strictly < 3q.
        // Two conditional subtractions reduce it exactly to [0, q - 1].
        rem64 = select(rem64, rem64 - q, rem64 >= q);
        rem64 = select(rem64, rem64 - q, rem64 >= q);
        uint rem = (uint)rem64;

        // Canonical butterfly
        uint sum = x + rem;
        a[j]          = select(sum, sum - q, sum >= q || sum < x);
        a[j + length] = select(q - (rem - x), x - rem, x >= rem);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Writeback canonical values
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.07 ms, 3.4 Gmodmul/s (u32) (6.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0067

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

struct BarrettCtx {
    uint q;
    uint y_lo;
    uint y_hi;
};

// Fast 64-bit to 32-bit Barrett reduction using 32-bit multiplications.
// Computes (a * b) % q exactly, given that (a * b) < q^2 and q <= 8380417.
inline uint mod_mul_fast(uint a, uint b, thread const BarrettCtx& ctx) {
    ulong X = (ulong)a * b;
    uint x_lo = (uint)X;
    uint x_hi = (uint)(X >> 32);
    
    uint p00_hi = mulhi(x_lo, ctx.y_lo);
    ulong p01 = (ulong)x_lo * ctx.y_hi;
    ulong p10 = (ulong)x_hi * ctx.y_lo;
    ulong p11 = (ulong)x_hi * ctx.y_hi;
    
    ulong mid = p01 + p00_hi;
    ulong mid2 = p10 + (uint)mid;
    
    // Top 64 bits of (X * M)
    uint k = (uint)(p11 + (mid >> 32) + (mid2 >> 32));
    
    // Remainder
    uint r = x_lo - k * ctx.q;
    return (r >= ctx.q) ? (r - ctx.q) : r;
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
    constant uint     &q           [[buffer(2)]],
    constant uint     &n           [[buffer(3)]],
    constant uint     &n_levels    [[buffer(4)]],
    constant uint     &batch       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;

    threadgroup uint a[256];
    threadgroup BarrettCtx tg_ctx;

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    
    // Thread 0 computes the Barrett magic constants to save 64-bit ALU divisions
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        tg_ctx.q = q;
        tg_ctx.y_lo = (uint)M;
        tg_ctx.y_hi = (uint)(M >> 32);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load into fast register context
    BarrettCtx ctx = tg_ctx;

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        // Bitwise operations replace integer division/modulo
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        uint zeta       = zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        uint t = mod_mul_fast(zeta, y, ctx);

        a[j]          = mod_add(x, t, ctx.q);
        a[j + length] = mod_sub(x, t, ctx.q);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Writeback
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.06 ms, 3.7 Gmodmul/s (u32) (6.9% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0070

## History

- iter  0: compile=OK | correct=True | score=0.003666302430838368
- iter  1: compile=OK | correct=True | score=0.0070393467115362
- iter  2: compile=OK | correct=True | score=0.005380907353825932
- iter  3: compile=OK | correct=True | score=0.006727627497053976

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
