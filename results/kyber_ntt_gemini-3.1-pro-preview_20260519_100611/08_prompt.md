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

    // Fast integer division precomputation
    ulong M = 0xFFFFFFFFFFFFFFFFull / q;
    uint M_lo = (uint)M;
    uint M_hi = (uint)(M >> 32);

    threadgroup uint shared_zetas[256];
    threadgroup uint shared_W[256];
    threadgroup uint a[256];

    uint half_n = n >> 1u;
    
    // Cooperatively load zetas and precompute fractional Barrett multipliers W
    uint num_zetas = 1u << n_levels;
    for (uint i = ltid; i < num_zetas; i += half_n) {
        uint z = zetas[i];
        shared_zetas[i] = z;
        shared_W[i]     = z * M_hi + mulhi(z, M_lo);
    }

    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint length  = half_n;
    uint shift   = 31u - clz(half_n);
    uint k_start = 1u;
    uint level   = 0u;

    if (n >= 64u) {
        // Phase 1: Shared memory NTT for length >= 32
        // Avoids cross-SIMD shuffles, zero bank conflicts for length >= 32
        while (length >= 32u && level < n_levels) {
            uint group_idx  = ltid >> shift;
            uint j_in_group = ltid & (length - 1u);
            uint j          = (group_idx << (shift + 1u)) | j_in_group;
            
            uint zeta = shared_zetas[k_start + group_idx];
            uint W    = shared_W[k_start + group_idx];

            uint x = a[j];
            uint y = a[j + length];
            
            uint k_barrett = mulhi(y, W);
            uint r = y * zeta - k_barrett * q;
            r = select(r, r - q, r >= q);

            uint sum = x + r;
            a[j]          = select(sum, sum - q, sum >= q);
            
            uint diff = x - r;
            a[j + length] = select(diff + q, diff, x >= r);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            k_start <<= 1u;
            length  >>= 1u;
            shift   -= 1u;
            level++;
        }

        // Phase 2: Register-based NTT for length <= 16
        // Completely eliminates threadgroup barriers and bank conflicts
        uint lane_id = ltid & 31u;
        uint simd_id = ltid >> 5u;
        
        uint base = simd_id * 64u;
        uint j_u = base + lane_id;
        uint j_v = base + 32u + lane_id;
        
        uint u = a[j_u];
        uint v = a[j_v];
        
        #pragma unroll
        for (uint iter = 0; iter < 5u; ++iter) {
            if (level >= n_levels) break;
            
            uint L = n >> (level + 1u);
            uint shift_2L = 31u - clz(n) - level;
            
            bool is_right = (j_u & L) != 0u;
            
            // Process element u
            uint other_u = simd_shuffle_xor(u, L);
            uint x_u = is_right ? other_u : u;
            uint y_u = is_right ? u : other_u;
            
            uint zeta_idx_u  = (1u << level) + (j_u >> shift_2L);
            uint zeta_u      = shared_zetas[zeta_idx_u];
            uint W_u         = shared_W[zeta_idx_u];
            
            uint k_barrett_u = mulhi(y_u, W_u);
            uint r_u         = y_u * zeta_u - k_barrett_u * q;
            r_u              = select(r_u, r_u - q, r_u >= q);
            
            uint sum_u   = x_u + r_u;
            uint x_new_u = select(sum_u, sum_u - q, sum_u >= q);
            uint diff_u  = x_u - r_u;
            uint y_new_u = select(diff_u + q, diff_u, x_u >= r_u);
            
            u = is_right ? y_new_u : x_new_u;

            // Process element v (reuses bounds math from u)
            uint other_v = simd_shuffle_xor(v, L);
            uint x_v = is_right ? other_v : v;
            uint y_v = is_right ? v : other_v;
            
            uint zeta_idx_v  = zeta_idx_u + (32u >> shift_2L);
            uint zeta_v      = shared_zetas[zeta_idx_v];
            uint W_v         = shared_W[zeta_idx_v];
            
            uint k_barrett_v = mulhi(y_v, W_v);
            uint r_v         = y_v * zeta_v - k_barrett_v * q;
            r_v              = select(r_v, r_v - q, r_v >= q);
            
            uint sum_v   = x_v + r_v;
            uint x_new_v = select(sum_v, sum_v - q, sum_v >= q);
            uint diff_v  = x_v - r_v;
            uint y_new_v = select(diff_v + q, diff_v, x_v >= r_v);
            
            v = is_right ? y_new_v : x_new_v;
            
            level++;
        }

        // Direct writeback from registers (perfectly coalesced)
        poly[j_u] = u;
        poly[j_v] = v;

    } else {
        // Fallback Phase 1 for extremely small n (n < 64)
        while (level < n_levels) {
            uint group_idx  = ltid >> shift;
            uint j_in_group = ltid & (length - 1u);
            uint j          = (group_idx << (shift + 1u)) | j_in_group;
            
            uint zeta = shared_zetas[k_start + group_idx];
            uint W    = shared_W[k_start + group_idx];

            uint x = a[j];
            uint y = a[j + length];
            
            uint k_barrett = mulhi(y, W);
            uint r = y * zeta - k_barrett * q;
            r = select(r, r - q, r >= q);

            uint sum = x + r;
            a[j]          = select(sum, sum - q, sum >= q);
            
            uint diff = x - r;
            a[j + length] = select(diff + q, diff, x >= r);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            k_start <<= 1u;
            length  >>= 1u;
            shift   -= 1u;
            level++;
        }
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
    }
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.4 Gmodmul/s (u32) (0.8% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.15 ms, 1.6 Gmodmul/s (u32) (3.0% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0050

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

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
    threadgroup uint shared_M[2];

    uint half_n = n >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n;

    // Load polynomial elements into threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Thread 0 evaluates the expensive 64-bit division once per threadgroup
    if (ltid == 0) {
        ulong M = 0xFFFFFFFFFFFFFFFFull / q;
        shared_M[0] = (uint)M;
        shared_M[1] = (uint)(M >> 32);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint M_lo = shared_M[0];
    uint M_hi = shared_M[1];

    uint length = half_n;
    uint shift = 31u - clz(half_n);
    uint k_start = 1u;
    
    for (uint level = 0u; level < n_levels; ++level) {
        // Compute indices bitwise to avoid division/modulo inside the loop
        uint group_idx  = ltid >> shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (shift + 1u)) | j_in_group;
        
        // Compute fractional Barrett multiplier W for the given zeta
        uint zeta = zetas[k_start + group_idx];
        uint W    = zeta * M_hi + mulhi(zeta, M_lo);

        uint x = a[j];
        uint y = a[j + length];
        
        // 32-bit fractional Barrett reduction.
        // It is mathematically guaranteed that the error in the quotient is <= 1,
        // bounding the remainder strictly < 2q, completely eliminating 64-bit math.
        uint k_barrett = mulhi(y, W);
        uint r = y * zeta - k_barrett * q;
        r = select(r, r - q, r >= q);

        // Canonical butterfly operations
        uint sum = x + r;
        a[j]          = select(sum, sum - q, sum >= q);
        
        uint diff = x - r;
        a[j + length] = select(diff + q, diff, x >= r);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        length  >>= 1u;
        shift   -= 1u;
    }

    // Writeback canonical values perfectly coalesced
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.06 ms, 3.9 Gmodmul/s (u32) (7.2% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0072

## History

- iter  0: compile=OK | correct=True | score=0.003666302430838368
- iter  1: compile=OK | correct=True | score=0.0070393467115362
- iter  2: compile=OK | correct=True | score=0.005380907353825932
- iter  3: compile=OK | correct=True | score=0.006727627497053976
- iter  4: compile=OK | correct=True | score=0.005499115050668452
- iter  5: compile=OK | correct=True | score=0.0053854041986306795
- iter  6: compile=OK | correct=True | score=0.007172281369140307
- iter  7: compile=OK | correct=True | score=0.005041192744476254

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
