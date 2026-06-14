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

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid, uint M_32) 
{
    // Phase 1: Threadgroup memory levels (Lengths 128 and 64)
    #pragma unroll
    for (uint lvl = 0; lvl < 2; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7u - lvl;
            uint length = 1u << length_shift;
            uint group_idx = ltid >> length_shift;
            uint j_in_group = ltid & (length - 1u);
            uint j = (group_idx << (length_shift + 1u)) | j_in_group;
            uint zeta = zetas[(1u << lvl) + group_idx];
            
            uint x = a[j];
            uint y = a[j + length];
            
            uint t_val;
            if (USE_32BIT) {
                uint t = y * zeta;
                uint q_est = mulhi(t, M_32);
                uint t_mod = t - q_est * q_val;
                t_val = t_mod >= q_val ? t_mod - q_val : t_mod;
            } else {
                t_val = (uint)(((ulong)zeta * y) % q_val);
            }
            
            uint sum = x + t_val;
            a[j]          = sum >= q_val ? sum - q_val : sum;
            uint diff = x - t_val;
            a[j + length] = x >= t_val ? diff : diff + q_val;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint2 reg = a2[ltid];

    // Phase 2: Register levels (Lengths 32 down to 1)
    #pragma unroll
    for (uint lvl = 2; lvl < 8; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7u - lvl; 
            uint length = 1u << length_shift; 
            
            if (length == 1u) {
                uint group_idx = ltid;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint t_val;
                if (USE_32BIT) {
                    uint t = reg[1] * zeta;
                    uint q_est = mulhi(t, M_32);
                    uint t_mod = t - q_est * q_val;
                    t_val = t_mod >= q_val ? t_mod - q_val : t_mod;
                } else {
                    t_val = (uint)(((ulong)reg[1] * zeta) % q_val);
                }
                
                uint x0 = reg[0];
                uint sum = x0 + t_val;
                reg[0] = sum >= q_val ? sum - q_val : sum;
                uint diff = x0 - t_val;
                reg[1] = x0 >= t_val ? diff : diff + q_val;
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                // Vectorized shuffle and assignments
                uint2 y = simd_shuffle_xor(reg, k);
                
                uint2 a_j     = is_left ? reg : y;
                uint2 a_j_len = is_left ? y : reg;
                
                uint2 t_val;
                if (USE_32BIT) {
                    uint2 t_raw = a_j_len * zeta;
                    uint2 q_est = mulhi(t_raw, uint2(M_32));
                    uint2 t_mod = t_raw - q_est * q_val;
                    t_val.x = t_mod.x >= q_val ? t_mod.x - q_val : t_mod.x;
                    t_val.y = t_mod.y >= q_val ? t_mod.y - q_val : t_mod.y;
                } else {
                    t_val.x = (uint)(((ulong)a_j_len.x * zeta) % q_val);
                    t_val.y = (uint)(((ulong)a_j_len.y * zeta) % q_val);
                }
                
                uint2 sum = a_j + t_val;
                uint2 add_res;
                add_res.x = sum.x >= q_val ? sum.x - q_val : sum.x;
                add_res.y = sum.y >= q_val ? sum.y - q_val : sum.y;
                
                uint2 diff = a_j - t_val;
                uint2 sub_res;
                sub_res.x = a_j.x >= t_val.x ? diff.x : diff.x + q_val;
                sub_res.y = a_j.y >= t_val.y ? diff.y : diff.y + q_val;
                
                reg = is_left ? add_res : sub_res;
            }
        }
    }

    poly2[ltid] = reg;
}

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
    
    uint q_val = q;
    uint n_levels_val = n_levels;

    // Fast 128-bit aligned vector accesses mapped linearly exactly for 256 uints (1024 bytes)
    device uint2 *poly2 = (device uint2 *)(coeffs + (size_t)tgid * 256u);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    // Phase 1: 100% coalesced uint2 load into fast threadgroup memory
    a2[ltid] = poly2[ltid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q_val <= 65535u) {
        uint M_32 = 0xFFFFFFFFu / q_val;
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, M_32);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid, 0);
    }
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 2.1 GB/s (1.0% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 20.0 GB/s (10.0% of 200 GB/s)
  score (gmean of fraction): 0.0087

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

template<bool USE_32BIT>
inline void ntt_process(threadgroup uint* a, device const uint* zetas, 
                        device uint2* poly2, threadgroup uint2* a2,
                        uint q_val, uint n_levels_val, uint ltid) 
{
    // Threadgroup levels (for length = 128 and 64)
    #pragma unroll
    for (uint lvl = 0; lvl < 2; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7 - lvl;
            uint length = 1u << length_shift;
            uint group_idx = ltid >> length_shift;
            uint j_in_group = ltid & (length - 1u);
            uint j = (group_idx << (length_shift + 1u)) | j_in_group;
            uint zeta = zetas[(1u << lvl) + group_idx];
            
            uint x = a[j];
            uint y = a[j + length];
            
            uint t_val;
            if (USE_32BIT) {
                t_val = (zeta * y) % q_val;
            } else {
                t_val = (uint)(((ulong)zeta * y) % q_val);
            }
            
            a[j]          = (x >= q_val - t_val) ? (x + t_val - q_val) : (x + t_val);
            a[j + length] = (x >= t_val) ? (x - t_val) : (x - t_val + q_val);
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint2 reg = a2[ltid];

    // Register levels (for length <= 32)
    #pragma unroll
    for (uint lvl = 2; lvl < 8; ++lvl) {
        if (lvl < n_levels_val) {
            uint length_shift = 7 - lvl; 
            uint length = 1u << length_shift; 
            
            if (length == 1) {
                uint group_idx = ltid;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint t_val;
                if (USE_32BIT) {
                    t_val = (reg[1] * zeta) % q_val;
                } else {
                    t_val = (uint)(((ulong)reg[1] * zeta) % q_val);
                }
                uint x0 = reg[0];
                
                reg[0] = (x0 >= q_val - t_val) ? (x0 + t_val - q_val) : (x0 + t_val);
                reg[1] = (x0 >= t_val) ? (x0 - t_val) : (x0 - t_val + q_val);
            } else {
                uint k = length >> 1u; 
                bool is_left = (ltid & k) == 0u;
                uint group_idx = ltid >> length_shift;
                uint zeta = zetas[(1u << lvl) + group_idx];
                
                uint y0 = simd_shuffle_xor(reg[0], k);
                uint y1 = simd_shuffle_xor(reg[1], k);
                
                uint x0 = reg[0];
                uint x1 = reg[1];
                
                uint a_j0     = is_left ? x0 : y0;
                uint a_j_len0 = is_left ? y0 : x0;
                
                uint a_j1     = is_left ? x1 : y1;
                uint a_j_len1 = is_left ? y1 : x1;
                
                uint t0, t1;
                if (USE_32BIT) {
                    t0 = (a_j_len0 * zeta) % q_val;
                    t1 = (a_j_len1 * zeta) % q_val;
                } else {
                    t0 = (uint)(((ulong)a_j_len0 * zeta) % q_val);
                    t1 = (uint)(((ulong)a_j_len1 * zeta) % q_val);
                }
                
                uint add_res0 = (a_j0 >= q_val - t0) ? (a_j0 + t0 - q_val) : (a_j0 + t0);
                uint sub_res0 = (a_j0 >= t0) ? (a_j0 - t0) : (a_j0 - t0 + q_val);
                reg[0] = is_left ? add_res0 : sub_res0;
                
                uint add_res1 = (a_j1 >= q_val - t1) ? (a_j1 + t1 - q_val) : (a_j1 + t1);
                uint sub_res1 = (a_j1 >= t1) ? (a_j1 - t1) : (a_j1 - t1 + q_val);
                reg[1] = is_left ? add_res1 : sub_res1;
            }
        }
    }

    poly2[ltid] = reg;
}

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
    
    uint q_val = q;
    uint n_levels_val = n_levels;

    device uint2 *poly2 = (device uint2 *)(coeffs + (size_t)tgid * 256u);
    threadgroup uint2 *a2 = (threadgroup uint2 *)a;

    // Phase 1: 100% coalesced uint2 load into fast threadgroup memory
    a2[ltid] = poly2[ltid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Fully unrolled constant-bounded loops.
    if (q_val <= 65535u) {
        ntt_process<true>(a, zetas, poly2, a2, q_val, n_levels_val, ltid);
    } else {
        ntt_process<false>(a, zetas, poly2, a2, q_val, n_levels_val, ltid);
    }
}
```

Incumbent result:
            kyb_B1: correct, 0.02 ms, 0.2 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.01 ms, 2.8 GB/s (1.4% of 200 GB/s)
          kyb_B256: correct, 0.03 ms, 19.9 GB/s (10.0% of 200 GB/s)
  score (gmean of fraction): 0.0104

## History

- iter  4: compile=OK | correct=True | score=0.010446650826301173
- iter  5: compile=OK | correct=True | score=0.008456352773607232
- iter  6: compile=OK | correct=True | score=0.008282150823371895
- iter  7: compile=OK | correct=True | score=0.008832365853156861
- iter  8: compile=OK | correct=True | score=0.0098528484698953
- iter  9: compile=OK | correct=True | score=0.008870383180943292
- iter 10: compile=OK | correct=True | score=0.008748793474799888
- iter 11: compile=FAIL | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
