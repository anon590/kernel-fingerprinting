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

inline uint mont_mul(uint a, uint b, uint q, uint q_inv) {
    uint P_lo = a * b;
    uint P_hi = mulhi(a, b);
    uint Q_val = P_lo * q_inv;
    uint M_hi = mulhi(Q_val, q);
    
    ulong T = (ulong)P_hi + M_hi + (P_lo != 0u ? 1u : 0u);
    return (uint)(T >= (ulong)q ? T - q : T);
}

inline uint mod_add_safe(uint a, uint b, uint q) {
    uint sum = a + b;
    return sum - ((sum >= q || sum < a) ? q : 0u);
}

inline uint mod_sub_safe(uint a, uint b, uint q) {
    uint diff = a - b;
    return diff + ((a < b) ? q : 0u);
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

    uint q_val = q;

    // Compute q_inv = -q^{-1} mod 2^32 using Newton-Raphson
    uint inv = q_val;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    uint q_inv = 0u - inv;

    // Compute R^2 mod q
    uint r = (0xFFFFFFFFu % q_val) + 1u;
    r = (r == q_val) ? 0u : r;
    uint r2 = (uint)(((ulong)r * r) % q_val);

    threadgroup uint shared_val[256];
    threadgroup uint zeta_mont[256];

    uint half_n = n >> 1u;
    uint num_zetas = 1u << n_levels;

    // Collaboratively load and pre-convert zetas
    for (uint i = ltid; i < num_zetas; i += half_n) {
        zeta_mont[i] = mont_mul(zetas[i], r2, q_val, q_inv);
    }

    device uint *poly = coeffs + (size_t)tgid * n;

    // Initial load: each thread owns two elements spanning exactly `half_n` apart
    uint X = poly[ltid];
    uint Y = poly[ltid + half_n];
    
    // Track original positions so the final writeback handles the permutation dynamically
    uint idx_X = ltid;
    uint idx_Y = ltid + half_n;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_L = 31u - clz(half_n);
    uint k_start = 1u;
    uint L = half_n;

    for (uint level = 0u; level < n_levels; ++level) {
        if (level > 0u) {
            bool is_top = (ltid & L) == 0u;
            uint partner = is_top ? (ltid + L) : (ltid - L);

            if (L >= 32u) {
                // Cross-SIMD exchange using threadgroup memory
                shared_val[ltid] = is_top ? Y : X;
                shared_val[128u + ltid] = is_top ? idx_Y : idx_X;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (is_top) {
                    Y = shared_val[partner];
                    idx_Y = shared_val[128u + partner];
                } else {
                    X = shared_val[partner];
                    idx_X = shared_val[128u + partner];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                // Intra-SIMD exchange avoiding threadgroup memory entirely
                ushort lane = (ushort)(partner & 31u);
                uint new_Y = simd_shuffle(X, lane);
                uint new_X = simd_shuffle(Y, lane);
                uint new_idx_Y = simd_shuffle(idx_X, lane);
                uint new_idx_X = simd_shuffle(idx_Y, lane);
                
                if (is_top) {
                    Y = new_Y;
                    idx_Y = new_idx_Y;
                } else {
                    X = new_X;
                    idx_X = new_idx_X;
                }
            }
        }

        uint group_idx = ltid >> log2_L;
        uint z_mont = zeta_mont[k_start + group_idx];

        // Butterfly strictly operates on local registers
        uint t = mont_mul(Y, z_mont, q_val, q_inv);

        uint new_X_val = mod_add_safe(X, t, q_val);
        uint new_Y_val = mod_sub_safe(X, t, q_val);

        X = new_X_val;
        Y = new_Y_val;

        log2_L--;
        k_start <<= 1u;
        L >>= 1u;
    }

    // Restore to strictly in-place, ordered array in threadgroup memory
    shared_val[idx_X] = X;
    shared_val[idx_Y] = Y;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 100% coalesced write to device
    poly[ltid] = shared_val[ltid];
    poly[ltid + half_n] = shared_val[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.1 GB/s (0.0% of 200 GB/s)
           kyb_B16: correct, 0.03 ms, 1.6 GB/s (0.8% of 200 GB/s)
          kyb_B256: correct, 0.08 ms, 8.2 GB/s (4.1% of 200 GB/s)
  score (gmean of fraction): 0.0054

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

// Overflow-safe Montgomery reduction for any q <= 2^32 - 1
inline uint mont_mul(uint a, uint b, uint q, uint q_inv) {
    uint P_lo = a * b;
    uint P_hi = mulhi(a, b);
    
    uint Q = P_lo * q_inv;
    uint M_hi = mulhi(Q, q);
    
    // Use ulong to prevent wrap-around if q is extremely close to 2^32 - 1
    ulong T = (ulong)P_hi + M_hi + (P_lo != 0u ? 1u : 0u);
    return (uint)(T >= (ulong)q ? T - q : T);
}

// 32-bit branchless modular addition
inline uint mod_add_safe(uint a, uint b, uint q) {
    uint sum = a + b;
    return sum - ((sum >= q || sum < a) ? q : 0u);
}

// 32-bit branchless modular subtraction
inline uint mod_sub_safe(uint a, uint b, uint q) {
    uint diff = a - b;
    return diff + ((a < b) ? q : 0u);
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

    uint q_val = q;

    // Compute q_inv = -q^{-1} mod 2^32 using Newton-Raphson
    uint inv = q_val;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    inv *= 2u - q_val * inv;
    uint q_inv = 0u - inv;

    // Compute R^2 mod q for converting twiddles to Montgomery form (R = 2^32)
    uint r = (0xFFFFFFFFu % q_val) + 1u;
    r = (r == q_val) ? 0u : r;
    uint r2 = (uint)(((ulong)r * r) % q_val);

    threadgroup uint a[256];
    threadgroup uint zeta_mont[256];

    uint half_n = n >> 1u;
    uint num_zetas = 1u << n_levels;

    // Collaboratively load and pre-convert zetas to Montgomery form
    for (uint i = ltid; i < num_zetas; i += half_n) {
        zeta_mont[i] = mont_mul(zetas[i], r2, q_val, q_inv);
    }

    // Coalesced loads into threadgroup memory
    device uint *poly = coeffs + (size_t)tgid * n;
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_len = 31u - clz(half_n);
    uint k_start  = 1u;
    
    // Cooley-Tukey NTT
    for (uint level = 0u; level < n_levels; ++level) {
        uint length = 1u << log2_len;
        
        // Fast bitwise indexing mapped to thread IDs
        uint group_idx  = ltid >> log2_len;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (log2_len + 1u)) | j_in_group;
        
        uint z_mont = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint t = mont_mul(y, z_mont, q_val, q_inv);

        a[j]          = mod_add_safe(x, t, q_val);
        a[j + length] = mod_sub_safe(x, t, q_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        log2_len--;
        k_start <<= 1u;
    }

    // Coalesced writeback
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Incumbent result:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 1.8 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.06 ms, 11.1 GB/s (5.6% of 200 GB/s)
  score (gmean of fraction): 0.0066

## History

- iter  0: compile=OK | correct=True | score=0.002808065069115763
- iter  1: compile=OK | correct=True | score=0.0041073575201597415
- iter  2: compile=OK | correct=True | score=0.0066307102238388725
- iter  3: compile=OK | correct=True | score=0.005421380221671123

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
