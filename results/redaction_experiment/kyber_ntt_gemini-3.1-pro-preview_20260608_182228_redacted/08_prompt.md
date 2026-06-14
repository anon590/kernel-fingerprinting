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

// Overflow-safe Montgomery reduction for any q <= 2^32 - 1
inline uint mont_mul(uint a, uint b, uint q, uint q_inv) {
    uint P_lo = a * b;
    uint P_hi = mulhi(a, b);
    
    uint Q = P_lo * q_inv;
    uint M_hi = mulhi(Q, q);
    
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
    uint n_levels_val = n_levels;

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

    uint num_zetas = 1u << n_levels_val;

    // 1. Collaboratively load and pre-convert zetas to Montgomery form
    if (ltid < num_zetas) {
        zeta_mont[ltid] = mont_mul(zetas[ltid], r2, q_val, q_inv);
    }
    if (ltid + 128u < num_zetas) {
        zeta_mont[ltid + 128u] = mont_mul(zetas[ltid + 128u], r2, q_val, q_inv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device uint *poly = coeffs + (size_t)tgid * n;

    // 2. Coalesced device load & Level 0
    uint u = poly[ltid];
    uint v = poly[ltid + 128u];
    
    if (n_levels_val > 0u) {
        uint z_mont = zeta_mont[1];
        uint t = mont_mul(v, z_mont, q_val, q_inv);
        uint new_u = mod_add_safe(u, t, q_val);
        uint new_v = mod_sub_safe(u, t, q_val);
        u = new_u; 
        v = new_v;
    }
    
    // Early exit cleanly limits execution if no further stages are requested
    if (n_levels_val <= 1u) {
        poly[ltid] = u;
        poly[ltid + 128u] = v;
        return;
    }
    
    a[ltid] = u;
    a[ltid + 128u] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Level 1 across threadgroup memory
    uint group_idx_1 = ltid >> 6; // ltid / 64
    uint j_in_group_1 = ltid & 63u; // ltid % 64
    uint j_1 = (group_idx_1 << 7) | j_in_group_1; 
    
    uint z_mont_1 = zeta_mont[2u + group_idx_1];
    uint u1 = a[j_1];
    uint v1 = a[j_1 + 64u];
    uint t1 = mont_mul(v1, z_mont_1, q_val, q_inv);
    u = mod_add_safe(u1, t1, q_val);
    v = mod_sub_safe(u1, t1, q_val);
    
    if (n_levels_val == 2u) {
        poly[j_1] = u;
        poly[j_1 + 64u] = v;
        return;
    }
    
    a[j_1] = u;
    a[j_1 + 64u] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Levels 2-7 completely in registers (independent SIMD sub-NTTs)
    uint sg = ltid >> 5;  // ltid / 32 (simdgroup ID)
    uint k = ltid & 31u;  // lane ID
    uint B = sg << 6;     // base block offset for this simdgroup
    uint B_plus_k = B + k;
    uint B_plus_k_32 = B_plus_k + 32u;

    uint x = a[B_plus_k];
    uint y = a[B_plus_k_32];

    // Level 2 fits right within the two variables per thread
    if (n_levels_val > 2u) {
        uint zeta_sg = zeta_mont[4u + sg];
        uint t = mont_mul(y, zeta_sg, q_val, q_inv);
        uint new_x = mod_add_safe(x, t, q_val);
        uint new_y = mod_sub_safe(x, t, q_val);
        x = new_x; 
        y = new_y;
    }

    // Remaining inner levels (lengths 16, 8, 4, 2, 1) use simd_shuffle to communicate
    #pragma unroll
    for (uint level = 3u; level < 8u; ++level) {
        if (level >= n_levels_val) break;

        uint L = 128u >> level;
        uint shift = 8u - level;
        bool is_top = (k & L) == 0u;
        
        // Branchless SIMD butterfly for x half
        uint zeta_x = zeta_mont[(1u << level) + (B_plus_k >> shift)];
        uint x_partner = simd_shuffle_xor(x, L);
        uint x_bot = is_top ? x_partner : x;
        uint x_top = is_top ? x : x_partner;
        uint tx = mont_mul(x_bot, zeta_x, q_val, q_inv);
        x = is_top ? mod_add_safe(x_top, tx, q_val) : mod_sub_safe(x_top, tx, q_val);
        
        // Branchless SIMD butterfly for y half
        uint zeta_y = zeta_mont[(1u << level) + (B_plus_k_32 >> shift)];
        uint y_partner = simd_shuffle_xor(y, L);
        uint y_bot = is_top ? y_partner : y;
        uint y_top = is_top ? y : y_partner;
        uint ty = mont_mul(y_bot, zeta_y, q_val, q_inv);
        y = is_top ? mod_add_safe(y_top, ty, q_val) : mod_sub_safe(y_top, ty, q_val);
    }

    // Coalesced writeback directly to device memory
    poly[B_plus_k] = x;
    poly[B_plus_k_32] = y;
}
```

Result of previous attempt:
            kyb_B1: correct, 0.02 ms, 0.1 GB/s (0.1% of 200 GB/s)
           kyb_B16: correct, 0.02 ms, 1.7 GB/s (0.9% of 200 GB/s)
          kyb_B256: correct, 0.06 ms, 10.2 GB/s (5.1% of 200 GB/s)
  score (gmean of fraction): 0.0062

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
- iter  4: compile=OK | correct=True | score=0.006230969701774426
- iter  5: compile=OK | correct=True | score=0.006464216825696328
- iter  6: compile=OK | correct=True | score=0.006434133418529552
- iter  7: compile=OK | correct=True | score=0.006238858411178923

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel and
propose a STRUCTURALLY different approach.

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
