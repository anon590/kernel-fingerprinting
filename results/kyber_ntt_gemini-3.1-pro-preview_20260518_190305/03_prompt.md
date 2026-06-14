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

// Computes -q^{-1} mod 2^32 using Newton-Raphson
inline uint get_q_inv(uint q) {
    uint x = q;
    x *= 2u - q * x;
    x *= 2u - q * x;
    x *= 2u - q * x;
    x *= 2u - q * x;
    return -x;
}

// Ultra-fast 32-bit Montgomery reduction (avoids 64-bit add/div)
// Computes: (P * 2^{-32}) mod q
inline uint mont_reduce(uint P_lo, uint P_hi, uint q, uint q_inv) {
    uint m = P_lo * q_inv;
    // P_lo + m*q is a multiple of 2^32. The carry to the upper 32 bits is 1 if P_lo != 0
    uint t = P_hi + mulhi(m, q) + (P_lo != 0u ? 1u : 0u);
    return (t >= q) ? t - q : t;
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

    // Localize scalars to registers
    uint local_q = q;
    uint local_n = n;
    uint local_n_levels = n_levels;

    uint q_inv = get_q_inv(local_q);
    uint half_n = local_n >> 1u;
    
    threadgroup uint a[256];
    threadgroup uint shared_zetas[256];

    // Load polynomial
    device uint *poly = coeffs + (size_t)tgid * local_n;
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];

    // Cooperatively load zetas and convert them to Montgomery form
    uint zetas_count = 1u << local_n_levels;
    ulong R_mod_q = 0x100000000ull % local_q;
    for (uint i = ltid; i < zetas_count; i += half_n) {
        shared_zetas[i] = (uint)(((ulong)zetas[i] * R_mod_q) % local_q);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log_len = ctz(half_n);
    uint k_start = 1u;
    uint current_level = 0u;

    // --- Phase 1: Shared Memory NTT (length >= 32) ---
    for (; current_level < local_n_levels; ++current_level) {
        if (log_len < 5u) break; // Delegate smaller lengths to Register NTT
        
        uint length     = 1u << log_len;
        uint group_idx  = ltid >> log_len;
        uint j          = (group_idx << (log_len + 1u)) | (ltid & (length - 1u));
        uint zeta_mont  = shared_zetas[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint P_lo = y * zeta_mont;
        uint P_hi = mulhi(y, zeta_mont);
        uint t = mont_reduce(P_lo, P_hi, local_q, q_inv);

        uint sum = x + t;
        a[j]          = (sum >= local_q) ? sum - local_q : sum;
        a[j + length] = (x >= t) ? x - t : x + local_q - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        log_len--;
    }

    // --- Phase 2: Register Shuffle NTT (length <= 16) ---
    // Avoids threadgroup barriers completely by executing over simd-groups
    if (current_level < local_n_levels) {
        uint s = ltid / 32u;
        uint k = ltid % 32u;
        uint base = s * 64u;
        
        uint x = a[base + k];
        uint y = a[base + k + 32u];
        
        for (; current_level < local_n_levels; ++current_level) {
            uint L = 1u << log_len;
            
            // Re-derive group index mappings for both items traversing the shuffle sizes
            uint group_idx_x = (base + k) >> (log_len + 1u);
            uint group_idx_y = (base + 32u + k) >> (log_len + 1u);
            
            uint zeta_mont_x = shared_zetas[k_start + group_idx_x];
            uint zeta_mont_y = shared_zetas[k_start + group_idx_y];
            
            uint other_x = simd_shuffle_xor(x, L);
            uint other_y = simd_shuffle_xor(y, L);
            
            bool is_bottom = (k & L) != 0u;
            
            // Resolve top & bottom items cooperatively
            uint top_x = is_bottom ? other_x : x;
            uint bot_x = is_bottom ? x : other_x;
            
            uint P_lo_x = bot_x * zeta_mont_x;
            uint P_hi_x = mulhi(bot_x, zeta_mont_x);
            uint t_x = mont_reduce(P_lo_x, P_hi_x, local_q, q_inv);
            
            uint sum_x = top_x + t_x;
            uint new_top_x = (sum_x >= local_q) ? sum_x - local_q : sum_x;
            uint new_bot_x = (top_x >= t_x) ? top_x - t_x : top_x + local_q - t_x;
            
            x = is_bottom ? new_bot_x : new_top_x;
            
            uint top_y = is_bottom ? other_y : y;
            uint bot_y = is_bottom ? y : other_y;
            
            uint P_lo_y = bot_y * zeta_mont_y;
            uint P_hi_y = mulhi(bot_y, zeta_mont_y);
            uint t_y = mont_reduce(P_lo_y, P_hi_y, local_q, q_inv);
            
            uint sum_y = top_y + t_y;
            uint new_top_y = (sum_y >= local_q) ? sum_y - local_q : sum_y;
            uint new_bot_y = (top_y >= t_y) ? top_y - t_y : top_y + local_q - t_y;
            
            y = is_bottom ? new_bot_y : new_top_y;
            
            k_start <<= 1u;
            log_len--;
        }
        
        a[base + k]       = x;
        a[base + k + 32u] = y;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write-back safely 
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```

Result of previous attempt:
            kyb_B1: correct, 0.03 ms, 0.0 Gmodmul/s (u32) (0.1% of 53 Gops/s (int64 mul, est))
           kyb_B16: correct, 0.03 ms, 0.5 Gmodmul/s (u32) (0.9% of 53 Gops/s (int64 mul, est))
          kyb_B256: correct, 0.11 ms, 2.0 Gmodmul/s (u32) (3.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0058

## History

- iter  0: compile=OK | correct=True | score=0.0043181043832901085
- iter  1: compile=OK | correct=True | score=0.005482531848977972
- iter  2: compile=OK | correct=True | score=0.0058484757307926466

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
