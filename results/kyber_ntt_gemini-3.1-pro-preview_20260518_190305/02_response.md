An elegant and extremely fast way to optimize NTT on Apple GPUs is to avoid 64-bit emulated division entirely inside the butterfly loops, and to avoid shared memory and barriers for the smaller butterfly stages. 

1. **Montgomery Reduction with `mulhi`**: Instead of a slow 64-bit modulo, we convert the twiddles (`zetas`) into Montgomery form once during the load phase. Inside the loop, we use a custom pure 32-bit Montgomery reduction that only requires 2 `mulhi` instructions and basic additions. The 64-bit carry handles natively via conditional increment, drastically reducing instruction count.
2. **Register-level NTT**: For the last 5 levels (`length <= 16`), we break out of the `threadgroup` loops and hold the polynomial coefficients strictly in thread registers. The butterflies are completed natively over `simd_shuffle_xor` operations, completely bypassing L1 memory latency and avoiding threadgroup barriers for over half of the NTT execution.

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