1. Replaced the expensive 64-bit modulo in the zeta conversion loop with a 32-bit Montgomery reduction by precomputing `R^2 mod q` once per threadgroup.
2. Fused Level 0 (length=128) computation directly into the initial global memory load to registers. This avoids writing un-transformed inputs to threadgroup memory, eliminating one shared memory roundtrip and one barrier.
3. Fused the final global memory write with the end of Phase 2 (SIMD shuffle NTT). Threads write their register-resident coefficients directly to global memory in a fully coalesced warp-stride pattern, saving the final shared memory write and its preceding barrier.
4. Simplified `mont_reduce` by using an integer cast for the carry flag rather than a ternary condition.

These changes significantly reduce synchronization overhead, threadgroup memory traffic, and ALU latency.

```metal
#include <metal_stdlib>
using namespace metal;

__attribute__((always_inline))
inline uint get_q_inv(uint q) {
    uint x = q;
    x *= 2u - q * x;
    x *= 2u - q * x;
    x *= 2u - q * x;
    x *= 2u - q * x;
    return -x;
}

__attribute__((always_inline))
inline uint mont_reduce(uint P_lo, uint P_hi, uint q, uint q_inv) {
    uint m = P_lo * q_inv;
    uint t = P_hi + mulhi(m, q) + (uint)(P_lo != 0u);
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

    uint local_q = q;
    uint local_n = n;
    uint local_n_levels = n_levels;

    uint q_inv = get_q_inv(local_q);
    uint half_n = local_n >> 1u;
    
    threadgroup uint a[256];
    threadgroup uint shared_zetas[256];

    device uint *poly = coeffs + (size_t)tgid * local_n;

    // Cooperatively load zetas and convert them to Montgomery form using 32-bit math
    uint zetas_count = 1u << local_n_levels;
    ulong R_mod_q = 0x100000000ull % local_q;
    uint R2 = (uint)((R_mod_q * R_mod_q) % local_q);
    for (uint i = ltid; i < zetas_count; i += half_n) {
        uint z = zetas[i];
        uint P_lo = z * R2;
        uint P_hi = mulhi(z, R2);
        shared_zetas[i] = mont_reduce(P_lo, P_hi, local_q, q_inv);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log_len = ctz(half_n);
    uint k_start = 1u;
    uint current_level = 0u;

    uint x = poly[ltid];
    uint y = poly[ltid + half_n];

    // --- Level 0: Fused Register NTT ---
    if (current_level < local_n_levels) {
        uint zeta_mont = shared_zetas[1];
        uint P_lo = y * zeta_mont;
        uint P_hi = mulhi(y, zeta_mont);
        uint t = mont_reduce(P_lo, P_hi, local_q, q_inv);
        
        uint sum = x + t;
        uint new_x = (sum >= local_q) ? sum - local_q : sum;
        y = (x >= t) ? x - t : x + local_q - t;
        x = new_x;

        k_start <<= 1u;
        log_len--;
        current_level++;
    }

    a[ltid] = x;
    a[ltid + half_n] = y;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 1: Shared Memory NTT (length >= 32) ---
    for (; current_level < local_n_levels; ++current_level) {
        if (log_len < 5u) break; // Delegate smaller lengths to Register NTT
        
        uint length     = 1u << log_len;
        uint group_idx  = ltid >> log_len;
        uint j          = (group_idx << (log_len + 1u)) | (ltid & (length - 1u));
        uint zeta_mont  = shared_zetas[k_start + group_idx];

        uint val_x = a[j];
        uint val_y = a[j + length];
        
        uint P_lo = val_y * zeta_mont;
        uint P_hi = mulhi(val_y, zeta_mont);
        uint t = mont_reduce(P_lo, P_hi, local_q, q_inv);

        uint sum = val_x + t;
        a[j]          = (sum >= local_q) ? sum - local_q : sum;
        a[j + length] = (val_x >= t) ? val_x - t : val_x + local_q - t;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        k_start <<= 1u;
        log_len--;
    }

    // --- Phase 2: Register Shuffle NTT (length <= 16) ---
    if (current_level < local_n_levels) {
        uint s = ltid / 32u;
        uint k = ltid % 32u;
        uint base = s * 64u;
        
        uint reg_x = a[base + k];
        uint reg_y = a[base + k + 32u];
        
        for (; current_level < local_n_levels; ++current_level) {
            uint L = 1u << log_len;
            
            uint group_idx_x = (base + k) >> (log_len + 1u);
            uint group_idx_y = (base + 32u + k) >> (log_len + 1u);
            
            uint zeta_mont_x = shared_zetas[k_start + group_idx_x];
            uint zeta_mont_y = shared_zetas[k_start + group_idx_y];
            
            uint other_x = simd_shuffle_xor(reg_x, L);
            uint other_y = simd_shuffle_xor(reg_y, L);
            
            bool is_bottom = (k & L) != 0u;
            
            uint top_x = is_bottom ? other_x : reg_x;
            uint bot_x = is_bottom ? reg_x : other_x;
            
            uint P_lo_x = bot_x * zeta_mont_x;
            uint P_hi_x = mulhi(bot_x, zeta_mont_x);
            uint t_x = mont_reduce(P_lo_x, P_hi_x, local_q, q_inv);
            
            uint sum_x = top_x + t_x;
            reg_x = is_bottom ? ((top_x >= t_x) ? top_x - t_x : top_x + local_q - t_x)
                              : ((sum_x >= local_q) ? sum_x - local_q : sum_x);
            
            uint top_y = is_bottom ? other_y : reg_y;
            uint bot_y = is_bottom ? reg_y : other_y;
            
            uint P_lo_y = bot_y * zeta_mont_y;
            uint P_hi_y = mulhi(bot_y, zeta_mont_y);
            uint t_y = mont_reduce(P_lo_y, P_hi_y, local_q, q_inv);
            
            uint sum_y = top_y + t_y;
            reg_y = is_bottom ? ((top_y >= t_y) ? top_y - t_y : top_y + local_q - t_y)
                              : ((sum_y >= local_q) ? sum_y - local_q : sum_y);
            
            k_start <<= 1u;
            log_len--;
        }
        
        // Write out directly to global memory, avoiding the final barrier and threadgroup store entirely
        poly[base + k]       = reg_x;
        poly[base + k + 32u] = reg_y;
    } else {
        // Only run if Phase 2 skipped completely
        poly[ltid]          = a[ltid];
        poly[ltid + half_n] = a[ltid + half_n];
    }
}
```