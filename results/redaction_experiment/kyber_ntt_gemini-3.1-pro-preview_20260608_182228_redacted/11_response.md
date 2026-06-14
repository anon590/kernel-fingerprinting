I am significantly restructuring the kernel to use a register-based NTT via SIMD shuffle instructions, drastically reducing reliance on threadgroup memory and synchronization.

### Optimization Summary
**What I applied:** 
I completely unrolled the Cooley-Tukey NTT into two phases. The first phase handles the two largest polynomial strides (`length = 128, 64`) using threadgroup memory and fuses the initial load from device memory into the `level = 0` butterfly. The second phase loads the partially transformed elements into SIMD registers and executes the final 5 levels (`length = 32` down to `1`) entirely using register math and `simd_shuffle_xor`, bypassing threadgroup memory completely. The final register values are written directly back to device memory.

**Why it improves performance:** 
The previous kernel was deeply bottlenecked by executing 7 threadgroup barriers per polynomial and generating 28 shared-memory accesses per thread. This structural redesign reduces `threadgroup_barrier` calls from 7 down to just 2, and eliminates over 70% of the threadgroup memory reads/writes. By resolving the bulk of the butterflies in isolated SIMD warp registers, we fully utilize the arithmetic ALUs and bypass shared memory latency, which will massively boost memory bandwidth throughput.

```metal
#include <metal_stdlib>
using namespace metal;

// Overflow-safe Montgomery reduction for any q <= 2^32 - 1
inline uint mont_mul(uint a, uint b, uint q, uint q_inv) {
    uint P_lo = a * b;
    uint P_hi = mulhi(a, b);
    
    uint Q = P_lo * q_inv;
    uint M_hi = mulhi(Q, q);
    
    // Manually handle carry to strictly prevent 64-bit bounds overflow for massive q
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
    constant uint     &q          [[buffer(2)]],
    constant uint     &n          [[buffer(3)]],
    constant uint     &n_levels   [[buffer(4)]],
    constant uint     &batch      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]])
{
    if (tgid >= batch) return;
    if (n_levels == 0u) return;

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

    // Collaboratively load and convert zetas to Montgomery form
    uint num_zetas = 1u << n_levels;
    for (uint i = ltid; i < num_zetas; i += 128u) {
        zeta_mont[i] = mont_mul(zetas[i], r2, q_val, q_inv);
    }
    
    device uint *poly = coeffs + (size_t)tgid * n;
    
    // Ensure all twiddles are visible before Cooley-Tukey loops begin
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_len = 7u; // Derived from n=256, n/2=128
    uint k_start = 1u;
    uint level = 0u;

    // Phase 1: level 0 (length 128) -> fused with the coalesced global memory load
    {
        uint length = 128u;
        uint j = ltid;
        uint z_mont = zeta_mont[k_start]; // group_idx = 0
        
        uint x = poly[j];
        uint y = poly[j + length];
        
        uint t = mont_mul(y, z_mont, q_val, q_inv);

        a[j]          = mod_add_safe(x, t, q_val);
        a[j + length] = mod_sub_safe(x, t, q_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        log2_len--;
        k_start <<= 1u;
        level++;
    }

    // Phase 1: level 1 (length 64) -> requires cross-simdgroup synchronization
    if (level < n_levels && log2_len == 6u) {
        uint length = 64u;
        uint group_idx  = ltid >> 6u;
        uint j_in_group = ltid & 63u;
        uint j          = (group_idx << 7u) | j_in_group;
        
        uint z_mont = zeta_mont[k_start + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint t = mont_mul(y, z_mont, q_val, q_inv);

        a[j]          = mod_add_safe(x, t, q_val);
        a[j + length] = mod_sub_safe(x, t, q_val);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        log2_len--;
        k_start <<= 1u;
        level++;
    }

    // Prepare indices mapped precisely to the local SIMD group registers
    uint g = ltid / 32u;
    uint s = ltid % 32u;
    uint idx_X = g * 64u + s;
    uint idx_Y = g * 64u + s + 32u;

    // Phase 2: Unrolled, fully contained within SIMD warps (no SMEM/barriers required)
    if (level < n_levels) {
        uint X = a[idx_X];
        uint Y = a[idx_Y];

        // level 2 (length 32) -> butterflies natively isolated inside current thread 
        if (level < n_levels && log2_len == 5u) {
            uint z_mont = zeta_mont[k_start + g];
            uint t = mont_mul(Y, z_mont, q_val, q_inv);
            
            uint X_new = mod_add_safe(X, t, q_val);
            uint Y_new = mod_sub_safe(X, t, q_val);
            X = X_new;
            Y = Y_new;
            
            log2_len--;
            k_start <<= 1u;
            level++;
        }

        // levels 3 through 7 (lengths 16, 8, 4, 2, 1) -> solved with SIMD shuffles
        while (level < n_levels) {
            uint L = 1u << log2_len;
            bool is_lower = (s & L) == 0;
            
            uint other_X = simd_shuffle_xor(X, L);
            uint other_Y = simd_shuffle_xor(Y, L);
            
            uint group_idx_X = idx_X >> (log2_len + 1u);
            uint group_idx_Y = idx_Y >> (log2_len + 1u);
            
            uint zeta_X = zeta_mont[k_start + group_idx_X];
            uint zeta_Y = zeta_mont[k_start + group_idx_Y];
            
            // Resolve X register bounds 
            uint b_X = is_lower ? other_X : X;
            uint a_X = is_lower ? X : other_X;
            uint T_X = mont_mul(b_X, zeta_X, q_val, q_inv);
            uint X_new = is_lower ? mod_add_safe(a_X, T_X, q_val) : mod_sub_safe(a_X, T_X, q_val);

            // Resolve Y register bounds 
            uint b_Y = is_lower ? other_Y : Y;
            uint a_Y = is_lower ? Y : other_Y;
            uint T_Y = mont_mul(b_Y, zeta_Y, q_val, q_inv);
            uint Y_new = is_lower ? mod_add_safe(a_Y, T_Y, q_val) : mod_sub_safe(a_Y, T_Y, q_val);

            X = X_new;
            Y = Y_new;

            log2_len--;
            k_start <<= 1u;
            level++;
        }

        // Write perfectly contiguous resulting vectors natively back to unified global device memory
        poly[idx_X] = X;
        poly[idx_Y] = Y;
    } else {
        // If n_levels completed entirely via Phase 1
        poly[idx_X] = a[idx_X];
        poly[idx_Y] = a[idx_Y];
    }
}
```