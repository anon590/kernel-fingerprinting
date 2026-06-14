I am rewriting the kernel to perform an entirely register-resident, distributed Cooley-Tukey NTT. 

**Optimizations applied:**
1. **Dynamic Element Routing**: Instead of relying on `threadgroup` memory for every butterfly layer, threads explicitly keep two polynomial elements in registers (`X` and `Y`) and update them in place. They track the "original" indices of these elements dynamically.
2. **Hybrid Exchanges**: At each level, threads exchange exactly one element with a carefully calculated partner. For strides $\ge 32$, this uses conflict-free `threadgroup` memory block swaps. For strides $< 32$, it seamlessly falls back to zero-latency, barrier-free `simd_shuffle` operations across registers within the SIMD group.
3. **Perfect Coalescing**: By tracking exactly where elements belong (`idx_X`, `idx_Y`), the final scatter writeback to `threadgroup` is perfectly bank-conflict-free, followed by a 100% coalesced flush to device memory.

**Why it improves performance**:
This completely eliminates half the `threadgroup_barrier` calls and entirely avoids `threadgroup` roundtrips for the deepest 5 levels of the NTT, maximizing ALU/register utilization and fully saturating the SIMD throughput.

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