Optimization: We drastically reduce `threadgroup_barrier` stalls and memory traffic by transitioning to an in-register algorithm for the last 5 levels of the NTT (lengths $\le 16$). Because all data dependencies for lengths $\le 32$ fall completely within a single 32-lane SIMD group, we can load coefficients into registers after Level 2 and exchange elements across lanes using `simd_shuffle_xor` without accessing memory. After finishing the levels in registers, we analytically compute the output indices and write back to threadgroup memory to guarantee 100% perfectly coalesced contiguous global memory writes. We fully specialize loops and modulus constants for known variants (Kyber and Dilithium) so the compiler emits optimally scheduled magic-number multiplications instead of slow modulo division. 

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

template<uint Q, uint N_LEVELS>
inline void ntt_core_fast(threadgroup uint* a, device const uint* zetas, uint ltid) {
    auto mul_mod = [](uint x, uint y) -> uint {
        if (Q <= 65535u) {
            return (x * y) % Q;
        } else {
            return (uint)(((ulong)x * y) % Q);
        }
    };
    auto add_mod = [](uint x, uint y) -> uint {
        uint t = x + y;
        return (t >= Q) ? (t - Q) : t;
    };
    auto sub_mod = [](uint x, uint y) -> uint {
        return (x >= y) ? (x - y) : (x + Q - y);
    };

    // Level 0: length = 128
    uint u = a[ltid];
    uint v = a[ltid + 128u];
    uint zeta = zetas[1];
    uint t = mul_mod(zeta, v);
    a[ltid] = add_mod(u, t);
    a[ltid + 128u] = sub_mod(u, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 1: length = 64
    uint group_idx = ltid >> 6;
    uint j = (group_idx << 7) | (ltid & 63u);
    u = a[j];
    v = a[j + 64u];
    zeta = zetas[2u + group_idx];
    t = mul_mod(zeta, v);
    a[j] = add_mod(u, t);
    a[j + 64u] = sub_mod(u, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Level 2: length = 32
    group_idx = ltid >> 5;
    j = (group_idx << 6) | (ltid & 31u);
    u = a[j];
    v = a[j + 32u];
    zeta = zetas[4u + group_idx];
    t = mul_mod(zeta, v);
    u = add_mod(u, t);
    v = sub_mod(u, t);

    // Levels 3 to N_LEVELS - 1 (Register Phase via Shuffle)
    #pragma unroll
    for (uint level = 3; level < N_LEVELS; ++level) {
        uint D = 128u >> level;
        bool is_even = (ltid & D) == 0;
        
        // Exchange cross-butterfly values dynamically 
        uint my_exchange = is_even ? v : u;
        uint exchanged = simd_shuffle_xor(my_exchange, (ushort)D);
        v = is_even ? exchanged : v;
        u = is_even ? u : exchanged;

        uint zeta_idx = (1u << level) + (ltid / D);
        uint current_zeta = zetas[zeta_idx];
        uint t_val = mul_mod(current_zeta, v);
        
        uint u_next = add_mod(u, t_val);
        uint v_next = sub_mod(u, t_val);
        u = u_next;
        v = v_next;
    }

    // Analytically trace positional index for writeback
    uint pos_u = 0, pos_v = 0;
    if (N_LEVELS == 8u) {
        pos_u = ltid << 1;
        pos_v = (ltid << 1) | 1u;
    } else if (N_LEVELS == 7u) {
        pos_u = ((ltid & ~1u) << 1) + (ltid & 1u);
        pos_v = pos_u + 2u;
    }
    
    a[pos_u] = u;
    a[pos_v] = v;
}

inline void ntt_core_dynamic(threadgroup uint* a, device const uint* zetas, uint n_levels_val, uint half_n, uint ltid, uint q_val) {
    uint length_shift = ctz(half_n);
    for (uint level = 0u; level < n_levels_val; ++level) {
        uint length     = 1u << length_shift;
        uint group_idx  = ltid >> length_shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (length_shift + 1u)) | j_in_group;
        uint zeta       = zetas[(1u << level) + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint t = (uint)(((ulong)zeta * y) % q_val);

        uint t_add = x + t;
        a[j]          = (t_add >= q_val) ? (t_add - q_val) : t_add;
        a[j + length] = (x >= t) ? (x - t) : (x + q_val - t);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        length_shift--;
    }
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

    threadgroup uint a[N_MAX];

    uint n_val = n;
    uint half_n = n_val >> 1u;
    device uint *poly = coeffs + (size_t)tgid * n_val;

    // Load coefficients into fast threadgroup memory
    a[ltid]          = poly[ltid];
    a[ltid + half_n] = poly[ltid + half_n];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint q_val = q;
    uint n_levels_val = n_levels;
    bool handled = false;

    // Unroll and optimize division constants specific to Kyber and Dilithium
    if (n_val == 256u) {
        if (q_val == 3329u && n_levels_val == 7u) {
            ntt_core_fast<3329u, 7u>(a, zetas, ltid);
            handled = true;
        } else if (q_val == 8380417u && n_levels_val == 8u) {
            ntt_core_fast<8380417u, 8u>(a, zetas, ltid);
            handled = true;
        } else if (q_val == 3329u && n_levels_val == 8u) {
            ntt_core_fast<3329u, 8u>(a, zetas, ltid);
            handled = true;
        } else if (q_val == 8380417u && n_levels_val == 7u) {
            ntt_core_fast<8380417u, 7u>(a, zetas, ltid);
            handled = true;
        }
    }

    if (!handled) {
        ntt_core_dynamic(a, zetas, n_levels_val, half_n, ltid, q_val);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Guaranteed 100% coalesced parallel memory writes back to the global buffer
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}
```