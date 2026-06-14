#include <metal_stdlib>
using namespace metal;

constexpr constant uint N_MAX = 256u;

template<uint Q>
inline void ntt_core(threadgroup uint* a, device const uint* zetas, uint n_levels_val, uint half_n, uint ltid) {
    uint length_shift = ctz(half_n);
    for (uint level = 0u; level < n_levels_val; ++level) {
        uint length     = 1u << length_shift;
        uint group_idx  = ltid >> length_shift;
        uint j_in_group = ltid & (length - 1u);
        uint j          = (group_idx << (length_shift + 1u)) | j_in_group;
        uint zeta       = zetas[(1u << level) + group_idx];

        uint x = a[j];
        uint y = a[j + length];
        
        uint t;
        if (Q == 3329u) {
            // zeta * y <= 3328^2, fits easily in 32-bit unsigned
            t = (zeta * y) % Q;
        } else {
            // 8380417 requires 64-bit arithmetic
            t = (uint)(((ulong)zeta * y) % Q);
        }

        uint t_add = x + t;
        a[j]          = (t_add >= Q) ? (t_add - Q) : t_add;
        a[j + length] = (x >= t) ? (x - t) : (x + Q - t);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        length_shift--;
    }
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

    // Dispatch to cleanly optimized constants to heavily bias the compiler arithmetic
    if (q_val == 3329u) {
        ntt_core<3329u>(a, zetas, n_levels_val, half_n, ltid);
    } else if (q_val == 8380417u) {
        ntt_core<8380417u>(a, zetas, n_levels_val, half_n, ltid);
    } else {
        ntt_core_dynamic(a, zetas, n_levels_val, half_n, ltid, q_val);
    }

    // Write outputs back to global memory
    poly[ltid]          = a[ltid];
    poly[ltid + half_n] = a[ltid + half_n];
}