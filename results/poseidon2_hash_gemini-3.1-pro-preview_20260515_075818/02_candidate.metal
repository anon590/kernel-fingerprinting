#include <metal_stdlib>
using namespace metal;

inline ulong gold_add_lazy(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += 0xFFFFFFFFul;
    return t;
}

inline ulong gold_canonicalize(ulong t) {
    return (t >= 0xFFFFFFFF00000001ul) ? (t - 0xFFFFFFFF00000001ul) : t;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong lo  = (ulong)(uint)p00 | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = (uint)hi;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    if (t0 > lo) t0 -= 0xFFFFFFFFul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += 0xFFFFFFFFul;

    return (t2 >= 0xFFFFFFFF00000001ul) ? (t2 - 0xFFFFFFFF00000001ul) : t2;
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint Arity>
inline void process_sponge(
    device const ulong *in_state,
    device       ulong *out_state,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint r_f, uint r_p, uint idx)
{
    ulong state[Arity];
    #pragma unroll
    for (uint i = 0; i < Arity; ++i) {
        state[i] = in_state[idx * Arity + i];
    }

    ulong mds[Arity * Arity];
    #pragma unroll
    for (uint i = 0; i < Arity * Arity; ++i) {
        mds[i] = ext_mds[i];
    }
    
    ulong diag[Arity];
    #pragma unroll
    for (uint i = 0; i < Arity; ++i) {
        diag[i] = int_diag[i];
    }

    ulong tmp[Arity];
    
    // Initial external MDS
    #pragma unroll
    for (uint i = 0; i < Arity; ++i) {
        ulong acc = 0ul;
        #pragma unroll
        for (uint j = 0; j < Arity; ++j) {
            acc = gold_add_lazy(acc, gold_mul(mds[i * Arity + j], state[j]));
        }
        tmp[i] = gold_canonicalize(acc);
    }
    #pragma unroll
    for (uint i = 0; i < Arity; ++i) state[i] = tmp[i];

    uint half_f = r_f >> 1;

    // First half full rounds
    for (uint r = 0; r < half_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) {
            state[i] = sbox(gold_add_lazy(state[i], rc_ext[r * Arity + i]));
        }
        
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0; j < Arity; ++j) {
                acc = gold_add_lazy(acc, gold_mul(mds[i * Arity + j], state[j]));
            }
            tmp[i] = gold_canonicalize(acc);
        }
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) state[i] = tmp[i];
    }

    // Partial rounds
    for (uint r = 0; r < r_p; ++r) {
        state[0] = sbox(gold_add_lazy(state[0], rc_int[r]));
        
        ulong s = 0ul;
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) s = gold_add_lazy(s, state[i]);
        s = gold_canonicalize(s);
        
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) {
            // Update in-place safe here because `s` was gathered prior to overwrite
            state[i] = gold_canonicalize(gold_add_lazy(s, gold_mul(diag[i], state[i])));
        }
    }

    // Second half full rounds
    for (uint r = half_f; r < r_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) {
            state[i] = sbox(gold_add_lazy(state[i], rc_ext[r * Arity + i]));
        }
        
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0; j < Arity; ++j) {
                acc = gold_add_lazy(acc, gold_mul(mds[i * Arity + j], state[j]));
            }
            tmp[i] = gold_canonicalize(acc);
        }
        #pragma unroll
        for (uint i = 0; i < Arity; ++i) state[i] = tmp[i];
    }

    #pragma unroll
    for (uint i = 0; i < Arity; ++i) {
        out_state[idx * Arity + i] = state[i];
    }
}

kernel void poseidon2_hash(
    device const ulong *in_state        [[buffer(0)]],
    device       ulong *out_state       [[buffer(1)]],
    device const ulong *rc_ext          [[buffer(2)]],
    device const ulong *rc_int          [[buffer(3)]],
    device const ulong *ext_mds         [[buffer(4)]],
    device const ulong *int_diag        [[buffer(5)]],
    constant uint      &t               [[buffer(6)]],
    constant uint      &r_f             [[buffer(7)]],
    constant uint      &r_p             [[buffer(8)]],
    constant uint      &batch           [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    if (t == 4) {
        process_sponge<4>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx);
    } else if (t == 3) {
        process_sponge<3>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx);
    } else if (t == 2) {
        process_sponge<2>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx);
    } else if (t == 1) {
        process_sponge<1>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx);
    }
}