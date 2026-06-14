#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong gold_reduce128(ulong lo, ulong hi) {
    uint x_hi_lo = (uint)hi;
    uint x_hi_hi = (uint)(hi >> 32);

    ulong t0 = lo - x_hi_hi;
    t0 -= (t0 > lo) ? 0xFFFFFFFFul : 0ul;

    ulong t1 = ((ulong)x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong a0b0 = (ulong)a0 * b0;
    ulong a0b1 = (ulong)a0 * b1;
    ulong a1b0 = (ulong)a1 * b0;
    ulong a1b1 = (ulong)a1 * b1;

    ulong mid = a1b0 + (a0b0 >> 32);
    ulong hi = a1b1 + (mid >> 32);
    mid = (mid & 0xFFFFFFFFul) + a0b1;
    hi += (mid >> 32);
    ulong lo = (mid << 32) | (uint)a0b0;

    return gold_reduce128(lo, hi);
}

inline ulong gold_mac(ulong a, ulong b, ulong c) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong a0b0 = (ulong)a0 * b0;
    ulong a0b1 = (ulong)a0 * b1;
    ulong a1b0 = (ulong)a1 * b0;
    ulong a1b1 = (ulong)a1 * b1;

    ulong mid = a1b0 + (a0b0 >> 32);
    ulong hi = a1b1 + (mid >> 32);
    mid = (mid & 0xFFFFFFFFul) + a0b1;
    hi += (mid >> 32);
    ulong lo = (mid << 32) | (uint)a0b0;

    // Fused accumulation of c into the 128-bit product prior to reduction.
    lo += c;
    hi += (lo < c) ? 1ul : 0ul;

    return gold_reduce128(lo, hi);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint T>
inline void process_sponge(
    device const ulong *in_state,
    device       ulong *out_state,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint r_f,
    uint r_p,
    uint idx)
{
    ulong state[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        state[i] = in_state[(size_t)idx * T + i];
    }

    ulong mds[T][T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            mds[i][j] = ext_mds[i * T + j];
        }
    }

    ulong diag[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        diag[i] = int_diag[i];
    }

    ulong tmp[T];

    // Pre-multiply by external MDS.
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong acc = 0ul;
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            acc = gold_mac(mds[i][j], state[j], acc);
        }
        tmp[i] = acc;
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];

    const uint half_f = r_f >> 1;

    // First half full rounds.
    for (uint r = 0; r < half_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext[r * T + i]));
        }
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0; j < T; ++j) {
                acc = gold_mac(mds[i][j], state[j], acc);
            }
            tmp[i] = acc;
        }
        #pragma unroll
        for (uint i = 0; i < T; ++i) state[i] = tmp[i];
    }

    // Partial rounds.
    for (uint r = 0; r < r_p; ++r) {
        state[0] = sbox(gold_add(state[0], rc_int[r]));
        
        ulong s = state[0];
        #pragma unroll
        for (uint i = 1; i < T; ++i) {
            s = gold_add(s, state[i]);
        }

        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            tmp[i] = gold_mac(diag[i], state[i], s);
        }
        #pragma unroll
        for (uint i = 0; i < T; ++i) state[i] = tmp[i];
    }

    // Second half full rounds.
    for (uint r = half_f; r < r_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext[r * T + i]));
        }
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            ulong acc = 0ul;
            #pragma unroll
            for (uint j = 0; j < T; ++j) {
                acc = gold_mac(mds[i][j], state[j], acc);
            }
            tmp[i] = acc;
        }
        #pragma unroll
        for (uint i = 0; i < T; ++i) state[i] = tmp[i];
    }

    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        out_state[(size_t)idx * T + i] = state[i];
    }
}

kernel void poseidon2_hash(
    device const ulong *in_state    [[buffer(0)]],
    device       ulong *out_state   [[buffer(1)]],
    device const ulong *rc_ext      [[buffer(2)]],
    device const ulong *rc_int      [[buffer(3)]],
    device const ulong *ext_mds     [[buffer(4)]],
    device const ulong *int_diag    [[buffer(5)]],
    constant uint      &t           [[buffer(6)]],
    constant uint      &r_f         [[buffer(7)]],
    constant uint      &r_p         [[buffer(8)]],
    constant uint      &batch       [[buffer(9)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    switch (t) {
        case 1: process_sponge<1>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 2: process_sponge<2>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 3: process_sponge<3>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 4: process_sponge<4>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
    }
}