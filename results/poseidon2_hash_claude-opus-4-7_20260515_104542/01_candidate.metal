#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;
constexpr constant uint RF_MAX = 8u;
constexpr constant uint RP_MAX = 32u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = hi & EPSILON;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
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

    uint tt = t;
    uint rf = r_f;
    uint rp = r_p;
    uint half_f = rf >> 1u;

    // Load MDS and int_diag into registers.
    ulong mds[T_MAX * T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX * T_MAX; ++i) {
        mds[i] = (i < tt * tt) ? ext_mds[i] : 0ul;
    }
    ulong diag[T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX; ++i) {
        diag[i] = (i < tt) ? int_diag[i] : 0ul;
    }

    // Load round constants into registers.
    ulong rce[RF_MAX * T_MAX];
    for (uint i = 0u; i < rf * tt; ++i) rce[i] = rc_ext[i];
    ulong rci[RP_MAX];
    for (uint i = 0u; i < rp; ++i) rci[i] = rc_int[i];

    thread ulong state[T_MAX];
    #pragma unroll
    for (uint i = 0u; i < T_MAX; ++i) {
        state[i] = (i < tt) ? in_state[idx * tt + i] : 0ul;
    }

    // Specialize for t=3 (the benchmark case).
    if (tt == 3u) {
        ulong s0 = state[0], s1 = state[1], s2 = state[2];
        ulong m00 = mds[0], m01 = mds[1], m02 = mds[2];
        ulong m10 = mds[3], m11 = mds[4], m12 = mds[5];
        ulong m20 = mds[6], m21 = mds[7], m22 = mds[8];
        ulong d0 = diag[0], d1 = diag[1], d2 = diag[2];

        // Pre-multiply by external MDS.
        {
            ulong n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // First half full rounds.
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rce[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rce[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rce[r * 3u + 2u]));
            ulong n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Partial rounds.
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rci[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = gold_add(sum, gold_mul(d0, s0));
            ulong n1 = gold_add(sum, gold_mul(d1, s1));
            ulong n2 = gold_add(sum, gold_mul(d2, s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Second half full rounds.
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rce[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rce[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rce[r * 3u + 2u]));
            ulong n0 = gold_add(gold_add(gold_mul(m00, s0), gold_mul(m01, s1)), gold_mul(m02, s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10, s0), gold_mul(m11, s1)), gold_mul(m12, s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20, s0), gold_mul(m21, s1)), gold_mul(m22, s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx * 3u + 0u] = s0;
        out_state[idx * 3u + 1u] = s1;
        out_state[idx * 3u + 2u] = s2;
        return;
    }

    // Generic path (t != 3).
    // Pre-multiply by external MDS.
    {
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc = gold_add(acc, gold_mul(mds[i * tt + j], state[j]));
            }
            tmp[i] = acc;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) {
            state[i] = sbox(gold_add(state[i], rce[r * tt + i]));
        }
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc = gold_add(acc, gold_mul(mds[i * tt + j], state[j]));
            }
            tmp[i] = acc;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint r = 0u; r < rp; ++r) {
        state[0] = sbox(gold_add(state[0], rci[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < tt; ++i) s = gold_add(s, state[i]);
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            tmp[i] = gold_add(s, gold_mul(diag[i], state[i]));
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint r = half_f; r < rf; ++r) {
        for (uint i = 0u; i < tt; ++i) {
            state[i] = sbox(gold_add(state[i], rce[r * tt + i]));
        }
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong acc = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc = gold_add(acc, gold_mul(mds[i * tt + j], state[j]));
            }
            tmp[i] = acc;
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }

    for (uint i = 0u; i < tt; ++i) {
        out_state[idx * tt + i] = state[i];
    }
}