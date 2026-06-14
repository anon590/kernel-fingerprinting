To optimize this kernel, we apply two primary enhancements over the incumbent:

1. **Templated Loop Unrolling via Dynamic Dispatch:** Instead of using variable-length loops constrained by `t` (which is passed via `constant uint &t`), we dispatch to a templated `process_sponge<T>` function. This forces the compiler to completely unroll the loops over the state vector. Crucially, unlike the previous attempt, we do **not** copy `ext_mds` and `int_diag` to thread-local register arrays—which severely increased register pressure and caused spills. Instead, we read them directly from the `device const` pointers, taking advantage of the GPU's uniform constant cache broadcast.
2. **Branchless Goldilocks Field Arithmetic:** We refactor the Goldilocks 128-bit multiplication and reduction. By computing the 128-bit product using four native 32x32->64 multipliers and reducing it with shift-and-subtracts leveraging $2^{64} \equiv 2^{32} - 1 \pmod p$, we bypass the heavier general-purpose 128-bit reduction function. We enforce branchless control flow natively using Metal's `select`.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += select(0ul, 0xFFFFFFFFul, t < a);
    return select(t, t - P_GOLD, t >= P_GOLD);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a_lo = (uint)a;
    uint a_hi = (uint)(a >> 32);
    uint b_lo = (uint)b;
    uint b_hi = (uint)(b >> 32);

    ulong p0 = (ulong)a_lo * b_lo;
    ulong p1 = (ulong)a_lo * b_hi;
    ulong p2 = (ulong)a_hi * b_lo;
    ulong p3 = (ulong)a_hi * b_hi;

    ulong mid = (p0 >> 32) + (uint)p1 + (uint)p2;
    ulong lo = (uint)p0 | (mid << 32);
    ulong hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);

    ulong hi_hi = hi >> 32;
    ulong hi_lo = (uint)hi;

    ulong t0 = lo - hi_hi;
    t0 -= select(0ul, 0xFFFFFFFFul, t0 > lo);

    ulong t1 = (hi_lo << 32) - hi_lo;
    
    ulong t2 = t0 + t1;
    t2 += select(0ul, 0xFFFFFFFFul, t2 < t0);

    return select(t2, t2 - P_GOLD, t2 >= P_GOLD);
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

    ulong tmp[T];

    // Pre-multiply by external MDS.
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong acc = 0ul;
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            acc = gold_add(acc, gold_mul(ext_mds[i * T + j], state[j]));
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
                acc = gold_add(acc, gold_mul(ext_mds[i * T + j], state[j]));
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

        // Apply internal diagonal matrix in-place
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = gold_add(s, gold_mul(int_diag[i], state[i]));
        }
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
                acc = gold_add(acc, gold_mul(ext_mds[i * T + j], state[j]));
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
```