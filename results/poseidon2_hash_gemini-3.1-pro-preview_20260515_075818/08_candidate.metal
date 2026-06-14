#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * b0;
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = p01 + p10 + (p00 >> 32);
    ulong lo = (mid << 32) | (uint)p00;
    ulong hi = p11 + (mid >> 32);
    
    return ulong2(lo, hi);
}

inline ulong2 usqr128(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    ulong p00 = (ulong)a0 * a0;
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;

    ulong mid = (p01 << 1) + (p00 >> 32);
    ulong lo = (mid << 32) | (uint)p00;
    ulong hi = p11 + (mid >> 32);
    
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = (uint)x_hi;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    t0 -= (t0 > x_lo) ? 0xFFFFFFFFul : 0ul;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;

    return (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

inline ulong gold_sqr(ulong a) {
    ulong2 p = usqr128(a);
    return gold_reduce128(p.x, p.y);
}

inline ulong sbox(ulong x) {
    // x^7 = x^4 * x^2 * x
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint T>
inline void apply_matvec_ext(thread ulong *state, thread const ulong *mds) {
    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong acc_lo = 0;
        ulong sum_hi_lo = 0;
        ulong sum_hi_hi = 0;
        
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            ulong2 p = umul128(mds[i * T + j], state[j]);
            
            ulong next_lo = acc_lo + p.x;
            sum_hi_lo += (next_lo < acc_lo) ? 1 : 0;
            acc_lo = next_lo;
            
            sum_hi_lo += (uint)p.y;
            sum_hi_hi += (p.y >> 32);
        }
        
        ulong L0 = (uint)sum_hi_lo;
        ulong L1 = sum_hi_lo >> 32;
        ulong S = L1 + sum_hi_hi;
        
        // Final reduction logic using 2^96 == -1 (mod P)
        ulong t1 = (L0 << 32) - L0;
        ulong t0 = acc_lo - S;
        t0 -= (t0 > acc_lo) ? 0xFFFFFFFFul : 0ul;
        
        ulong t2 = t0 + t1;
        t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;
        
        tmp[i] = (t2 >= P_GOLD) ? (t2 - P_GOLD) : t2;
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
}

template <uint T>
inline void apply_matvec_int(thread ulong *state, thread const ulong *diag) {
    ulong sum_lo = state[0];
    uint sum_hi = 0;
    #pragma unroll
    for (uint i = 1; i < T; ++i) {
        ulong next = sum_lo + state[i];
        sum_hi += (next < sum_lo) ? 1 : 0;
        sum_lo = next;
    }
    
    ulong t1_sum = ((ulong)sum_hi << 32) - sum_hi;
    ulong t2_sum = sum_lo + t1_sum;
    t2_sum += (t2_sum < sum_lo) ? 0xFFFFFFFFul : 0ul;
    ulong sum = (t2_sum >= P_GOLD) ? (t2_sum - P_GOLD) : t2_sum;

    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong2 p = umul128(diag[i], state[i]);
        ulong lo = p.x + sum;
        ulong hi = p.y + ((lo < p.x) ? 1 : 0);
        tmp[i] = gold_reduce128(lo, hi);
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
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
        state[i] = in_state[idx * T + i];
    }

    ulong mds[T * T];
    #pragma unroll
    for (uint i = 0; i < T * T; ++i) {
        mds[i] = ext_mds[i];
    }

    ulong diag[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        diag[i] = int_diag[i];
    }

    apply_matvec_ext<T>(state, mds);

    uint half_f = r_f >> 1;
    device const ulong *rc_ext_ptr = rc_ext;

    for (uint r = 0; r < half_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext_ptr[i]));
        }
        rc_ext_ptr += T;
        apply_matvec_ext<T>(state, mds);
    }

    device const ulong *rc_int_ptr = rc_int;
    
    for (uint r = 0; r < r_p; ++r) {
        state[0] = sbox(gold_add(state[0], rc_int_ptr[r]));
        apply_matvec_int<T>(state, diag);
    }

    for (uint r = half_f; r < r_f; ++r) {
        #pragma unroll
        for (uint i = 0; i < T; ++i) {
            state[i] = sbox(gold_add(state[i], rc_ext_ptr[i]));
        }
        rc_ext_ptr += T;
        apply_matvec_ext<T>(state, mds);
    }

    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        out_state[idx * T + i] = state[i];
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

    switch (t) {
        case 1: process_sponge<1>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 2: process_sponge<2>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 3: process_sponge<3>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
        case 4: process_sponge<4>(in_state, out_state, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p, idx); break;
    }
}