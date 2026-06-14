#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? (t - P_GOLD) : t;
}

inline ulong2 umul128(ulong a, ulong b) {
    ulong lo = a * b;
    
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_hi = mulhi(a0, b0);
    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p11 = (ulong)a1 * b1;
    
    // Ordered to gracefully handle what would be a >64-bit overflow 
    ulong mid1 = p10 + p00_hi;
    ulong hi = p11 + (mid1 >> 32);
    ulong mid2 = (uint)mid1 + p01;
    hi += (mid2 >> 32);
    
    return ulong2(lo, hi);
}

inline ulong2 usqr128(ulong a) {
    ulong lo = a * a;
    
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00_hi = mulhi(a0, a0);
    ulong p01 = (ulong)a0 * a1;
    ulong p11 = (ulong)a1 * a1;
    
    ulong p01_lo = (uint)p01;
    ulong p01_hi = p01 >> 32;

    ulong mid = (p01_lo << 1) + p00_hi;
    ulong hi = p11 + (p01_hi << 1) + (mid >> 32);
    
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

inline ulong sbox(ulong x) {
    ulong2 x2_128 = usqr128(x);
    ulong x2 = gold_reduce128(x2_128.x, x2_128.y);
    
    ulong2 x4_128 = usqr128(x2);
    ulong x4 = gold_reduce128(x4_128.x, x4_128.y);
    
    ulong2 x6_128 = umul128(x4, x2);
    ulong x6 = gold_reduce128(x6_128.x, x6_128.y);
    
    ulong2 x7_128 = umul128(x6, x);
    return gold_reduce128(x7_128.x, x7_128.y);
}

template <uint T>
inline void apply_matvec_ext(thread ulong *state, thread const ulong *mds) {
    ulong tmp[T];
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        ulong acc_lo = 0;
        ulong acc_hi = 0;
        ulong acc_top = 0;
        
        #pragma unroll
        for (uint j = 0; j < T; ++j) {
            ulong2 p = umul128(mds[i * T + j], state[j]);
            
            ulong next_lo = acc_lo + p.x;
            ulong c1 = (next_lo < acc_lo) ? 1 : 0;
            acc_lo = next_lo;
            
            ulong next_hi = acc_hi + p.y;
            ulong c2 = (next_hi < acc_hi) ? 1 : 0;
            next_hi += c1;
            c2 += (next_hi < c1) ? 1 : 0;
            acc_hi = next_hi;
            
            acc_top += c2;
        }
        
        ulong rem = gold_reduce128(acc_lo, acc_hi);
        ulong sub = acc_top << 32;
        ulong rem2 = rem - sub;
        rem2 += (rem < sub) ? P_GOLD : 0;
        tmp[i] = rem2;
    }
    #pragma unroll
    for (uint i = 0; i < T; ++i) state[i] = tmp[i];
}

template <uint T>
inline void apply_matvec_int(thread ulong *state, thread const ulong *diag) {
    ulong sum = 0;
    #pragma unroll
    for (uint i = 0; i < T; ++i) {
        sum = gold_add(sum, state[i]);
    }
    
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