#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0xFFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    ulong carry = (t < a) ? EPSILON : 0ul;
    t += carry;
    ulong over = (t >= P_GOLD) ? P_GOLD : 0ul;
    t -= over;
    return t;
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
    ulong lo  = ((mid & EPSILON) << 32) | (uint)p00;
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong x_hi_lo = (uint)hi;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    ulong under = (t0 > lo) ? EPSILON : 0ul;
    t0 -= under;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    ulong carry = (t2 < t0) ? EPSILON : 0ul;
    t2 += carry;

    ulong over = (t2 >= P_GOLD) ? P_GOLD : 0ul;
    t2 -= over;

    return t2;
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

template <uint t>
inline void matvec_ext_unrolled(thread ulong& s0, thread ulong& s1, thread ulong& s2, thread ulong& s3,
                                threadgroup const ulong* mds) {
    if (t == 4) {
        ulong n0 = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_add(gold_mul(mds[2], s2), gold_mul(mds[3], s3)));
        ulong n1 = gold_add(gold_add(gold_mul(mds[4], s0), gold_mul(mds[5], s1)), gold_add(gold_mul(mds[6], s2), gold_mul(mds[7], s3)));
        ulong n2 = gold_add(gold_add(gold_mul(mds[8], s0), gold_mul(mds[9], s1)), gold_add(gold_mul(mds[10], s2), gold_mul(mds[11], s3)));
        ulong n3 = gold_add(gold_add(gold_mul(mds[12], s0), gold_mul(mds[13], s1)), gold_add(gold_mul(mds[14], s2), gold_mul(mds[15], s3)));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    } else if (t == 3) {
        ulong n0 = gold_add(gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1)), gold_mul(mds[2], s2));
        ulong n1 = gold_add(gold_add(gold_mul(mds[3], s0), gold_mul(mds[4], s1)), gold_mul(mds[5], s2));
        ulong n2 = gold_add(gold_add(gold_mul(mds[6], s0), gold_mul(mds[7], s1)), gold_mul(mds[8], s2));
        s0 = n0; s1 = n1; s2 = n2;
    } else if (t == 2) {
        ulong n0 = gold_add(gold_mul(mds[0], s0), gold_mul(mds[1], s1));
        ulong n1 = gold_add(gold_mul(mds[2], s0), gold_mul(mds[3], s1));
        s0 = n0; s1 = n1;
    } else if (t == 1) {
        s0 = gold_mul(mds[0], s0);
    }
}

template <uint t>
inline void matvec_int_unrolled(thread ulong& s0, thread ulong& s1, thread ulong& s2, thread ulong& s3,
                                threadgroup const ulong* diag) {
    if (t == 4) {
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        ulong n0 = gold_add(sum, gold_mul(diag[0], s0));
        ulong n1 = gold_add(sum, gold_mul(diag[1], s1));
        ulong n2 = gold_add(sum, gold_mul(diag[2], s2));
        ulong n3 = gold_add(sum, gold_mul(diag[3], s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    } else if (t == 3) {
        ulong sum = gold_add(gold_add(s0, s1), s2);
        ulong n0 = gold_add(sum, gold_mul(diag[0], s0));
        ulong n1 = gold_add(sum, gold_mul(diag[1], s1));
        ulong n2 = gold_add(sum, gold_mul(diag[2], s2));
        s0 = n0; s1 = n1; s2 = n2;
    } else if (t == 2) {
        ulong sum = gold_add(s0, s1);
        ulong n0 = gold_add(sum, gold_mul(diag[0], s0));
        ulong n1 = gold_add(sum, gold_mul(diag[1], s1));
        s0 = n0; s1 = n1;
    } else if (t == 1) {
        ulong sum = s0;
        s0 = gold_add(sum, gold_mul(diag[0], s0));
    }
}

template <uint t>
inline void process_sponge(device const ulong* in_state,
                           device       ulong* out_state,
                           threadgroup const ulong* tg_rc_ext,
                           threadgroup const ulong* tg_rc_int,
                           threadgroup const ulong* tg_ext_mds,
                           threadgroup const ulong* tg_int_diag,
                           uint r_f, uint r_p)
{
    ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    if (t > 0) s0 = in_state[0];
    if (t > 1) s1 = in_state[1];
    if (t > 2) s2 = in_state[2];
    if (t > 3) s3 = in_state[3];

    matvec_ext_unrolled<t>(s0, s1, s2, s3, tg_ext_mds);

    uint half_f = r_f >> 1;

    for (uint r = 0; r < half_f; ++r) {
        if (t > 0) s0 = sbox(gold_add(s0, tg_rc_ext[r * t + 0]));
        if (t > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r * t + 1]));
        if (t > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r * t + 2]));
        if (t > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r * t + 3]));
        matvec_ext_unrolled<t>(s0, s1, s2, s3, tg_ext_mds);
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_int[r]));
        matvec_int_unrolled<t>(s0, s1, s2, s3, tg_int_diag);
    }

    for (uint r = half_f; r < r_f; ++r) {
        if (t > 0) s0 = sbox(gold_add(s0, tg_rc_ext[r * t + 0]));
        if (t > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r * t + 1]));
        if (t > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r * t + 2]));
        if (t > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r * t + 3]));
        matvec_ext_unrolled<t>(s0, s1, s2, s3, tg_ext_mds);
    }

    if (t > 0) out_state[0] = s0;
    if (t > 1) out_state[1] = s1;
    if (t > 2) out_state[2] = s2;
    if (t > 3) out_state[3] = s3;
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
    uint idx [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsz [[threads_per_threadgroup]])
{
    uint t_loc = t;
    uint r_f_loc = r_f;
    uint r_p_loc = r_p;

    threadgroup ulong tg_ext_mds[T_MAX * T_MAX];
    threadgroup ulong tg_int_diag[T_MAX];
    threadgroup ulong tg_rc_ext[8 * T_MAX];
    threadgroup ulong tg_rc_int[32];

    for (uint i = tid; i < t_loc * t_loc; i += tgsz) tg_ext_mds[i] = ext_mds[i];
    for (uint i = tid; i < t_loc; i += tgsz)         tg_int_diag[i] = int_diag[i];
    for (uint i = tid; i < r_f_loc * t_loc; i += tgsz) tg_rc_ext[i] = rc_ext[i];
    for (uint i = tid; i < r_p_loc; i += tgsz)         tg_rc_int[i] = rc_int[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx >= batch) return;

    if (t_loc == 4) {
        process_sponge<4>(in_state + idx * 4, out_state + idx * 4, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    } else if (t_loc == 3) {
        process_sponge<3>(in_state + idx * 3, out_state + idx * 3, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    } else if (t_loc == 2) {
        process_sponge<2>(in_state + idx * 2, out_state + idx * 2, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    } else if (t_loc == 1) {
        process_sponge<1>(in_state + idx * 1, out_state + idx * 1, tg_rc_ext, tg_rc_int, tg_ext_mds, tg_int_diag, r_f_loc, r_p_loc);
    }
}