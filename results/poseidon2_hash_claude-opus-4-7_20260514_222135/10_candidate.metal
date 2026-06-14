#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;
constexpr constant uint RF_MAX = 8u;
constexpr constant uint RP_MAX = 32u;

inline ulong gold_canon(ulong x) {
    return select(x, x - P_GOLD, x >= P_GOLD);
}

// Lazy add: branchless carry correction.
inline ulong gold_add_lazy(ulong a, ulong b) {
    ulong t = a + b;
    ulong c = select(0ul, EPSILON, t < a);
    return t + c;
}

// Reduce (lo + hi*2^64) mod p, branchless.
inline ulong gold_reduce128_lazy(ulong lo, ulong hi) {
    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    ulong t0 = lo - hi_hi;
    ulong b0 = select(0ul, EPSILON, t0 > lo);
    t0 -= b0;

    ulong t1 = (hi_lo << 32) - hi_lo;

    ulong t2 = t0 + t1;
    ulong c1 = select(0ul, EPSILON, t2 < t0);
    return t2 + c1;
}

inline ulong gold_mul_lazy(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return gold_reduce128_lazy(lo, hi);
}

inline ulong gold_muladd_lazy(ulong a, ulong b, ulong c) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    ulong lo2 = lo + c;
    ulong carry = select(0ul, 1ul, lo2 < lo);
    return gold_reduce128_lazy(lo2, hi + carry);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul_lazy(x, x);
    ulong x4 = gold_mul_lazy(x2, x2);
    ulong x6 = gold_mul_lazy(x4, x2);
    return gold_mul_lazy(x6, x);
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
    uint idx        [[thread_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint tg_sz      [[threads_per_threadgroup]])
{
    threadgroup ulong tg_ext_mds[T_MAX * T_MAX];
    threadgroup ulong tg_int_diag[T_MAX];
    threadgroup ulong tg_rc_ext[RF_MAX * T_MAX];
    threadgroup ulong tg_rc_int[RP_MAX];

    uint tl       = t;
    uint rf_local = r_f;
    uint rp_local = r_p;
    uint half_f   = rf_local >> 1u;

    uint ext_mds_sz = tl * tl;
    uint rc_ext_sz  = rf_local * tl;

    for (uint i = lid; i < ext_mds_sz; i += tg_sz) tg_ext_mds[i]  = ext_mds[i];
    for (uint i = lid; i < tl;         i += tg_sz) tg_int_diag[i] = int_diag[i];
    for (uint i = lid; i < rc_ext_sz;  i += tg_sz) tg_rc_ext[i]   = rc_ext[i];
    for (uint i = lid; i < rp_local;   i += tg_sz) tg_rc_int[i]   = rc_int[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (idx >= batch) return;

    ulong m00=0,m01=0,m02=0,m03=0;
    ulong m10=0,m11=0,m12=0,m13=0;
    ulong m20=0,m21=0,m22=0,m23=0;
    ulong m30=0,m31=0,m32=0,m33=0;
    ulong d0=0,d1=0,d2=0,d3=0;

    m00 = tg_ext_mds[0];
    if (tl > 1) m01 = tg_ext_mds[0*tl + 1];
    if (tl > 2) m02 = tg_ext_mds[0*tl + 2];
    if (tl > 3) m03 = tg_ext_mds[0*tl + 3];
    if (tl > 1) {
        m10 = tg_ext_mds[1*tl + 0];
        m11 = tg_ext_mds[1*tl + 1];
        if (tl > 2) m12 = tg_ext_mds[1*tl + 2];
        if (tl > 3) m13 = tg_ext_mds[1*tl + 3];
    }
    if (tl > 2) {
        m20 = tg_ext_mds[2*tl + 0];
        m21 = tg_ext_mds[2*tl + 1];
        m22 = tg_ext_mds[2*tl + 2];
        if (tl > 3) m23 = tg_ext_mds[2*tl + 3];
    }
    if (tl > 3) {
        m30 = tg_ext_mds[3*tl + 0];
        m31 = tg_ext_mds[3*tl + 1];
        m32 = tg_ext_mds[3*tl + 2];
        m33 = tg_ext_mds[3*tl + 3];
    }
    d0 = tg_int_diag[0];
    if (tl > 1) d1 = tg_int_diag[1];
    if (tl > 2) d2 = tg_int_diag[2];
    if (tl > 3) d3 = tg_int_diag[3];

    ulong s0=0, s1=0, s2=0, s3=0;
    s0 = in_state[idx*tl + 0];
    if (tl > 1) s1 = in_state[idx*tl + 1];
    if (tl > 2) s2 = in_state[idx*tl + 2];
    if (tl > 3) s3 = in_state[idx*tl + 3];

    if (tl == 3) {
        {
            ulong n0 = gold_muladd_lazy(m02, s2, gold_muladd_lazy(m01, s1, gold_mul_lazy(m00, s0)));
            ulong n1 = gold_muladd_lazy(m12, s2, gold_muladd_lazy(m11, s1, gold_mul_lazy(m10, s0)));
            ulong n2 = gold_muladd_lazy(m22, s2, gold_muladd_lazy(m21, s1, gold_mul_lazy(m20, s0)));
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add_lazy(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add_lazy(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add_lazy(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = gold_muladd_lazy(m02, s2, gold_muladd_lazy(m01, s1, gold_mul_lazy(m00, s0)));
            ulong n1 = gold_muladd_lazy(m12, s2, gold_muladd_lazy(m11, s1, gold_mul_lazy(m10, s0)));
            ulong n2 = gold_muladd_lazy(m22, s2, gold_muladd_lazy(m21, s1, gold_mul_lazy(m20, s0)));
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = 0u; r < rp_local; ++r) {
            s0 = sbox(gold_add_lazy(s0, tg_rc_int[r]));
            ulong sum = gold_add_lazy(gold_add_lazy(s0, s1), s2);
            ulong n0 = gold_muladd_lazy(d0, s0, sum);
            ulong n1 = gold_muladd_lazy(d1, s1, sum);
            ulong n2 = gold_muladd_lazy(d2, s2, sum);
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = half_f; r < rf_local; ++r) {
            s0 = sbox(gold_add_lazy(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add_lazy(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add_lazy(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = gold_muladd_lazy(m02, s2, gold_muladd_lazy(m01, s1, gold_mul_lazy(m00, s0)));
            ulong n1 = gold_muladd_lazy(m12, s2, gold_muladd_lazy(m11, s1, gold_mul_lazy(m10, s0)));
            ulong n2 = gold_muladd_lazy(m22, s2, gold_muladd_lazy(m21, s1, gold_mul_lazy(m20, s0)));
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx*3 + 0] = gold_canon(s0);
        out_state[idx*3 + 1] = gold_canon(s1);
        out_state[idx*3 + 2] = gold_canon(s2);
        return;
    }

    {
        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul_lazy(m00, s0);
        if (tl > 1) n0 = gold_muladd_lazy(m01, s1, n0);
        if (tl > 2) n0 = gold_muladd_lazy(m02, s2, n0);
        if (tl > 3) n0 = gold_muladd_lazy(m03, s3, n0);
        if (tl > 1) {
            n1 = gold_mul_lazy(m10, s0);
            n1 = gold_muladd_lazy(m11, s1, n1);
            if (tl > 2) n1 = gold_muladd_lazy(m12, s2, n1);
            if (tl > 3) n1 = gold_muladd_lazy(m13, s3, n1);
        }
        if (tl > 2) {
            n2 = gold_mul_lazy(m20, s0);
            n2 = gold_muladd_lazy(m21, s1, n2);
            n2 = gold_muladd_lazy(m22, s2, n2);
            if (tl > 3) n2 = gold_muladd_lazy(m23, s3, n2);
        }
        if (tl > 3) {
            n3 = gold_mul_lazy(m30, s0);
            n3 = gold_muladd_lazy(m31, s1, n3);
            n3 = gold_muladd_lazy(m32, s2, n3);
            n3 = gold_muladd_lazy(m33, s3, n3);
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < half_f; ++r) {
        s0 = sbox(gold_add_lazy(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add_lazy(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add_lazy(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add_lazy(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul_lazy(m00, s0);
        if (tl > 1) n0 = gold_muladd_lazy(m01, s1, n0);
        if (tl > 2) n0 = gold_muladd_lazy(m02, s2, n0);
        if (tl > 3) n0 = gold_muladd_lazy(m03, s3, n0);
        if (tl > 1) {
            n1 = gold_mul_lazy(m10, s0);
            n1 = gold_muladd_lazy(m11, s1, n1);
            if (tl > 2) n1 = gold_muladd_lazy(m12, s2, n1);
            if (tl > 3) n1 = gold_muladd_lazy(m13, s3, n1);
        }
        if (tl > 2) {
            n2 = gold_mul_lazy(m20, s0);
            n2 = gold_muladd_lazy(m21, s1, n2);
            n2 = gold_muladd_lazy(m22, s2, n2);
            if (tl > 3) n2 = gold_muladd_lazy(m23, s3, n2);
        }
        if (tl > 3) {
            n3 = gold_mul_lazy(m30, s0);
            n3 = gold_muladd_lazy(m31, s1, n3);
            n3 = gold_muladd_lazy(m32, s2, n3);
            n3 = gold_muladd_lazy(m33, s3, n3);
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < rp_local; ++r) {
        s0 = sbox(gold_add_lazy(s0, tg_rc_int[r]));
        ulong sum = s0;
        if (tl > 1) sum = gold_add_lazy(sum, s1);
        if (tl > 2) sum = gold_add_lazy(sum, s2);
        if (tl > 3) sum = gold_add_lazy(sum, s3);
        ulong n0 = gold_muladd_lazy(d0, s0, sum);
        ulong n1 = (tl > 1) ? gold_muladd_lazy(d1, s1, sum) : 0ul;
        ulong n2 = (tl > 2) ? gold_muladd_lazy(d2, s2, sum) : 0ul;
        ulong n3 = (tl > 3) ? gold_muladd_lazy(d3, s3, sum) : 0ul;
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = half_f; r < rf_local; ++r) {
        s0 = sbox(gold_add_lazy(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add_lazy(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add_lazy(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add_lazy(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul_lazy(m00, s0);
        if (tl > 1) n0 = gold_muladd_lazy(m01, s1, n0);
        if (tl > 2) n0 = gold_muladd_lazy(m02, s2, n0);
        if (tl > 3) n0 = gold_muladd_lazy(m03, s3, n0);
        if (tl > 1) {
            n1 = gold_mul_lazy(m10, s0);
            n1 = gold_muladd_lazy(m11, s1, n1);
            if (tl > 2) n1 = gold_muladd_lazy(m12, s2, n1);
            if (tl > 3) n1 = gold_muladd_lazy(m13, s3, n1);
        }
        if (tl > 2) {
            n2 = gold_mul_lazy(m20, s0);
            n2 = gold_muladd_lazy(m21, s1, n2);
            n2 = gold_muladd_lazy(m22, s2, n2);
            if (tl > 3) n2 = gold_muladd_lazy(m23, s3, n2);
        }
        if (tl > 3) {
            n3 = gold_mul_lazy(m30, s0);
            n3 = gold_muladd_lazy(m31, s1, n3);
            n3 = gold_muladd_lazy(m32, s2, n3);
            n3 = gold_muladd_lazy(m33, s3, n3);
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    out_state[idx*tl + 0] = gold_canon(s0);
    if (tl > 1) out_state[idx*tl + 1] = gold_canon(s1);
    if (tl > 2) out_state[idx*tl + 2] = gold_canon(s2);
    if (tl > 3) out_state[idx*tl + 3] = gold_canon(s3);
}