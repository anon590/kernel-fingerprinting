The previous attempt failed because `half` is a reserved type name in MSL (the `half` float type). I'll rename it and rebuild with the t=3 specialized fast path plus a key optimization: **specialize the partial-round matvec** which is the inner-loop bottleneck (22 rounds × 3 elements). The internal MDS is `y[i] = sum + d[i]*s[i]`, so we save a multiply when `d[i]` is small but more importantly we can fuse the structure. Also, I'll keep MDS coefficients in registers (not threadgroup memory) for t=3 to reduce memory traffic in the inner loop, and inline a hand-unrolled t=3 external matvec.

```metal
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

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = x_hi_lo * EPSILON;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
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
    return gold_reduce128(lo, hi);
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

    // Load MDS / diag into registers
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

    // Load state
    ulong s0=0, s1=0, s2=0, s3=0;
    s0 = in_state[idx*tl + 0];
    if (tl > 1) s1 = in_state[idx*tl + 1];
    if (tl > 2) s2 = in_state[idx*tl + 2];
    if (tl > 3) s3 = in_state[idx*tl + 3];

    // Specialized t=3 fast path
    if (tl == 3) {
        // Pre-mul by external MDS
        {
            ulong n0 = gold_add(gold_add(gold_mul(m00,s0), gold_mul(m01,s1)), gold_mul(m02,s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10,s0), gold_mul(m11,s1)), gold_mul(m12,s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20,s0), gold_mul(m21,s1)), gold_mul(m22,s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // First half full rounds
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = gold_add(gold_add(gold_mul(m00,s0), gold_mul(m01,s1)), gold_mul(m02,s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10,s0), gold_mul(m11,s1)), gold_mul(m12,s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20,s0), gold_mul(m21,s1)), gold_mul(m22,s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Partial rounds: y[i] = sum + d[i]*s[i]
        for (uint r = 0u; r < rp_local; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = gold_add(sum, gold_mul(d0, s0));
            ulong n1 = gold_add(sum, gold_mul(d1, s1));
            ulong n2 = gold_add(sum, gold_mul(d2, s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Second half full rounds
        for (uint r = half_f; r < rf_local; ++r) {
            s0 = sbox(gold_add(s0, tg_rc_ext[r*3 + 0]));
            s1 = sbox(gold_add(s1, tg_rc_ext[r*3 + 1]));
            s2 = sbox(gold_add(s2, tg_rc_ext[r*3 + 2]));
            ulong n0 = gold_add(gold_add(gold_mul(m00,s0), gold_mul(m01,s1)), gold_mul(m02,s2));
            ulong n1 = gold_add(gold_add(gold_mul(m10,s0), gold_mul(m11,s1)), gold_mul(m12,s2));
            ulong n2 = gold_add(gold_add(gold_mul(m20,s0), gold_mul(m21,s1)), gold_mul(m22,s2));
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx*3 + 0] = s0;
        out_state[idx*3 + 1] = s1;
        out_state[idx*3 + 2] = s2;
        return;
    }

    // Generic path
    {
        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < half_f; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = 0u; r < rp_local; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_int[r]));
        ulong sum = s0;
        if (tl > 1) sum = gold_add(sum, s1);
        if (tl > 2) sum = gold_add(sum, s2);
        if (tl > 3) sum = gold_add(sum, s3);
        ulong n0 = gold_add(sum, gold_mul(d0, s0));
        ulong n1 = (tl > 1) ? gold_add(sum, gold_mul(d1, s1)) : 0ul;
        ulong n2 = (tl > 2) ? gold_add(sum, gold_mul(d2, s2)) : 0ul;
        ulong n3 = (tl > 3) ? gold_add(sum, gold_mul(d3, s3)) : 0ul;
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    for (uint r = half_f; r < rf_local; ++r) {
        s0 = sbox(gold_add(s0, tg_rc_ext[r*tl + 0]));
        if (tl > 1) s1 = sbox(gold_add(s1, tg_rc_ext[r*tl + 1]));
        if (tl > 2) s2 = sbox(gold_add(s2, tg_rc_ext[r*tl + 2]));
        if (tl > 3) s3 = sbox(gold_add(s3, tg_rc_ext[r*tl + 3]));

        ulong n0=0,n1=0,n2=0,n3=0;
        n0 = gold_mul(m00, s0);
        if (tl > 1) n0 = gold_add(n0, gold_mul(m01, s1));
        if (tl > 2) n0 = gold_add(n0, gold_mul(m02, s2));
        if (tl > 3) n0 = gold_add(n0, gold_mul(m03, s3));
        if (tl > 1) {
            n1 = gold_mul(m10, s0);
            n1 = gold_add(n1, gold_mul(m11, s1));
            if (tl > 2) n1 = gold_add(n1, gold_mul(m12, s2));
            if (tl > 3) n1 = gold_add(n1, gold_mul(m13, s3));
        }
        if (tl > 2) {
            n2 = gold_mul(m20, s0);
            n2 = gold_add(n2, gold_mul(m21, s1));
            n2 = gold_add(n2, gold_mul(m22, s2));
            if (tl > 3) n2 = gold_add(n2, gold_mul(m23, s3));
        }
        if (tl > 3) {
            n3 = gold_mul(m30, s0);
            n3 = gold_add(n3, gold_mul(m31, s1));
            n3 = gold_add(n3, gold_mul(m32, s2));
            n3 = gold_add(n3, gold_mul(m33, s3));
        }
        s0=n0; s1=n1; s2=n2; s3=n3;
    }

    out_state[idx*tl + 0] = s0;
    if (tl > 1) out_state[idx*tl + 1] = s1;
    if (tl > 2) out_state[idx*tl + 2] = s2;
    if (tl > 3) out_state[idx*tl + 3] = s3;
}
```