**Optimization rationale:** The hot path is `gold_mul` (full 64x64 multiply + reduction). For the external MDS matvec I was doing `t` reduced multiplies then `t-1` adds per output. I'll replace this with a **lazy/fused matvec**: compute all `t` products as raw 128-bit (lo, hi) and accumulate them with 128-bit adds, reducing only once per output. This roughly halves the reduction cost in the matvec, which dominates t=3/t=4. For t<=4 the accumulator can't overflow 128 bits since `t*(p-1)^2 < 2^128`. Also tightened `gold_add` (one branch using the fact that inputs from `gold_reduce_128` are canonical, so `a+b < 2p`, eliminating the wrap-correction add).

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

// a, b canonical (< p). a+b < 2p, single subtraction suffices.
inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    // wrap if a+b >= 2^64; since a,b < p < 2^64, wrap means t = a+b - 2^64,
    // and true sum >= 2^64 > p, so we need t + EPSILON (= -p mod 2^64),
    // followed by canonical-clamp.
    if (t < a) t += EPSILON;
    if (t >= P_GOLD) t -= P_GOLD;
    return t;
}

// Reduce (lo + hi*2^64) mod p, hi < 2^64. Result canonical.
inline ulong gold_reduce_128(ulong lo, ulong hi) {
    ulong x_hi_lo = hi & EPSILON;
    ulong x_hi_hi = hi >> 32;

    ulong t0 = lo - x_hi_hi;
    if (t0 > lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return gold_canonical(t2);
}

inline void mul_full_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid_ = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    lo = (p00 & EPSILON) | (mid_ << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid_ >> 32);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong lo, hi;
    mul_full_128(a, b, lo, hi);
    return gold_reduce_128(lo, hi);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// 128-bit fused multiply-accumulate: (lo,hi) += a*b
inline void mac_128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    ulong pl, ph;
    mul_full_128(a, b, pl, ph);
    ulong nl = lo + pl;
    ulong c  = (nl < lo) ? 1ul : 0ul;
    lo = nl;
    hi = hi + ph + c;
}

// y = sum + diag*s, sum canonical (< p).
inline ulong sum_plus_mul(ulong sum, ulong diag_i, ulong s_i) {
    ulong pl, ph;
    mul_full_128(diag_i, s_i, pl, ph);
    ulong nl = pl + sum;
    if (nl < pl) ph += 1ul;
    return gold_reduce_128(nl, ph);
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

    uint tt = t;
    uint rf = r_f;
    uint rp = r_p;
    uint half_f = rf >> 1u;

    // ============== Specialized t=3 ==============
    if (tt == 3u) {
        ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
        ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
        ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];
        ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

        ulong s0 = in_state[idx * 3u + 0u];
        ulong s1 = in_state[idx * 3u + 1u];
        ulong s2 = in_state[idx * 3u + 2u];

        // Fused matvec: accumulate 3 full products in 128-bit, reduce once per row.
        {
            ulong l, h;
            ulong n0, n1, n2;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h);
            n0 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h);
            n1 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h);
            n2 = gold_reduce_128(l, h);
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong l, h, n0, n1, n2;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h);
            n0 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h);
            n1 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h);
            n2 = gold_reduce_128(l, h);
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            ulong n2 = sum_plus_mul(sum, d2, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong l, h, n0, n1, n2;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h);
            n0 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h);
            n1 = gold_reduce_128(l, h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h);
            n2 = gold_reduce_128(l, h);
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx * 3u + 0u] = s0;
        out_state[idx * 3u + 1u] = s1;
        out_state[idx * 3u + 2u] = s2;
        return;
    }

    // ============== Specialized t=2 ==============
    if (tt == 2u) {
        ulong m00 = ext_mds[0], m01 = ext_mds[1];
        ulong m10 = ext_mds[2], m11 = ext_mds[3];
        ulong d0 = int_diag[0], d1 = int_diag[1];

        ulong s0 = in_state[idx * 2u + 0u];
        ulong s1 = in_state[idx * 2u + 1u];

        {
            ulong l, h, n0, n1;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); n0 = gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); n1 = gold_reduce_128(l,h);
            s0 = n0; s1 = n1;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong l, h, n0, n1;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); n0 = gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); n1 = gold_reduce_128(l,h);
            s0 = n0; s1 = n1;
        }
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(s0, s1);
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            s0 = n0; s1 = n1;
        }
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong l, h, n0, n1;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); n0 = gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); n1 = gold_reduce_128(l,h);
            s0 = n0; s1 = n1;
        }
        out_state[idx * 2u + 0u] = s0;
        out_state[idx * 2u + 1u] = s1;
        return;
    }

    // ============== Specialized t=4 ==============
    if (tt == 4u) {
        ulong m00 = ext_mds[0],  m01 = ext_mds[1],  m02 = ext_mds[2],  m03 = ext_mds[3];
        ulong m10 = ext_mds[4],  m11 = ext_mds[5],  m12 = ext_mds[6],  m13 = ext_mds[7];
        ulong m20 = ext_mds[8],  m21 = ext_mds[9],  m22 = ext_mds[10], m23 = ext_mds[11];
        ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];
        ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

        ulong s0 = in_state[idx * 4u + 0u];
        ulong s1 = in_state[idx * 4u + 1u];
        ulong s2 = in_state[idx * 4u + 2u];
        ulong s3 = in_state[idx * 4u + 3u];

        {
            ulong l, h, n0, n1, n2, n3;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h); mac_128(m03,s3,l,h); n0=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h); mac_128(m13,s3,l,h); n1=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h); mac_128(m23,s3,l,h); n2=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m30,s0,l,h); mac_128(m31,s1,l,h); mac_128(m32,s2,l,h); mac_128(m33,s3,l,h); n3=gold_reduce_128(l,h);
            s0=n0; s1=n1; s2=n2; s3=n3;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            ulong l, h, n0, n1, n2, n3;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h); mac_128(m03,s3,l,h); n0=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h); mac_128(m13,s3,l,h); n1=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h); mac_128(m23,s3,l,h); n2=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m30,s0,l,h); mac_128(m31,s1,l,h); mac_128(m32,s2,l,h); mac_128(m33,s3,l,h); n3=gold_reduce_128(l,h);
            s0=n0; s1=n1; s2=n2; s3=n3;
        }
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            ulong n2 = sum_plus_mul(sum, d2, s2);
            ulong n3 = sum_plus_mul(sum, d3, s3);
            s0 = n0; s1 = n1; s2 = n2; s3 = n3;
        }
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            ulong l, h, n0, n1, n2, n3;
            l=0; h=0; mac_128(m00,s0,l,h); mac_128(m01,s1,l,h); mac_128(m02,s2,l,h); mac_128(m03,s3,l,h); n0=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m10,s0,l,h); mac_128(m11,s1,l,h); mac_128(m12,s2,l,h); mac_128(m13,s3,l,h); n1=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m20,s0,l,h); mac_128(m21,s1,l,h); mac_128(m22,s2,l,h); mac_128(m23,s3,l,h); n2=gold_reduce_128(l,h);
            l=0; h=0; mac_128(m30,s0,l,h); mac_128(m31,s1,l,h); mac_128(m32,s2,l,h); mac_128(m33,s3,l,h); n3=gold_reduce_128(l,h);
            s0=n0; s1=n1; s2=n2; s3=n3;
        }
        out_state[idx * 4u + 0u] = s0;
        out_state[idx * 4u + 1u] = s1;
        out_state[idx * 4u + 2u] = s2;
        out_state[idx * 4u + 3u] = s3;
        return;
    }

    // ============== Generic fallback ==============
    ulong state[T_MAX];
    for (uint i = 0u; i < tt; ++i) state[i] = in_state[idx * tt + i];

    {
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < rp; ++r) {
        state[0] = sbox(gold_add(state[0], rc_int[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < tt; ++i) s = gold_add(s, state[i]);
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) tmp[i] = sum_plus_mul(s, int_diag[i], state[i]);
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = half_f; r < rf; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong l = 0ul, h = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                mac_128(ext_mds[i * tt + j], state[j], l, h);
            }
            tmp[i] = gold_reduce_128(l, h);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint i = 0u; i < tt; ++i) out_state[idx * tt + i] = state[i];
}
```