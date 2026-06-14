#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX  = 4u;
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

// Reduce a 128-bit value (lo, hi) modulo Goldilocks. Result canonical.
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

// Reduce a 192-bit value (lo, mid, hi) mod p.
// Strategy: fold hi into (lo, mid) first via the same identity recursively,
// then do one final 128-bit reduction.
inline ulong gold_reduce_192(ulong lo, ulong mid, ulong hi) {
    // (mid, hi) is the "upper 128 bits", contributing 2^64 * (mid + hi*2^64).
    // Reduce (mid, hi) mod p first; call it u. Then result = (lo + u*2^64) mod p.
    // But u*2^64 mod p we can get by reducing 128-bit value (0, u).
    // Simpler: fold hi into a partial sum.
    //
    // We do: result = reduce_128(lo, mid')  where mid' accounts for hi too,
    // but mid' might overflow. Use two-step:
    //   step1: reduce_128(mid, hi) -> m   (m < p < 2^64)
    //   step2: reduce_128(lo, m)   -> wrong! that treats m as the hi-64 part,
    //          which is correct only if upper 128 was (m, 0). Since
    //          (mid + hi*2^64) ≡ m (mod p), we have
    //          (lo + (mid + hi*2^64)*2^64) ≡ lo + m * 2^64 (mod p).
    //   That equals reduce_128(lo, m).
    ulong m = gold_reduce_128(mid, hi);
    return gold_reduce_128(lo, m);
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

// 192-bit accumulator add of a 128-bit (alo, ahi).
inline void acc192_add128(thread ulong &lo, thread ulong &mid, thread ulong &hi,
                          ulong alo, ulong ahi) {
    ulong nl = lo + alo;
    ulong c1 = (nl < lo) ? 1ul : 0ul;
    lo = nl;
    ulong nm = mid + ahi;
    ulong c2 = (nm < mid) ? 1ul : 0ul;
    nm += c1;
    if (nm < c1) c2 += 1ul;
    mid = nm;
    hi += c2;
}

// MAC into 192-bit accumulator: acc += a*b.
inline void acc192_mac(thread ulong &lo, thread ulong &mid, thread ulong &hi,
                       ulong a, ulong b) {
    ulong pl, ph;
    mul_full_128(a, b, pl, ph);
    acc192_add128(lo, mid, hi, pl, ph);
}

inline ulong sbox(ulong x) {
    ulong x2 = gold_mul(x, x);
    ulong x4 = gold_mul(x2, x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

// Dot product of length 3 with single reduction at end.
inline ulong dot3(ulong a0, ulong b0, ulong a1, ulong b1, ulong a2, ulong b2) {
    ulong lo = 0ul, mid = 0ul, hi = 0ul;
    acc192_mac(lo, mid, hi, a0, b0);
    acc192_mac(lo, mid, hi, a1, b1);
    acc192_mac(lo, mid, hi, a2, b2);
    return gold_reduce_192(lo, mid, hi);
}

inline ulong dot4(ulong a0, ulong b0, ulong a1, ulong b1,
                  ulong a2, ulong b2, ulong a3, ulong b3) {
    ulong lo = 0ul, mid = 0ul, hi = 0ul;
    acc192_mac(lo, mid, hi, a0, b0);
    acc192_mac(lo, mid, hi, a1, b1);
    acc192_mac(lo, mid, hi, a2, b2);
    acc192_mac(lo, mid, hi, a3, b3);
    return gold_reduce_192(lo, mid, hi);
}

inline ulong dot2(ulong a0, ulong b0, ulong a1, ulong b1) {
    ulong lo = 0ul, mid = 0ul, hi = 0ul;
    acc192_mac(lo, mid, hi, a0, b0);
    acc192_mac(lo, mid, hi, a1, b1);
    return gold_reduce_192(lo, mid, hi);
}

// y = sum + diag*s, where sum is already reduced (< p) and we want one product + add.
// Do as 192-bit accumulator: lo,mid,hi = diag*s, then add sum.
inline ulong sum_plus_mul(ulong sum, ulong diag_i, ulong s_i) {
    ulong pl, ph;
    mul_full_128(diag_i, s_i, pl, ph);
    ulong nl = pl + sum;
    ulong c  = (nl < pl) ? 1ul : 0ul;
    pl = nl;
    ph = ph + c;
    return gold_reduce_128(pl, ph);
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

    // ---------- Specialized t=3 path ----------
    if (tt == 3u) {
        ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
        ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
        ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];
        ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

        ulong s0 = in_state[idx * 3u + 0u];
        ulong s1 = in_state[idx * 3u + 1u];
        ulong s2 = in_state[idx * 3u + 2u];

        // Pre-multiply by external MDS.
        {
            ulong n0 = dot3(m00, s0, m01, s1, m02, s2);
            ulong n1 = dot3(m10, s0, m11, s1, m12, s2);
            ulong n2 = dot3(m20, s0, m21, s1, m22, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        // First half full rounds.
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong n0 = dot3(m00, s0, m01, s1, m02, s2);
            ulong n1 = dot3(m10, s0, m11, s1, m12, s2);
            ulong n2 = dot3(m20, s0, m21, s1, m22, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Partial rounds.
        for (uint r = 0u; r < rp; ++r) {
            s0 = sbox(gold_add(s0, rc_int[r]));
            ulong sum = gold_add(gold_add(s0, s1), s2);
            ulong n0 = sum_plus_mul(sum, d0, s0);
            ulong n1 = sum_plus_mul(sum, d1, s1);
            ulong n2 = sum_plus_mul(sum, d2, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        // Second half full rounds.
        for (uint r = half_f; r < rf; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 3u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 3u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 3u + 2u]));
            ulong n0 = dot3(m00, s0, m01, s1, m02, s2);
            ulong n1 = dot3(m10, s0, m11, s1, m12, s2);
            ulong n2 = dot3(m20, s0, m21, s1, m22, s2);
            s0 = n0; s1 = n1; s2 = n2;
        }

        out_state[idx * 3u + 0u] = s0;
        out_state[idx * 3u + 1u] = s1;
        out_state[idx * 3u + 2u] = s2;
        return;
    }

    // ---------- Specialized t=2 path ----------
    if (tt == 2u) {
        ulong m00 = ext_mds[0], m01 = ext_mds[1];
        ulong m10 = ext_mds[2], m11 = ext_mds[3];
        ulong d0 = int_diag[0], d1 = int_diag[1];

        ulong s0 = in_state[idx * 2u + 0u];
        ulong s1 = in_state[idx * 2u + 1u];

        {
            ulong n0 = dot2(m00, s0, m01, s1);
            ulong n1 = dot2(m10, s0, m11, s1);
            s0 = n0; s1 = n1;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 2u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 2u + 1u]));
            ulong n0 = dot2(m00, s0, m01, s1);
            ulong n1 = dot2(m10, s0, m11, s1);
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
            ulong n0 = dot2(m00, s0, m01, s1);
            ulong n1 = dot2(m10, s0, m11, s1);
            s0 = n0; s1 = n1;
        }
        out_state[idx * 2u + 0u] = s0;
        out_state[idx * 2u + 1u] = s1;
        return;
    }

    // ---------- Specialized t=4 path ----------
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
            ulong n0 = dot4(m00, s0, m01, s1, m02, s2, m03, s3);
            ulong n1 = dot4(m10, s0, m11, s1, m12, s2, m13, s3);
            ulong n2 = dot4(m20, s0, m21, s1, m22, s2, m23, s3);
            ulong n3 = dot4(m30, s0, m31, s1, m32, s2, m33, s3);
            s0 = n0; s1 = n1; s2 = n2; s3 = n3;
        }
        for (uint r = 0u; r < half_f; ++r) {
            s0 = sbox(gold_add(s0, rc_ext[r * 4u + 0u]));
            s1 = sbox(gold_add(s1, rc_ext[r * 4u + 1u]));
            s2 = sbox(gold_add(s2, rc_ext[r * 4u + 2u]));
            s3 = sbox(gold_add(s3, rc_ext[r * 4u + 3u]));
            ulong n0 = dot4(m00, s0, m01, s1, m02, s2, m03, s3);
            ulong n1 = dot4(m10, s0, m11, s1, m12, s2, m13, s3);
            ulong n2 = dot4(m20, s0, m21, s1, m22, s2, m23, s3);
            ulong n3 = dot4(m30, s0, m31, s1, m32, s2, m33, s3);
            s0 = n0; s1 = n1; s2 = n2; s3 = n3;
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
            ulong n0 = dot4(m00, s0, m01, s1, m02, s2, m03, s3);
            ulong n1 = dot4(m10, s0, m11, s1, m12, s2, m13, s3);
            ulong n2 = dot4(m20, s0, m21, s1, m22, s2, m23, s3);
            ulong n3 = dot4(m30, s0, m31, s1, m32, s2, m33, s3);
            s0 = n0; s1 = n1; s2 = n2; s3 = n3;
        }
        out_state[idx * 4u + 0u] = s0;
        out_state[idx * 4u + 1u] = s1;
        out_state[idx * 4u + 2u] = s2;
        out_state[idx * 4u + 3u] = s3;
        return;
    }

    // ---------- Generic fallback ----------
    ulong state[T_MAX];
    for (uint i = 0u; i < tt; ++i) state[i] = in_state[idx * tt + i];

    {
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong lo = 0ul, mid = 0ul, hi = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc192_mac(lo, mid, hi, ext_mds[i * tt + j], state[j]);
            }
            tmp[i] = gold_reduce_192(lo, mid, hi);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < tt; ++i) state[i] = sbox(gold_add(state[i], rc_ext[r * tt + i]));
        ulong tmp[T_MAX];
        for (uint i = 0u; i < tt; ++i) {
            ulong lo = 0ul, mid = 0ul, hi = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc192_mac(lo, mid, hi, ext_mds[i * tt + j], state[j]);
            }
            tmp[i] = gold_reduce_192(lo, mid, hi);
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
            ulong lo = 0ul, mid = 0ul, hi = 0ul;
            for (uint j = 0u; j < tt; ++j) {
                acc192_mac(lo, mid, hi, ext_mds[i * tt + j], state[j]);
            }
            tmp[i] = gold_reduce_192(lo, mid, hi);
        }
        for (uint i = 0u; i < tt; ++i) state[i] = tmp[i];
    }
    for (uint i = 0u; i < tt; ++i) out_state[idx * tt + i] = state[i];
}