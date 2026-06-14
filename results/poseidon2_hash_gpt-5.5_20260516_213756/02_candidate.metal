#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD     = 0xFFFFFFFF00000001ul;
constant ulong P_MINUS_1  = 0xFFFFFFFF00000000ul;
constant ulong EPSILON    = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    ulong y = x - P_GOLD;
    return (x >= P_GOLD) ? y : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong s = a + b;
    s += (s < a) ? EPSILON : 0ul;
    return gold_canonical(s);
}

inline ulong gold_add3(ulong a, ulong b, ulong c) {
    ulong s = a + b;
    uint carry = (s < a) ? 1u : 0u;

    ulong old = s;
    s += c;
    carry += (s < old) ? 1u : 0u;

    old = s;
    s += ((ulong)carry) * EPSILON;
    s += (s < old) ? EPSILON : 0ul;

    return gold_canonical(s);
}

inline ulong gold_neg(ulong a) {
    return (a == 0ul) ? 0ul : (P_GOLD - a);
}

inline ulong gold_reduce_limbs(uint lo0, uint lo1, uint hi0, uint hi1) {
    ulong x_lo = ((ulong)lo1 << 32) | (ulong)lo0;

    ulong t0 = x_lo - (ulong)hi1;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = ((ulong)hi0 << 32) - (ulong)hi0;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00lo = a0 * b0;
    uint p00hi = mulhi(a0, b0);

    uint p01lo = a0 * b1;
    uint p01hi = mulhi(a0, b1);

    uint p10lo = a1 * b0;
    uint p10hi = mulhi(a1, b0);

    uint p11lo = a1 * b1;
    uint p11hi = mulhi(a1, b1);

    uint low_sum = p01lo + p10lo;
    uint carry = (low_sum < p01lo) ? 1u : 0u;

    uint lo1 = p00hi + low_sum;
    carry += (lo1 < p00hi) ? 1u : 0u;

    uint high_sum = p01hi + p10hi;
    uint high_carry = (high_sum < p01hi) ? 1u : 0u;

    uint hi0 = p11lo + high_sum;
    uint hi1 = p11hi + high_carry + ((hi0 < p11lo) ? 1u : 0u);

    uint old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
}

inline ulong gold_mul_u32(ulong a, uint b0) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00lo = a0 * b0;
    uint p00hi = mulhi(a0, b0);

    uint p10lo = a1 * b0;
    uint p10hi = mulhi(a1, b0);

    uint lo1 = p00hi + p10lo;
    uint carry = (lo1 < p00hi) ? 1u : 0u;

    uint hi0 = p10hi + carry;

    return gold_reduce_limbs(p00lo, lo1, hi0, 0u);
}

inline ulong gold_mul_const(ulong c, ulong a) {
    uint ch = (uint)(c >> 32);
    if (ch == 0u) {
        uint cl = (uint)c;
        switch (cl) {
            case 0u: return 0ul;
            case 1u: return a;
            case 2u: return gold_add(a, a);
            case 3u: return gold_add3(a, a, a);
            case 4u: {
                ulong d = gold_add(a, a);
                return gold_add(d, d);
            }
            default:
                return gold_mul_u32(a, cl);
        }
    }

    if (c == P_MINUS_1) {
        return gold_neg(a);
    }

    return gold_mul(c, a);
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00lo = a0 * a0;
    uint p00hi = mulhi(a0, a0);

    uint p01lo = a0 * a1;
    uint p01hi = mulhi(a0, a1);

    uint p11lo = a1 * a1;
    uint p11hi = mulhi(a1, a1);

    uint dbl_lo = p01lo << 1;
    uint carry = p01lo >> 31;

    uint lo1 = p00hi + dbl_lo;
    carry += (lo1 < p00hi) ? 1u : 0u;

    uint dbl_hi = p01hi << 1;
    uint high_carry = p01hi >> 31;

    uint hi0 = p11lo + dbl_hi;
    uint hi1 = p11hi + high_carry + ((hi0 < p11lo) ? 1u : 0u);

    uint old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void matvec_ext_generic(thread ulong *state,
                               device const ulong *ext_mds,
                               uint t)
{
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        ulong acc = 0ul;
        for (uint j = 0u; j < t; ++j) {
            acc = gold_add(acc, gold_mul_const(ext_mds[i * t + j], state[j]));
        }
        tmp[i] = acc;
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

inline void matvec_int_generic(thread ulong *state,
                               device const ulong *int_diag,
                               uint t)
{
    ulong s = 0ul;
    for (uint i = 0u; i < t; ++i) {
        s = gold_add(s, state[i]);
    }

    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        tmp[i] = gold_add(s, gold_mul_const(int_diag[i], state[i]));
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

#define P2_EXT3_STEP() do {                                                            \
    ulong e0_ = gold_add3(gold_mul_const(m00, s0),                                    \
                          gold_mul_const(m01, s1),                                    \
                          gold_mul_const(m02, s2));                                   \
    ulong e1_ = gold_add3(gold_mul_const(m10, s0),                                    \
                          gold_mul_const(m11, s1),                                    \
                          gold_mul_const(m12, s2));                                   \
    ulong e2_ = gold_add3(gold_mul_const(m20, s0),                                    \
                          gold_mul_const(m21, s1),                                    \
                          gold_mul_const(m22, s2));                                   \
    s0 = e0_;                                                                          \
    s1 = e1_;                                                                          \
    s2 = e2_;                                                                          \
} while (0)

#define P2_FULL3_AT(R_) do {                                                           \
    s0 = sbox7(gold_add(s0, rc_ext[(R_) * 3u + 0u]));                                  \
    s1 = sbox7(gold_add(s1, rc_ext[(R_) * 3u + 1u]));                                  \
    s2 = sbox7(gold_add(s2, rc_ext[(R_) * 3u + 2u]));                                  \
    P2_EXT3_STEP();                                                                    \
} while (0)

#define P2_PART3_AT(R_) do {                                                           \
    s0 = sbox7(gold_add(s0, rc_int[(R_)]));                                            \
    ulong sum_ = gold_add3(s0, s1, s2);                                                \
    ulong p0_ = gold_add(sum_, gold_mul_const(d0, s0));                                \
    ulong p1_ = gold_add(sum_, gold_mul_const(d1, s1));                                \
    ulong p2_ = gold_add(sum_, gold_mul_const(d2, s2));                                \
    s0 = p0_;                                                                          \
    s1 = p1_;                                                                          \
    s2 = p2_;                                                                          \
} while (0)

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

    if (t == 3u) {
        uint base = idx * 3u;

        ulong s0 = in_state[base + 0u];
        ulong s1 = in_state[base + 1u];
        ulong s2 = in_state[base + 2u];

        ulong m00 = ext_mds[0u];
        ulong m01 = ext_mds[1u];
        ulong m02 = ext_mds[2u];
        ulong m10 = ext_mds[3u];
        ulong m11 = ext_mds[4u];
        ulong m12 = ext_mds[5u];
        ulong m20 = ext_mds[6u];
        ulong m21 = ext_mds[7u];
        ulong m22 = ext_mds[8u];

        ulong d0 = int_diag[0u];
        ulong d1 = int_diag[1u];
        ulong d2 = int_diag[2u];

        P2_EXT3_STEP();

        if (r_f == 8u && r_p == 22u) {
            P2_FULL3_AT(0u);
            P2_FULL3_AT(1u);
            P2_FULL3_AT(2u);
            P2_FULL3_AT(3u);

            P2_PART3_AT(0u);
            P2_PART3_AT(1u);
            P2_PART3_AT(2u);
            P2_PART3_AT(3u);
            P2_PART3_AT(4u);
            P2_PART3_AT(5u);
            P2_PART3_AT(6u);
            P2_PART3_AT(7u);
            P2_PART3_AT(8u);
            P2_PART3_AT(9u);
            P2_PART3_AT(10u);
            P2_PART3_AT(11u);
            P2_PART3_AT(12u);
            P2_PART3_AT(13u);
            P2_PART3_AT(14u);
            P2_PART3_AT(15u);
            P2_PART3_AT(16u);
            P2_PART3_AT(17u);
            P2_PART3_AT(18u);
            P2_PART3_AT(19u);
            P2_PART3_AT(20u);
            P2_PART3_AT(21u);

            P2_FULL3_AT(4u);
            P2_FULL3_AT(5u);
            P2_FULL3_AT(6u);
            P2_FULL3_AT(7u);
        } else {
            uint half_f = r_f >> 1u;

            for (uint r = 0u; r < half_f; ++r) {
                uint off = r * 3u;
                s0 = sbox7(gold_add(s0, rc_ext[off + 0u]));
                s1 = sbox7(gold_add(s1, rc_ext[off + 1u]));
                s2 = sbox7(gold_add(s2, rc_ext[off + 2u]));
                P2_EXT3_STEP();
            }

            for (uint r = 0u; r < r_p; ++r) {
                s0 = sbox7(gold_add(s0, rc_int[r]));
                ulong sum_ = gold_add3(s0, s1, s2);
                ulong p0_ = gold_add(sum_, gold_mul_const(d0, s0));
                ulong p1_ = gold_add(sum_, gold_mul_const(d1, s1));
                ulong p2_ = gold_add(sum_, gold_mul_const(d2, s2));
                s0 = p0_;
                s1 = p1_;
                s2 = p2_;
            }

            for (uint r = half_f; r < r_f; ++r) {
                uint off = r * 3u;
                s0 = sbox7(gold_add(s0, rc_ext[off + 0u]));
                s1 = sbox7(gold_add(s1, rc_ext[off + 1u]));
                s2 = sbox7(gold_add(s2, rc_ext[off + 2u]));
                P2_EXT3_STEP();
            }
        }

        out_state[base + 0u] = s0;
        out_state[base + 1u] = s1;
        out_state[base + 2u] = s2;
        return;
    }

    thread ulong state[T_MAX];
    uint base = idx * t;

    for (uint i = 0u; i < t; ++i) {
        state[i] = in_state[base + i];
    }

    matvec_ext_generic(state, ext_mds, t);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gold_add(state[0], rc_int[r]));
        matvec_int_generic(state, int_diag, t);
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint i = 0u; i < t; ++i) {
        out_state[base + i] = state[i];
    }
}

#undef P2_EXT3_STEP
#undef P2_FULL3_AT
#undef P2_PART3_AT