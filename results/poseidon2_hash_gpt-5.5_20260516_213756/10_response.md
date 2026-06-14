This version avoids the previous code-size blowup from fully unrolling the hot Poseidon rounds. It instead uses weak/non-canonical representatives only inside S-box multiplication chains and before S-box additions, canonicalizing at the S-box output. This removes several redundant Goldilocks final-subtract steps per S-box while preserving exact canonical outputs at round boundaries. It also keeps a compact pre-split diagonal path for the t=3 partial rounds to reduce repeated invariant work without harming instruction-cache behavior.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

// Weak addition: returns a 64-bit representative congruent mod p, not necessarily < p.
// Use only when the consumer accepts weak representatives.
inline ulong gold_add_weak(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    return s;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong s = a - b;
    if (a < b) s -= EPSILON;
    return s;
}

inline ulong gold_fold_sum_carry_weak(ulong s, uint carry) {
    ulong corr = (((ulong)carry) << 32) - (ulong)carry;
    ulong old = s;
    s += corr;
    if (s < old) s += EPSILON;
    return s;
}

inline ulong gold_fold_sum_carry(ulong s, uint carry) {
    return gold_canonical(gold_fold_sum_carry_weak(s, carry));
}

inline ulong gold_add3(ulong a, ulong b, ulong c) {
    ulong s = a + b;
    uint carry = (s < a) ? 1u : 0u;
    ulong old = s;
    s += c;
    carry += (s < old) ? 1u : 0u;
    return gold_fold_sum_carry(s, carry);
}

inline ulong gold_add3_weak(ulong a, ulong b, ulong c) {
    ulong s = a + b;
    uint carry = (s < a) ? 1u : 0u;
    ulong old = s;
    s += c;
    carry += (s < old) ? 1u : 0u;
    return gold_fold_sum_carry_weak(s, carry);
}

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
}

inline ulong gold_reduce_limbs_weak(uint lo0, uint lo1, uint hi0, uint hi1) {
    ulong x_lo = ((ulong)lo1 << 32) | (ulong)lo0;

    ulong t0 = x_lo - (ulong)hi1;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = ((ulong)hi0 << 32) - (ulong)hi0;

    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;

    return t2;
}

inline ulong gold_reduce_limbs(uint lo0, uint lo1, uint hi0, uint hi1) {
    return gold_canonical(gold_reduce_limbs_weak(lo0, lo1, hi0, hi1));
}

inline ulong gold_mul_weak(ulong a, ulong b) {
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

    uint mid = p00hi + p01lo;
    uint carry = (mid < p00hi) ? 1u : 0u;
    uint lo1 = mid + p10lo;
    carry += (lo1 < mid) ? 1u : 0u;

    uint hi0 = p11lo;
    uint hi1 = p11hi;

    uint old = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += p10hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs_weak(p00lo, lo1, hi0, hi1);
}

inline ulong gold_mul(ulong a, ulong b) {
    return gold_canonical(gold_mul_weak(a, b));
}

inline ulong gold_mul_add_limb(uint a0, uint a1, ulong b, ulong addend) {
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

    uint mid = p00hi + p01lo;
    uint carry = (mid < p00hi) ? 1u : 0u;
    uint lo1 = mid + p10lo;
    carry += (lo1 < mid) ? 1u : 0u;

    uint hi0 = p11lo;
    uint hi1 = p11hi;

    uint old32 = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old32) ? 1u : 0u;

    old32 = hi0;
    hi0 += p10hi;
    hi1 += (hi0 < old32) ? 1u : 0u;

    old32 = hi0;
    hi0 += carry;
    hi1 += (hi0 < old32) ? 1u : 0u;

    ulong lo = ((ulong)lo1 << 32) | (ulong)p00lo;
    ulong hi = ((ulong)hi1 << 32) | (ulong)hi0;

    ulong old = lo;
    lo += addend;
    hi += (lo < old) ? 1ul : 0ul;

    return gold_reduce_limbs((uint)lo, (uint)(lo >> 32),
                             (uint)hi, (uint)(hi >> 32));
}

inline ulong gold_mul_add(ulong a, ulong b, ulong addend) {
    return gold_mul_add_limb((uint)a, (uint)(a >> 32), b, addend);
}

inline ulong gold_square_weak(ulong a) {
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

    uint hi0 = p11lo;
    uint hi1 = p11hi;

    uint old = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += p01hi;
    hi1 += (hi0 < old) ? 1u : 0u;

    old = hi0;
    hi0 += carry;
    hi1 += (hi0 < old) ? 1u : 0u;

    return gold_reduce_limbs_weak(p00lo, lo1, hi0, hi1);
}

inline ulong gold_square(ulong a) {
    return gold_canonical(gold_square_weak(a));
}

// Accepts either canonical or weak 64-bit representatives; returns canonical.
inline ulong sbox7(ulong x) {
    ulong x2 = gold_square_weak(x);
    ulong x4 = gold_square_weak(x2);
    ulong x6 = gold_mul_weak(x4, x2);
    return gold_mul(x6, x);
}

inline bool gold_small_const(ulong c) {
    return (c <= 4ul) || (c == (P_GOLD - 1ul));
}

inline ulong gold_mul_small_add(ulong c, ulong x, ulong addend) {
    if (c == 0ul) return addend;
    if (c == 1ul) return gold_add(addend, x);
    if (c == 2ul) {
        ulong x2 = gold_add(x, x);
        return gold_add(addend, x2);
    }
    if (c == 3ul) {
        ulong x2 = gold_add(x, x);
        ulong x3 = gold_add(x2, x);
        return gold_add(addend, x3);
    }
    if (c == 4ul) {
        ulong x2 = gold_add(x, x);
        ulong x4 = gold_add(x2, x2);
        return gold_add(addend, x4);
    }
    if (c == (P_GOLD - 1ul)) {
        return gold_sub(addend, x);
    }
    return gold_mul_add(c, x, addend);
}

inline void matvec_ext_generic(thread ulong *state,
                               device const ulong *ext_mds,
                               uint t)
{
    ulong tmp[T_MAX];
    for (uint i = 0u; i < t; ++i) {
        ulong acc = gold_mul_weak(ext_mds[i * t], state[0]);
        for (uint j = 1u; j < t; ++j) {
            acc = gold_mul_add(ext_mds[i * t + j], state[j], acc);
        }
        tmp[i] = gold_canonical(acc);
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
        tmp[i] = gold_mul_add(int_diag[i], state[i], s);
    }
    for (uint i = 0u; i < t; ++i) {
        state[i] = tmp[i];
    }
}

#define P2_STORE3() do {                                                        \
    out_state[base + 0u] = s0;                                                  \
    out_state[base + 1u] = s1;                                                  \
    out_state[base + 2u] = s2;                                                  \
} while (0)

#define P2_EXT3_JI_STEP() do {                                                  \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3(a0_, a1_, a2_);                                      \
    s0 = gold_add(sum_, a0_);                                                   \
    s1 = gold_add(sum_, a1_);                                                   \
    s2 = gold_add(sum_, a2_);                                                   \
} while (0)

#define P2_EXT3_JDIAG_STEP() do {                                               \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3_weak(a0_, a1_, a2_);                                 \
    s0 = gold_mul_add(e0, a0_, sum_);                                           \
    s1 = gold_mul_add(e1, a1_, sum_);                                           \
    s2 = gold_mul_add(e2, a2_, sum_);                                           \
} while (0)

#define P2_EXT3_DENSE_STEP() do {                                               \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong e0_ = gold_mul_weak(m00, a0_);                                        \
    e0_ = gold_mul_add(m01, a1_, e0_);                                          \
    e0_ = gold_mul_add(m02, a2_, e0_);                                          \
    ulong e1_ = gold_mul_weak(m10, a0_);                                        \
    e1_ = gold_mul_add(m11, a1_, e1_);                                          \
    e1_ = gold_mul_add(m12, a2_, e1_);                                          \
    ulong e2_ = gold_mul_weak(m20, a0_);                                        \
    e2_ = gold_mul_add(m21, a1_, e2_);                                          \
    e2_ = gold_mul_add(m22, a2_, e2_);                                          \
    s0 = e0_;                                                                   \
    s1 = e1_;                                                                   \
    s2 = e2_;                                                                   \
} while (0)

#define P2_PART3_DIAG1_PTR() do {                                               \
    s0 = sbox7(gold_add_weak(s0, rip[0]));                                      \
    P2_EXT3_JI_STEP();                                                          \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_SMALL_PTR() do {                                               \
    s0 = sbox7(gold_add_weak(s0, rip[0]));                                      \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_small_add(d0, s0, sum_);                               \
    ulong p1_ = gold_mul_small_add(d1, s1, sum_);                               \
    ulong p2_ = gold_mul_small_add(d2, s2, sum_);                               \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_LIMB_PTR() do {                                                \
    s0 = sbox7(gold_add_weak(s0, rip[0]));                                      \
    ulong sum_ = gold_add3_weak(s0, s1, s2);                                    \
    ulong p0_ = gold_mul_add_limb(d0l, d0h, s0, sum_);                          \
    ulong p1_ = gold_mul_add_limb(d1l, d1h, s1, sum_);                          \
    ulong p2_ = gold_mul_add_limb(d2l, d2h, s2, sum_);                          \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_FULL3_JI_PTR() do {                                                  \
    s0 = sbox7(gold_add_weak(s0, rcp[0]));                                      \
    s1 = sbox7(gold_add_weak(s1, rcp[1]));                                      \
    s2 = sbox7(gold_add_weak(s2, rcp[2]));                                      \
    P2_EXT3_JI_STEP();                                                          \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_JDIAG_PTR() do {                                               \
    s0 = sbox7(gold_add_weak(s0, rcp[0]));                                      \
    s1 = sbox7(gold_add_weak(s1, rcp[1]));                                      \
    s2 = sbox7(gold_add_weak(s2, rcp[2]));                                      \
    P2_EXT3_JDIAG_STEP();                                                       \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_DENSE_PTR() do {                                               \
    s0 = sbox7(gold_add_weak(s0, rcp[0]));                                      \
    s1 = sbox7(gold_add_weak(s1, rcp[1]));                                      \
    s2 = sbox7(gold_add_weak(s2, rcp[2]));                                      \
    P2_EXT3_DENSE_STEP();                                                       \
    rcp += 3;                                                                   \
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

        bool offdiag_one = (m01 == 1ul) & (m02 == 1ul) &
                           (m10 == 1ul) & (m12 == 1ul) &
                           (m20 == 1ul) & (m21 == 1ul);

        bool d_all_one = (d0 == 1ul) & (d1 == 1ul) & (d2 == 1ul);
        bool d_small = gold_small_const(d0) & gold_small_const(d1) & gold_small_const(d2);

        uint half_f = r_f >> 1u;

        if (offdiag_one && (m00 == 2ul) && (m11 == 2ul) && (m22 == 2ul)) {
            P2_EXT3_JI_STEP();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JI_PTR();
            }

            device const ulong *rip = rc_int;
            if (d_all_one) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_DIAG1_PTR();
                }
            } else if (d_small) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_SMALL_PTR();
                }
            } else {
                uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
                uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
                uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_LIMB_PTR();
                }
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JI_PTR();
            }

            P2_STORE3();
            return;
        }

        if (offdiag_one) {
            ulong e0 = gold_sub_one(m00);
            ulong e1 = gold_sub_one(m11);
            ulong e2 = gold_sub_one(m22);

            P2_EXT3_JDIAG_STEP();

            device const ulong *rcp = rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JDIAG_PTR();
            }

            device const ulong *rip = rc_int;
            if (d_all_one) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_DIAG1_PTR();
                }
            } else if (d_small) {
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_SMALL_PTR();
                }
            } else {
                uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
                uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
                uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);
                for (uint r = 0u; r < r_p; ++r) {
                    P2_PART3_LIMB_PTR();
                }
            }

            rcp = rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JDIAG_PTR();
            }

            P2_STORE3();
            return;
        }

        P2_EXT3_DENSE_STEP();

        device const ulong *rcp = rc_ext;
        for (uint r = 0u; r < half_f; ++r) {
            P2_FULL3_DENSE_PTR();
        }

        device const ulong *rip = rc_int;
        if (d_all_one) {
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_DIAG1_PTR();
            }
        } else if (d_small) {
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_SMALL_PTR();
            }
        } else {
            uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
            uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
            uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_LIMB_PTR();
            }
        }

        rcp = rc_ext + half_f * 3u;
        for (uint r = half_f; r < r_f; ++r) {
            P2_FULL3_DENSE_PTR();
        }

        P2_STORE3();
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
            state[i] = sbox7(gold_add_weak(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gold_add_weak(state[0], rc_int[r]));
        matvec_int_generic(state, int_diag, t);
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint off = r * t;
        for (uint i = 0u; i < t; ++i) {
            state[i] = sbox7(gold_add_weak(state[i], rc_ext[off + i]));
        }
        matvec_ext_generic(state, ext_mds, t);
    }

    for (uint i = 0u; i < t; ++i) {
        out_state[base + i] = state[i];
    }
}

#undef P2_STORE3
#undef P2_EXT3_JI_STEP
#undef P2_EXT3_JDIAG_STEP
#undef P2_EXT3_DENSE_STEP
#undef P2_PART3_DIAG1_PTR
#undef P2_PART3_SMALL_PTR
#undef P2_PART3_LIMB_PTR
#undef P2_FULL3_JI_PTR
#undef P2_FULL3_JDIAG_PTR
#undef P2_FULL3_DENSE_PTR
```