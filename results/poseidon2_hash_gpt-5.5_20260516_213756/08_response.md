I added a t=3 fast path that cooperatively caches the small, uniform parameter tables in threadgroup memory and fully unrolls the common Poseidon2 Goldilocks configuration (R_F=8, R_P=22, external MDS=J+I). The partial-round diagonal multiplications use pre-split 32-bit limbs to avoid repeated constant extraction. This should reduce uniform global-load traffic, loop overhead, and instruction count in the benchmarked path while preserving the generic fallback for all specified runtime parameters.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

constexpr constant uint T_MAX = 4u;

inline ulong gold_canonical(ulong x) {
    return x - ((x >= P_GOLD) ? P_GOLD : 0ul);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong s = a + b;
    s += (s < a) ? EPSILON : 0ul;
    s -= (s >= P_GOLD) ? P_GOLD : 0ul;
    return s;
}

inline ulong gold_fold_sum_carry(ulong s, uint carry) {
    ulong corr = (((ulong)carry) << 32) - (ulong)carry;
    ulong old = s;
    s += corr;
    s += (s < old) ? EPSILON : 0ul;
    return gold_canonical(s);
}

inline ulong gold_add3(ulong a, ulong b, ulong c) {
    ulong s = a + b;
    uint carry = (s < a) ? 1u : 0u;
    ulong old = s;
    s += c;
    carry += (s < old) ? 1u : 0u;
    return gold_fold_sum_carry(s, carry);
}

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
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

inline ulong gold_mul_limb(uint a0, uint a1, ulong b) {
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

    return gold_reduce_limbs(p00lo, lo1, hi0, hi1);
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

inline ulong gold_mul(ulong a, ulong b) {
    return gold_mul_limb((uint)a, (uint)(a >> 32), b);
}

inline ulong gold_mul_add(ulong a, ulong b, ulong addend) {
    return gold_mul_add_limb((uint)a, (uint)(a >> 32), b, addend);
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
        ulong acc = gold_mul(ext_mds[i * t], state[0]);
        for (uint j = 1u; j < t; ++j) {
            acc = gold_mul_add(ext_mds[i * t + j], state[j], acc);
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

#define P2_EXT3_JDIAG_STEP_LIMB() do {                                          \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong sum_ = gold_add3(a0_, a1_, a2_);                                      \
    s0 = gold_mul_add_limb(e0l, e0h, a0_, sum_);                                \
    s1 = gold_mul_add_limb(e1l, e1h, a1_, sum_);                                \
    s2 = gold_mul_add_limb(e2l, e2h, a2_, sum_);                                \
} while (0)

#define P2_EXT3_DENSE_STEP_LIMB() do {                                          \
    ulong a0_ = s0;                                                             \
    ulong a1_ = s1;                                                             \
    ulong a2_ = s2;                                                             \
    ulong y0_ = gold_mul_limb(m00l, m00h, a0_);                                 \
    y0_ = gold_mul_add_limb(m01l, m01h, a1_, y0_);                              \
    y0_ = gold_mul_add_limb(m02l, m02h, a2_, y0_);                              \
    ulong y1_ = gold_mul_limb(m10l, m10h, a0_);                                 \
    y1_ = gold_mul_add_limb(m11l, m11h, a1_, y1_);                              \
    y1_ = gold_mul_add_limb(m12l, m12h, a2_, y1_);                              \
    ulong y2_ = gold_mul_limb(m20l, m20h, a0_);                                 \
    y2_ = gold_mul_add_limb(m21l, m21h, a1_, y2_);                              \
    y2_ = gold_mul_add_limb(m22l, m22h, a2_, y2_);                              \
    s0 = y0_;                                                                   \
    s1 = y1_;                                                                   \
    s2 = y2_;                                                                   \
} while (0)

#define P2_PART3_PTR_TG() do {                                                  \
    s0 = sbox7(gold_add(s0, rip[0]));                                           \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_add_limb(d0l, d0h, s0, sum_);                          \
    ulong p1_ = gold_mul_add_limb(d1l, d1h, s1, sum_);                          \
    ulong p2_ = gold_mul_add_limb(d2l, d2h, s2, sum_);                          \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
    rip += 1;                                                                   \
} while (0)

#define P2_PART3_K(K) do {                                                      \
    s0 = sbox7(gold_add(s0, tg_rc_int[(K)]));                                   \
    ulong sum_ = gold_add3(s0, s1, s2);                                         \
    ulong p0_ = gold_mul_add_limb(d0l, d0h, s0, sum_);                          \
    ulong p1_ = gold_mul_add_limb(d1l, d1h, s1, sum_);                          \
    ulong p2_ = gold_mul_add_limb(d2l, d2h, s2, sum_);                          \
    s0 = p0_;                                                                   \
    s1 = p1_;                                                                   \
    s2 = p2_;                                                                   \
} while (0)

#define P2_FULL3_JI_PTR_TG() do {                                               \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_JI_STEP();                                                          \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_JDIAG_PTR_TG() do {                                            \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_JDIAG_STEP_LIMB();                                                  \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_DENSE_PTR_TG() do {                                            \
    s0 = sbox7(gold_add(s0, rcp[0]));                                           \
    s1 = sbox7(gold_add(s1, rcp[1]));                                           \
    s2 = sbox7(gold_add(s2, rcp[2]));                                           \
    P2_EXT3_DENSE_STEP_LIMB();                                                  \
    rcp += 3;                                                                   \
} while (0)

#define P2_FULL3_JI_K(K) do {                                                   \
    s0 = sbox7(gold_add(s0, tg_rc_ext[(K) * 3u + 0u]));                         \
    s1 = sbox7(gold_add(s1, tg_rc_ext[(K) * 3u + 1u]));                         \
    s2 = sbox7(gold_add(s2, tg_rc_ext[(K) * 3u + 2u]));                         \
    P2_EXT3_JI_STEP();                                                          \
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
    threadgroup ulong tg_rc_ext[32];
    threadgroup ulong tg_rc_int[32];
    threadgroup ulong tg_ext_mds[9];
    threadgroup ulong tg_int_diag[3];

    if (t == 3u) {
        uint tgw = (batch >= 64u) ? 64u : ((batch == 0u) ? 1u : batch);
        uint lid = (batch >= 64u) ? (idx & 63u) : idx;

        for (uint k = lid; k < r_f * 3u; k += tgw) {
            tg_rc_ext[k] = rc_ext[k];
        }
        for (uint k = lid; k < r_p; k += tgw) {
            tg_rc_int[k] = rc_int[k];
        }
        for (uint k = lid; k < 9u; k += tgw) {
            tg_ext_mds[k] = ext_mds[k];
        }
        for (uint k = lid; k < 3u; k += tgw) {
            tg_int_diag[k] = int_diag[k];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx >= batch) return;

        uint base = idx * 3u;

        ulong s0 = in_state[base + 0u];
        ulong s1 = in_state[base + 1u];
        ulong s2 = in_state[base + 2u];

        ulong m00 = tg_ext_mds[0u];
        ulong m01 = tg_ext_mds[1u];
        ulong m02 = tg_ext_mds[2u];
        ulong m10 = tg_ext_mds[3u];
        ulong m11 = tg_ext_mds[4u];
        ulong m12 = tg_ext_mds[5u];
        ulong m20 = tg_ext_mds[6u];
        ulong m21 = tg_ext_mds[7u];
        ulong m22 = tg_ext_mds[8u];

        ulong d0 = tg_int_diag[0u];
        ulong d1 = tg_int_diag[1u];
        ulong d2 = tg_int_diag[2u];

        uint d0l = (uint)d0, d0h = (uint)(d0 >> 32);
        uint d1l = (uint)d1, d1h = (uint)(d1 >> 32);
        uint d2l = (uint)d2, d2h = (uint)(d2 >> 32);

        bool offdiag_one = (m01 == 1ul) & (m02 == 1ul) &
                           (m10 == 1ul) & (m12 == 1ul) &
                           (m20 == 1ul) & (m21 == 1ul);

        uint half_f = r_f >> 1u;

        if (offdiag_one && (m00 == 2ul) && (m11 == 2ul) && (m22 == 2ul)) {
            if ((r_f == 8u) & (r_p == 22u)) {
                P2_EXT3_JI_STEP();

                P2_FULL3_JI_K(0u);
                P2_FULL3_JI_K(1u);
                P2_FULL3_JI_K(2u);
                P2_FULL3_JI_K(3u);

                P2_PART3_K(0u);
                P2_PART3_K(1u);
                P2_PART3_K(2u);
                P2_PART3_K(3u);
                P2_PART3_K(4u);
                P2_PART3_K(5u);
                P2_PART3_K(6u);
                P2_PART3_K(7u);
                P2_PART3_K(8u);
                P2_PART3_K(9u);
                P2_PART3_K(10u);
                P2_PART3_K(11u);
                P2_PART3_K(12u);
                P2_PART3_K(13u);
                P2_PART3_K(14u);
                P2_PART3_K(15u);
                P2_PART3_K(16u);
                P2_PART3_K(17u);
                P2_PART3_K(18u);
                P2_PART3_K(19u);
                P2_PART3_K(20u);
                P2_PART3_K(21u);

                P2_FULL3_JI_K(4u);
                P2_FULL3_JI_K(5u);
                P2_FULL3_JI_K(6u);
                P2_FULL3_JI_K(7u);

                P2_STORE3();
                return;
            }

            P2_EXT3_JI_STEP();

            threadgroup const ulong *rcp = tg_rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JI_PTR_TG();
            }

            threadgroup const ulong *rip = tg_rc_int;
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_PTR_TG();
            }

            rcp = tg_rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JI_PTR_TG();
            }

            P2_STORE3();
            return;
        }

        if (offdiag_one) {
            ulong e0 = gold_sub_one(m00);
            ulong e1 = gold_sub_one(m11);
            ulong e2 = gold_sub_one(m22);
            uint e0l = (uint)e0, e0h = (uint)(e0 >> 32);
            uint e1l = (uint)e1, e1h = (uint)(e1 >> 32);
            uint e2l = (uint)e2, e2h = (uint)(e2 >> 32);

            P2_EXT3_JDIAG_STEP_LIMB();

            threadgroup const ulong *rcp = tg_rc_ext;
            for (uint r = 0u; r < half_f; ++r) {
                P2_FULL3_JDIAG_PTR_TG();
            }

            threadgroup const ulong *rip = tg_rc_int;
            for (uint r = 0u; r < r_p; ++r) {
                P2_PART3_PTR_TG();
            }

            rcp = tg_rc_ext + half_f * 3u;
            for (uint r = half_f; r < r_f; ++r) {
                P2_FULL3_JDIAG_PTR_TG();
            }

            P2_STORE3();
            return;
        }

        uint m00l = (uint)m00, m00h = (uint)(m00 >> 32);
        uint m01l = (uint)m01, m01h = (uint)(m01 >> 32);
        uint m02l = (uint)m02, m02h = (uint)(m02 >> 32);
        uint m10l = (uint)m10, m10h = (uint)(m10 >> 32);
        uint m11l = (uint)m11, m11h = (uint)(m11 >> 32);
        uint m12l = (uint)m12, m12h = (uint)(m12 >> 32);
        uint m20l = (uint)m20, m20h = (uint)(m20 >> 32);
        uint m21l = (uint)m21, m21h = (uint)(m21 >> 32);
        uint m22l = (uint)m22, m22h = (uint)(m22 >> 32);

        P2_EXT3_DENSE_STEP_LIMB();

        threadgroup const ulong *rcp = tg_rc_ext;
        for (uint r = 0u; r < half_f; ++r) {
            P2_FULL3_DENSE_PTR_TG();
        }

        threadgroup const ulong *rip = tg_rc_int;
        for (uint r = 0u; r < r_p; ++r) {
            P2_PART3_PTR_TG();
        }

        rcp = tg_rc_ext + half_f * 3u;
        for (uint r = half_f; r < r_f; ++r) {
            P2_FULL3_DENSE_PTR_TG();
        }

        P2_STORE3();
        return;
    }

    if (idx >= batch) return;

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

#undef P2_STORE3
#undef P2_EXT3_JI_STEP
#undef P2_EXT3_JDIAG_STEP_LIMB
#undef P2_EXT3_DENSE_STEP_LIMB
#undef P2_PART3_PTR_TG
#undef P2_PART3_K
#undef P2_FULL3_JI_PTR_TG
#undef P2_FULL3_JDIAG_PTR_TG
#undef P2_FULL3_DENSE_PTR_TG
#undef P2_FULL3_JI_K
```