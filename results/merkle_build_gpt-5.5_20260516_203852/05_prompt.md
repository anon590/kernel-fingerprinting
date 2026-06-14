## Task: merkle_build

Level-by-level Merkle tree build over the Goldilocks field (p = 2^64 - 2^32 + 1) using Poseidon2 as the inner compression function. The tree has ``n_leaves`` input leaves at level 0; level k+1 has ``ceil(level_k / arity)`` nodes, computed by hashing groups of ``arity`` consecutive children. The build terminates at the 1-element root.

Compression convention (1 Goldilocks element per digest):
  state = [c0, c1, ..., c_{arity-1}, 0, ..., 0]   (zero-pad to width t)
  state = Poseidon2_t(state)
  parent_digest = state[0]

The Poseidon2 permutation parameters (alpha=7 S-box, ``r_f`` full rounds split half+half, ``r_p`` partial rounds, external MDS, internal-MDS diagonal ``int_diag`` with M_I = J + diag(int_diag) where J is the all-ones matrix) are all read at runtime from the bound device buffers, mirroring the layout of the Z3 ``poseidon2_hash`` task. The same kernel must therefore work at t=3 / arity=2 (in-distribution) and t=4 / arity=4 (held-out) without changes -- in particular, the kernel must use the runtime arity, the runtime t, and the runtime round-count parameters, not compile-time constants.

Tree layout: a single contiguous ``ulong`` buffer holds **all levels** concatenated -- leaves first, then each parent level, finally the 1-element root. The total length is the sum of all level node counts. The host issues one kernel dispatch per parent level, binding the per-level scalars (``in_offset``, ``out_offset``, ``child_count``); each dispatch reads from ``tree[in_offset .. in_offset + child_count)`` and writes to ``tree[out_offset .. out_offset + parent_count)`` with ``parent_count = ceil(child_count / arity)`` computed in-kernel. The serial compute encoder gives read-after-write ordering between consecutive level dispatches; the candidate need not insert any explicit barriers between levels.

Boundary policy: at each level, if ``child_count`` is not a multiple of ``arity`` the last group is padded with **zero** field elements (i.e. the missing children read as zero into the Poseidon2 state). The CPU reference uses the same policy; any other padding scheme is a correctness failure. At arity=4 with N=2^19 leaves the padding kicks in only at the topmost level (2 children -> [c0, c1, 0, 0]).

Correctness is bit-exact against the Python bigint reference applied to the entire tree (every intermediate digest is checked, not only the root). Outputs MUST be canonical (< p); a non-canonical value with the same residue class still counts as a mismatch. All test sizes satisfy t <= 4 and R_F <= 8, R_P <= 32; thread-private state arrays of size 4 are sufficient.

## Required kernel signature(s)

```
kernel void merkle_build_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &arity        [[buffer(5)]],
    constant uint      &t            [[buffer(6)]],
    constant uint      &r_f          [[buffer(7)]],
    constant uint      &r_p          [[buffer(8)]],
    constant uint      &in_offset    [[buffer(9)]],
    constant uint      &out_offset   [[buffer(10)]],
    constant uint      &child_count  [[buffer(11)]],
    uint p [[thread_position_in_grid]]);

Dispatch (host-fixed, one call per parent level):
  threadsPerGrid        = (parent_count, 1, 1)   rounded up to the TG width
  threadsPerThreadgroup = (min(parent_count, 64), 1, 1)
Each thread owns ONE parent; guard against p >= parent_count (the grid is rounded up to a multiple of the TG width). Threadgroup-cooperative and simdgroup-cooperative implementations are valid so long as the external buffer layout above is preserved and the host's one-dispatch-per-level invocation pattern is honored (the kernel sees exactly one level's worth of parents per dispatch via ``child_count``, ``in_offset``, ``out_offset``).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

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

    ulong e = ((ulong)carry << 32) - (ulong)carry;
    old = s;
    s += e;
    if (s < old) s += EPSILON;

    return gold_canonical(s);
}

inline ulong gold_add4(ulong a, ulong b, ulong c, ulong d) {
    ulong s = a + b;
    uint carry = (s < a) ? 1u : 0u;

    ulong old = s;
    s += c;
    carry += (s < old) ? 1u : 0u;

    old = s;
    s += d;
    carry += (s < old) ? 1u : 0u;

    ulong e = ((ulong)carry << 32) - (ulong)carry;
    old = s;
    s += e;
    if (s < old) s += EPSILON;

    return gold_canonical(s);
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
}

inline ulong gold_reduce_words(uint w0, uint w1, uint w2, uint w3) {
    ulong x_lo = ((ulong)w1 << 32) | (ulong)w0;

    ulong t0 = x_lo - (ulong)w3;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = ((ulong)w2 << 32) - (ulong)w2;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);
    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);
    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);
    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1 + p10_lo;
    c1 += (w1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;
    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;
    uint w2 = s3 + c1;
    c2 += (w2 < s3) ? 1u : 0u;

    uint w3 = p11_hi + c2;

    return gold_reduce_words(p00_lo, w1, w2, w3);
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00_lo = a0 * a0;
    uint p00_hi = mulhi(a0, a0);

    uint q_lo = a0 * a1;
    uint q_hi = mulhi(a0, a1);

    uint p11_lo = a1 * a1;
    uint p11_hi = mulhi(a1, a1);

    uint dbl0 = q_lo << 1;
    uint dbl1 = (q_hi << 1) | (q_lo >> 31);
    uint dbl2 = q_hi >> 31;

    uint s1 = p00_hi + dbl0;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1;

    uint s2 = dbl1 + p11_lo;
    uint c2 = (s2 < dbl1) ? 1u : 0u;
    uint w2 = s2 + c1;
    c2 += (w2 < s2) ? 1u : 0u;

    uint w3 = p11_hi + dbl2 + c2;

    return gold_reduce_words(p00_lo, w1, w2, w3);
}

inline ulong gold_mul_small_or_full(ulong c, ulong x) {
    if (x == 0ul) return 0ul;
    if (c <= 7ul) {
        if (c == 0ul) return 0ul;
        if (c == 1ul) return x;
        ulong r2 = gold_add(x, x);
        if (c == 2ul) return r2;
        ulong r3 = gold_add(r2, x);
        if (c == 3ul) return r3;
        ulong r4 = gold_add(r2, r2);
        if (c == 4ul) return r4;
        ulong r5 = gold_add(r4, x);
        if (c == 5ul) return r5;
        ulong r6 = gold_add(r4, r2);
        if (c == 6ul) return r6;
        return gold_add(r6, x);
    }
    if (c == (P_GOLD - 1ul)) return gold_neg(x);
    return gold_mul(c, x);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline bool mds3_is_jplusi(device const ulong *m) {
    return m[0] == 2ul && m[1] == 1ul && m[2] == 1ul &&
           m[3] == 1ul && m[4] == 2ul && m[5] == 1ul &&
           m[6] == 1ul && m[7] == 1ul && m[8] == 2ul;
}

inline bool mds3_has_unit_offdiag(device const ulong *m) {
    return m[1] == 1ul && m[2] == 1ul &&
           m[3] == 1ul && m[5] == 1ul &&
           m[6] == 1ul && m[7] == 1ul;
}

inline bool mds4_is_m4(device const ulong *m) {
    return m[0]  == 5ul && m[1]  == 7ul && m[2]  == 1ul && m[3]  == 3ul &&
           m[4]  == 4ul && m[5]  == 6ul && m[6]  == 1ul && m[7]  == 1ul &&
           m[8]  == 1ul && m[9]  == 3ul && m[10] == 5ul && m[11] == 7ul &&
           m[12] == 1ul && m[13] == 1ul && m[14] == 4ul && m[15] == 6ul;
}

#define APPLY_MDS3() do {                                      \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    y0 = gold_add(y0, gold_mul(m02, x2));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    y1 = gold_add(y1, gold_mul(m12, x2));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    y2 = gold_add(y2, gold_mul(m22, x2));                      \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_INIT2() do {                                \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_JPLUSI() do {                               \
    ulong s = gold_add3(x0, x1, x2);                           \
    ulong y0 = gold_add(s, x0);                                \
    ulong y1 = gold_add(s, x1);                                \
    ulong y2 = gold_add(s, x2);                                \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_JPLUSI_INIT_A2() do {                       \
    ulong s = gold_add(x0, x1);                                \
    ulong y0 = gold_add(s, x0);                                \
    ulong y1 = gold_add(s, x1);                                \
    x0 = y0; x1 = y1; x2 = s;                                  \
} while (false)

#define MDS3_JPLUSI_X0() (gold_add(gold_add3(x0, x1, x2), x0))

#define APPLY_MDS3_JDIAG() do {                                \
    ulong s = gold_add3(x0, x1, x2);                           \
    ulong y0 = gold_add(s, gold_mul_small_or_full(e0, x0));    \
    ulong y1 = gold_add(s, gold_mul_small_or_full(e1, x1));    \
    ulong y2 = gold_add(s, gold_mul_small_or_full(e2, x2));    \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define MDS3_JDIAG_X0() (gold_add(gold_add3(x0, x1, x2), gold_mul_small_or_full(e0, x0)))

#define APPLY_INT3() do {                                      \
    ulong s = gold_add3(x0, x1, x2);                           \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS4() do {                                      \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    y0 = gold_add(y0, gold_mul(m02, x2));                      \
    y0 = gold_add(y0, gold_mul(m03, x3));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    y1 = gold_add(y1, gold_mul(m12, x2));                      \
    y1 = gold_add(y1, gold_mul(m13, x3));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    y2 = gold_add(y2, gold_mul(m22, x2));                      \
    y2 = gold_add(y2, gold_mul(m23, x3));                      \
    ulong y3 = gold_mul(m30, x0);                              \
    y3 = gold_add(y3, gold_mul(m31, x1));                      \
    y3 = gold_add(y3, gold_mul(m32, x2));                      \
    y3 = gold_add(y3, gold_mul(m33, x3));                      \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_MDS4_INIT2() do {                                \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    ulong y3 = gold_mul(m30, x0);                              \
    y3 = gold_add(y3, gold_mul(m31, x1));                      \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_MDS4_M4() do {                                   \
    ulong a0 = gold_add(x0, x1);                               \
    ulong a1 = gold_add(x2, x3);                               \
    ulong a2 = gold_add(gold_add(x1, x1), a1);                 \
    ulong a3 = gold_add(gold_add(x3, x3), a0);                 \
    ulong b1 = gold_add(a1, a1);                               \
    ulong c1 = gold_add(b1, b1);                               \
    ulong a4 = gold_add(c1, a3);                               \
    ulong b0 = gold_add(a0, a0);                               \
    ulong c0 = gold_add(b0, b0);                               \
    ulong a5 = gold_add(c0, a2);                               \
    ulong y0 = gold_add(a3, a5);                               \
    ulong y1 = a5;                                             \
    ulong y2 = gold_add(a2, a4);                               \
    ulong y3 = a4;                                             \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

inline ulong mds4_m4_x0(ulong x0, ulong x1, ulong x2, ulong x3) {
    ulong a0 = gold_add(x0, x1);
    ulong a1 = gold_add(x2, x3);
    ulong a2 = gold_add(gold_add(x1, x1), a1);
    ulong a3 = gold_add(gold_add(x3, x3), a0);
    ulong b0 = gold_add(a0, a0);
    ulong c0 = gold_add(b0, b0);
    ulong a5 = gold_add(c0, a2);
    return gold_add(a3, a5);
}

#define APPLY_INT4() do {                                      \
    ulong s = gold_add4(x0, x1, x2, x3);                       \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    ulong y3 = gold_add(s, gold_mul(d3, x3));                  \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

inline ulong poseidon2_t3_jplusi_a2(
    ulong x0, ulong x1,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    ulong x2 = 0ul;
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JPLUSI_INIT_A2();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r + 1u < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    if (half_f < r_f) {
        uint o = (r_f - 1u) * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        return MDS3_JPLUSI_X0();
    }

    return x0;
}

inline ulong poseidon2_t3_generic(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint arity,
    uint r_f,
    uint r_p)
{
    const ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    const ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    const ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    if (arity <= 2u) {
        APPLY_MDS3_INIT2();
    } else {
        APPLY_MDS3();
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r + 1u < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3();
    }

    if (half_f < r_f) {
        uint o = (r_f - 1u) * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        ulong y0 = gold_mul(m00, x0);
        y0 = gold_add(y0, gold_mul(m01, x1));
        y0 = gold_add(y0, gold_mul(m02, x2));
        return y0;
    }

    return x0;
}

inline ulong poseidon2_t3_jplusi(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JPLUSI();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r + 1u < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    if (half_f < r_f) {
        uint o = (r_f - 1u) * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        return MDS3_JPLUSI_X0();
    }

    return x0;
}

inline ulong poseidon2_t3_jdiag(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    ulong e0, ulong e1, ulong e2,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JDIAG();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r + 1u < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG();
    }

    if (half_f < r_f) {
        uint o = (r_f - 1u) * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        return MDS3_JDIAG_X0();
    }

    return x0;
}

inline ulong poseidon2_t4_generic(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint arity,
    uint r_f,
    uint r_p)
{
    const ulong m00 = ext_mds[0],  m01 = ext_mds[1],  m02 = ext_mds[2],  m03 = ext_mds[3];
    const ulong m10 = ext_mds[4],  m11 = ext_mds[5],  m12 = ext_mds[6],  m13 = ext_mds[7];
    const ulong m20 = ext_mds[8],  m21 = ext_mds[9],  m22 = ext_mds[10], m23 = ext_mds[11];
    const ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    if (arity <= 2u) {
        APPLY_MDS4_INIT2();
    } else {
        APPLY_MDS4();
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT4();
    }

    for (uint r = half_f; r + 1u < r_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4();
    }

    if (half_f < r_f) {
        uint o = (r_f - 1u) << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        ulong y0 = gold_mul(m00, x0);
        y0 = gold_add(y0, gold_mul(m01, x1));
        y0 = gold_add(y0, gold_mul(m02, x2));
        y0 = gold_add(y0, gold_mul(m03, x3));
        return y0;
    }

    return x0;
}

inline ulong poseidon2_t4_m4(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    APPLY_MDS4_M4();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4_M4();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT4();
    }

    for (uint r = half_f; r + 1u < r_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4_M4();
    }

    if (half_f < r_f) {
        uint o = (r_f - 1u) << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        return mds4_m4_x0(x0, x1, x2, x3);
    }

    return x0;
}

kernel void merkle_build_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &arity        [[buffer(5)]],
    constant uint      &t            [[buffer(6)]],
    constant uint      &r_f          [[buffer(7)]],
    constant uint      &r_p          [[buffer(8)]],
    constant uint      &in_offset    [[buffer(9)]],
    constant uint      &out_offset   [[buffer(10)]],
    constant uint      &child_count  [[buffer(11)]],
    uint p [[thread_position_in_grid]])
{
    if (t == 3u && arity == 2u) {
        uint parent_count = (child_count + 1u) >> 1;
        if (p >= parent_count) return;

        uint base = p << 1;
        uint idx = in_offset + base;

        ulong x0 = tree[idx];
        ulong x1;
        if ((child_count & 1u) == 0u) {
            x1 = tree[idx + 1u];
        } else {
            x1 = (base + 1u < child_count) ? tree[idx + 1u] : 0ul;
        }

        ulong out;
        if (mds3_is_jplusi(ext_mds)) {
            out = poseidon2_t3_jplusi_a2(x0, x1, rc_ext, rc_int, int_diag, r_f, r_p);
        } else if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            out = poseidon2_t3_jdiag(x0, x1, 0ul, rc_ext, rc_int, int_diag, e0, e1, e2, r_f, r_p);
        } else {
            out = poseidon2_t3_generic(x0, x1, 0ul, rc_ext, rc_int, ext_mds, int_diag, 2u, r_f, r_p);
        }

        tree[out_offset + p] = out;
        return;
    }

    uint parent_count;
    if (arity == 2u) {
        parent_count = (child_count + 1u) >> 1;
    } else if (arity == 4u) {
        parent_count = (child_count + 3u) >> 2;
    } else {
        parent_count = (child_count + arity - 1u) / arity;
    }

    if (p >= parent_count) return;

    uint base;
    if (arity == 2u) {
        base = p << 1;
    } else if (arity == 4u) {
        base = p << 2;
    } else {
        base = p * arity;
    }

    uint idx = in_offset + base;

    if (t == 3u) {
        ulong x0 = tree[idx];
        ulong x1 = 0ul;
        ulong x2 = 0ul;

        if (arity > 1u && base + 1u < child_count) x1 = tree[idx + 1u];
        if (arity > 2u && base + 2u < child_count) x2 = tree[idx + 2u];

        ulong out;
        if (mds3_is_jplusi(ext_mds)) {
            out = poseidon2_t3_jplusi(x0, x1, x2, rc_ext, rc_int, int_diag, r_f, r_p);
        } else if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            out = poseidon2_t3_jdiag(x0, x1, x2, rc_ext, rc_int, int_diag, e0, e1, e2, r_f, r_p);
        } else {
            out = poseidon2_t3_generic(x0, x1, x2, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    } else {
        ulong x0 = tree[idx];
        ulong x1 = 0ul;
        ulong x2 = 0ul;
        ulong x3 = 0ul;

        if (arity == 4u && ((child_count & 3u) == 0u)) {
            x1 = tree[idx + 1u];
            x2 = tree[idx + 2u];
            x3 = tree[idx + 3u];
        } else if (arity == 4u && base + 3u < child_count) {
            x1 = tree[idx + 1u];
            x2 = tree[idx + 2u];
            x3 = tree[idx + 3u];
        } else {
            if (arity > 1u && base + 1u < child_count) x1 = tree[idx + 1u];
            if (arity > 2u && base + 2u < child_count) x2 = tree[idx + 2u];
            if (arity > 3u && base + 3u < child_count) x3 = tree[idx + 3u];
        }

        ulong out;
        if (mds4_is_m4(ext_mds)) {
            out = poseidon2_t4_m4(x0, x1, x2, x3, rc_ext, rc_int, int_diag, r_f, r_p);
        } else {
            out = poseidon2_t4_generic(x0, x1, x2, x3, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    }
}
```

Result of previous attempt:
           a2_N64K: correct, 1.37 ms, 15.8 Gmodmul/s (int64) (29.6% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 3.46 ms, 25.1 Gmodmul/s (int64) (47.0% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 11.60 ms, 29.9 Gmodmul/s (int64) (56.1% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.4272

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gold_canonical(ulong x) {
    ulong y = x - P_GOLD;
    return (x >= P_GOLD) ? y : x;
}

inline ulong gold_add(ulong a, ulong b) {
    ulong s = a + b;
    s += (s < a) ? EPSILON : 0ul;
    return gold_canonical(s);
}

inline ulong gold_neg(ulong x) {
    return (x == 0ul) ? 0ul : (P_GOLD - x);
}

inline ulong gold_sub_one(ulong x) {
    return (x == 0ul) ? (P_GOLD - 1ul) : (x - 1ul);
}

inline ulong gold_reduce_words(uint w0, uint w1, uint w2, uint w3) {
    ulong x_lo = ((ulong)w1 << 32) | (ulong)w0;

    ulong t0 = x_lo - (ulong)w3;
    t0 -= (t0 > x_lo) ? EPSILON : 0ul;

    ulong t1 = ((ulong)w2 << 32) - (ulong)w2;

    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? EPSILON : 0ul;

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    uint p00_lo = a0 * b0;
    uint p00_hi = mulhi(a0, b0);
    uint p01_lo = a0 * b1;
    uint p01_hi = mulhi(a0, b1);
    uint p10_lo = a1 * b0;
    uint p10_hi = mulhi(a1, b0);
    uint p11_lo = a1 * b1;
    uint p11_hi = mulhi(a1, b1);

    uint s1 = p00_hi + p01_lo;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1 + p10_lo;
    c1 += (w1 < s1) ? 1u : 0u;

    uint s2 = p01_hi + p10_hi;
    uint c2 = (s2 < p01_hi) ? 1u : 0u;
    uint s3 = s2 + p11_lo;
    c2 += (s3 < s2) ? 1u : 0u;
    uint w2 = s3 + c1;
    c2 += (w2 < s3) ? 1u : 0u;

    uint w3 = p11_hi + c2;

    return gold_reduce_words(p00_lo, w1, w2, w3);
}

inline ulong gold_square(ulong a) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);

    uint p00_lo = a0 * a0;
    uint p00_hi = mulhi(a0, a0);

    uint q_lo = a0 * a1;
    uint q_hi = mulhi(a0, a1);

    uint p11_lo = a1 * a1;
    uint p11_hi = mulhi(a1, a1);

    uint dbl0 = q_lo << 1;
    uint dbl1 = (q_hi << 1) | (q_lo >> 31);
    uint dbl2 = q_hi >> 31;

    uint s1 = p00_hi + dbl0;
    uint c1 = (s1 < p00_hi) ? 1u : 0u;
    uint w1 = s1;

    uint s2 = dbl1 + p11_lo;
    uint c2 = (s2 < dbl1) ? 1u : 0u;
    uint w2 = s2 + c1;
    c2 += (w2 < s2) ? 1u : 0u;

    uint w3 = p11_hi + dbl2 + c2;

    return gold_reduce_words(p00_lo, w1, w2, w3);
}

inline ulong gold_mul_small_or_full(ulong c, ulong x) {
    if (c <= 7ul) {
        if (c == 0ul) return 0ul;
        if (c == 1ul) return x;
        ulong r2 = gold_add(x, x);
        if (c == 2ul) return r2;
        ulong r3 = gold_add(r2, x);
        if (c == 3ul) return r3;
        ulong r4 = gold_add(r2, r2);
        if (c == 4ul) return r4;
        ulong r5 = gold_add(r4, x);
        if (c == 5ul) return r5;
        ulong r6 = gold_add(r4, r2);
        if (c == 6ul) return r6;
        return gold_add(r6, x);
    }
    if (c == (P_GOLD - 1ul)) return gold_neg(x);
    return gold_mul(c, x);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_square(x);
    ulong x4 = gold_square(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline bool mds3_is_jplusi(device const ulong *m) {
    return m[0] == 2ul && m[1] == 1ul && m[2] == 1ul &&
           m[3] == 1ul && m[4] == 2ul && m[5] == 1ul &&
           m[6] == 1ul && m[7] == 1ul && m[8] == 2ul;
}

inline bool mds3_has_unit_offdiag(device const ulong *m) {
    return m[1] == 1ul && m[2] == 1ul &&
           m[3] == 1ul && m[5] == 1ul &&
           m[6] == 1ul && m[7] == 1ul;
}

inline bool mds4_is_m4(device const ulong *m) {
    return m[0]  == 5ul && m[1]  == 7ul && m[2]  == 1ul && m[3]  == 3ul &&
           m[4]  == 4ul && m[5]  == 6ul && m[6]  == 1ul && m[7]  == 1ul &&
           m[8]  == 1ul && m[9]  == 3ul && m[10] == 5ul && m[11] == 7ul &&
           m[12] == 1ul && m[13] == 1ul && m[14] == 4ul && m[15] == 6ul;
}

#define APPLY_MDS3() do {                                      \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    y0 = gold_add(y0, gold_mul(m02, x2));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    y1 = gold_add(y1, gold_mul(m12, x2));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    y2 = gold_add(y2, gold_mul(m22, x2));                      \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_INIT2() do {                                \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_JPLUSI() do {                               \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, x0);                                \
    ulong y1 = gold_add(s, x1);                                \
    ulong y2 = gold_add(s, x2);                                \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS3_JDIAG() do {                                \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul_small_or_full(e0, x0));    \
    ulong y1 = gold_add(s, gold_mul_small_or_full(e1, x1));    \
    ulong y2 = gold_add(s, gold_mul_small_or_full(e2, x2));    \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_INT3() do {                                      \
    ulong s = gold_add(gold_add(x0, x1), x2);                  \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    x0 = y0; x1 = y1; x2 = y2;                                 \
} while (false)

#define APPLY_MDS4() do {                                      \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    y0 = gold_add(y0, gold_mul(m02, x2));                      \
    y0 = gold_add(y0, gold_mul(m03, x3));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    y1 = gold_add(y1, gold_mul(m12, x2));                      \
    y1 = gold_add(y1, gold_mul(m13, x3));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    y2 = gold_add(y2, gold_mul(m22, x2));                      \
    y2 = gold_add(y2, gold_mul(m23, x3));                      \
    ulong y3 = gold_mul(m30, x0);                              \
    y3 = gold_add(y3, gold_mul(m31, x1));                      \
    y3 = gold_add(y3, gold_mul(m32, x2));                      \
    y3 = gold_add(y3, gold_mul(m33, x3));                      \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_MDS4_INIT2() do {                                \
    ulong y0 = gold_mul(m00, x0);                              \
    y0 = gold_add(y0, gold_mul(m01, x1));                      \
    ulong y1 = gold_mul(m10, x0);                              \
    y1 = gold_add(y1, gold_mul(m11, x1));                      \
    ulong y2 = gold_mul(m20, x0);                              \
    y2 = gold_add(y2, gold_mul(m21, x1));                      \
    ulong y3 = gold_mul(m30, x0);                              \
    y3 = gold_add(y3, gold_mul(m31, x1));                      \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_MDS4_M4() do {                                   \
    ulong a0 = gold_add(x0, x1);                               \
    ulong a1 = gold_add(x2, x3);                               \
    ulong a2 = gold_add(gold_add(x1, x1), a1);                 \
    ulong a3 = gold_add(gold_add(x3, x3), a0);                 \
    ulong b1 = gold_add(a1, a1);                               \
    ulong c1 = gold_add(b1, b1);                               \
    ulong a4 = gold_add(c1, a3);                               \
    ulong b0 = gold_add(a0, a0);                               \
    ulong c0 = gold_add(b0, b0);                               \
    ulong a5 = gold_add(c0, a2);                               \
    ulong y0 = gold_add(a3, a5);                               \
    ulong y1 = a5;                                             \
    ulong y2 = gold_add(a2, a4);                               \
    ulong y3 = a4;                                             \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

#define APPLY_INT4() do {                                      \
    ulong s = gold_add(gold_add(gold_add(x0, x1), x2), x3);    \
    ulong y0 = gold_add(s, gold_mul(d0, x0));                  \
    ulong y1 = gold_add(s, gold_mul(d1, x1));                  \
    ulong y2 = gold_add(s, gold_mul(d2, x2));                  \
    ulong y3 = gold_add(s, gold_mul(d3, x3));                  \
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;                        \
} while (false)

inline ulong poseidon2_t3_generic(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint arity,
    uint r_f,
    uint r_p)
{
    const ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    const ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    const ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    if (arity <= 2u) {
        APPLY_MDS3_INIT2();
    } else {
        APPLY_MDS3();
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3();
    }

    return x0;
}

inline ulong poseidon2_t3_jplusi(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JPLUSI();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JPLUSI();
    }

    return x0;
}

inline ulong poseidon2_t3_jdiag(
    ulong x0, ulong x1, ulong x2,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    ulong e0, ulong e1, ulong e2,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    APPLY_MDS3_JDIAG();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT3();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r * 3u;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        APPLY_MDS3_JDIAG();
    }

    return x0;
}

inline ulong poseidon2_t4_generic(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *ext_mds,
    device const ulong *int_diag,
    uint arity,
    uint r_f,
    uint r_p)
{
    const ulong m00 = ext_mds[0],  m01 = ext_mds[1],  m02 = ext_mds[2],  m03 = ext_mds[3];
    const ulong m10 = ext_mds[4],  m11 = ext_mds[5],  m12 = ext_mds[6],  m13 = ext_mds[7];
    const ulong m20 = ext_mds[8],  m21 = ext_mds[9],  m22 = ext_mds[10], m23 = ext_mds[11];
    const ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];

    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    if (arity <= 2u) {
        APPLY_MDS4_INIT2();
    } else {
        APPLY_MDS4();
    }

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT4();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4();
    }

    return x0;
}

inline ulong poseidon2_t4_m4(
    ulong x0, ulong x1, ulong x2, ulong x3,
    device const ulong *rc_ext,
    device const ulong *rc_int,
    device const ulong *int_diag,
    uint r_f,
    uint r_p)
{
    const ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    APPLY_MDS4_M4();

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4_M4();
    }

    for (uint r = 0u; r < r_p; ++r) {
        x0 = sbox7(gold_add(x0, rc_int[r]));
        APPLY_INT4();
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint o = r << 2;
        x0 = sbox7(gold_add(x0, rc_ext[o + 0u]));
        x1 = sbox7(gold_add(x1, rc_ext[o + 1u]));
        x2 = sbox7(gold_add(x2, rc_ext[o + 2u]));
        x3 = sbox7(gold_add(x3, rc_ext[o + 3u]));
        APPLY_MDS4_M4();
    }

    return x0;
}

kernel void merkle_build_level(
    device       ulong *tree         [[buffer(0)]],
    device const ulong *rc_ext       [[buffer(1)]],
    device const ulong *rc_int       [[buffer(2)]],
    device const ulong *ext_mds      [[buffer(3)]],
    device const ulong *int_diag     [[buffer(4)]],
    constant uint      &arity        [[buffer(5)]],
    constant uint      &t            [[buffer(6)]],
    constant uint      &r_f          [[buffer(7)]],
    constant uint      &r_p          [[buffer(8)]],
    constant uint      &in_offset    [[buffer(9)]],
    constant uint      &out_offset   [[buffer(10)]],
    constant uint      &child_count  [[buffer(11)]],
    uint p [[thread_position_in_grid]])
{
    if (t == 3u && arity == 2u) {
        uint parent_count = (child_count + 1u) >> 1;
        if (p >= parent_count) return;

        uint base = p << 1;
        uint idx = in_offset + base;

        ulong x0 = tree[idx];
        ulong x1;
        if ((child_count & 1u) == 0u) {
            x1 = tree[idx + 1u];
        } else {
            x1 = (base + 1u < child_count) ? tree[idx + 1u] : 0ul;
        }
        ulong x2 = 0ul;

        ulong out;
        if (mds3_is_jplusi(ext_mds)) {
            out = poseidon2_t3_jplusi(x0, x1, x2, rc_ext, rc_int, int_diag, r_f, r_p);
        } else if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            out = poseidon2_t3_jdiag(x0, x1, x2, rc_ext, rc_int, int_diag, e0, e1, e2, r_f, r_p);
        } else {
            out = poseidon2_t3_generic(x0, x1, x2, rc_ext, rc_int, ext_mds, int_diag, 2u, r_f, r_p);
        }

        tree[out_offset + p] = out;
        return;
    }

    uint parent_count;
    if (arity == 2u) {
        parent_count = (child_count + 1u) >> 1;
    } else if (arity == 4u) {
        parent_count = (child_count + 3u) >> 2;
    } else {
        parent_count = (child_count + arity - 1u) / arity;
    }

    if (p >= parent_count) return;

    uint base;
    if (arity == 2u) {
        base = p << 1;
    } else if (arity == 4u) {
        base = p << 2;
    } else {
        base = p * arity;
    }

    uint idx = in_offset + base;

    if (t == 3u) {
        ulong x0 = tree[idx];
        ulong x1 = 0ul;
        ulong x2 = 0ul;

        if (arity > 1u && base + 1u < child_count) x1 = tree[idx + 1u];
        if (arity > 2u && base + 2u < child_count) x2 = tree[idx + 2u];

        ulong out;
        if (mds3_is_jplusi(ext_mds)) {
            out = poseidon2_t3_jplusi(x0, x1, x2, rc_ext, rc_int, int_diag, r_f, r_p);
        } else if (mds3_has_unit_offdiag(ext_mds)) {
            ulong e0 = gold_sub_one(ext_mds[0]);
            ulong e1 = gold_sub_one(ext_mds[4]);
            ulong e2 = gold_sub_one(ext_mds[8]);
            out = poseidon2_t3_jdiag(x0, x1, x2, rc_ext, rc_int, int_diag, e0, e1, e2, r_f, r_p);
        } else {
            out = poseidon2_t3_generic(x0, x1, x2, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    } else {
        ulong x0 = tree[idx];
        ulong x1 = 0ul;
        ulong x2 = 0ul;
        ulong x3 = 0ul;

        if (arity == 4u && ((child_count & 3u) == 0u)) {
            x1 = tree[idx + 1u];
            x2 = tree[idx + 2u];
            x3 = tree[idx + 3u];
        } else if (arity == 4u && base + 3u < child_count) {
            x1 = tree[idx + 1u];
            x2 = tree[idx + 2u];
            x3 = tree[idx + 3u];
        } else {
            if (arity > 1u && base + 1u < child_count) x1 = tree[idx + 1u];
            if (arity > 2u && base + 2u < child_count) x2 = tree[idx + 2u];
            if (arity > 3u && base + 3u < child_count) x3 = tree[idx + 3u];
        }

        ulong out;
        if (mds4_is_m4(ext_mds)) {
            out = poseidon2_t4_m4(x0, x1, x2, x3, rc_ext, rc_int, int_diag, r_f, r_p);
        } else {
            out = poseidon2_t4_generic(x0, x1, x2, x3, rc_ext, rc_int, ext_mds, int_diag, arity, r_f, r_p);
        }

        tree[out_offset + p] = out;
    }
}
```

Incumbent result:
           a2_N64K: correct, 1.37 ms, 15.8 Gmodmul/s (int64) (29.6% of 53 Gops/s (int64 mul, est))
          a2_N256K: correct, 3.39 ms, 25.6 Gmodmul/s (int64) (47.9% of 53 Gops/s (int64 mul, est))
            a2_N1M: correct, 11.54 ms, 30.1 Gmodmul/s (int64) (56.4% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.4309

## History

- iter  0: compile=OK | correct=True | score=0.3112673594115535
- iter  1: compile=OK | correct=True | score=0.35198822540211555
- iter  2: compile=OK | correct=True | score=0.43086253049174383
- iter  3: compile=OK | correct=True | score=0.3679635157239129
- iter  4: compile=OK | correct=True | score=0.42718162310163005

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
