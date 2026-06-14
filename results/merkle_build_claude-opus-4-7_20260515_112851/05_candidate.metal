#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;

inline ulong gadd(ulong a, ulong b) {
    ulong s = a + b;
    if (s < a) s += EPSILON;
    if (s >= P_GOLD) s -= P_GOLD;
    return s;
}

inline void umul128(ulong a, ulong b, thread ulong &lo, thread ulong &hi) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;

    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    lo = (p00 & EPSILON) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// Reduce 128-bit value with x_hi possibly up to ~2^62 (still safe as long as
// computations don't overflow intermediate u64). We use the standard
// Goldilocks reduction: x = x_lo + x_hi_lo * 2^32 - x_hi_lo - x_hi_hi (mod p).
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong r = t0 + t1;
    if (r < t0) r += EPSILON;
    if (r >= P_GOLD) r -= P_GOLD;
    return r;
}

inline ulong gmul(ulong a, ulong b) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    return gold_reduce128(lo, hi);
}

// Accumulate a*b into (acc_lo, acc_hi, acc_carry). Sum of up to 4 products of
// canonical values fits in 130 bits, so a 3-limb accumulator (64+64+ small) is safe.
inline void mac128(ulong a, ulong b,
                   thread ulong &acc_lo, thread ulong &acc_hi, thread ulong &acc_c) {
    ulong lo, hi;
    umul128(a, b, lo, hi);
    ulong new_lo = acc_lo + lo;
    ulong c1 = (ulong)(new_lo < acc_lo);
    acc_lo = new_lo;
    ulong new_hi = acc_hi + hi + c1;
    ulong c2 = (ulong)(new_hi < acc_hi) | (ulong)(new_hi == acc_hi && (hi + c1) != 0ul && new_hi < hi + c1);
    // Simpler: detect carry by comparing.
    // Actually: new_hi = acc_hi + (hi + c1); carry if new_hi < acc_hi OR (hi+c1 overflowed).
    // hi+c1 overflows only if hi == ~0 and c1==1; treat as edge.
    ulong sum2 = hi + c1;
    ulong cA = (ulong)(sum2 < hi);
    ulong new_hi2 = acc_hi + sum2;
    ulong cB = (ulong)(new_hi2 < acc_hi);
    acc_hi = new_hi2;
    acc_c += cA + cB;
    (void)c2; (void)new_hi;
}

// Reduce a 3-limb accumulator (lo + hi*2^64 + c*2^128) mod p.
// p = 2^64 - 2^32 + 1.  2^64 ≡ 2^32 - 1 (mod p).  2^128 ≡ (2^32-1)^2 mod p.
// We fold: total = lo + hi*(2^32-1) + c*(2^32-1)^2 effectively, then reduce.
// Simpler: reduce (hi, c) first to a single u64-ish, then call gold_reduce128.
inline ulong reduce_acc(ulong acc_lo, ulong acc_hi, ulong acc_c) {
    // First reduce the top: combine acc_c (small) into acc_hi-style.
    // We have value V = acc_lo + acc_hi * 2^64 + acc_c * 2^128.
    // 2^128 mod p: compute once. (2^32-1)^2 = 2^64 - 2^33 + 1 ≡ (2^32-1) - 2^33 + 1 (mod p)
    //   = 2^32 - 1 - 2^33 + 1 = -2^32 (mod p) = p - 2^32 = 2^64 - 2*2^32 + 1.
    // So acc_c * (p - 2^32) ≡ -acc_c * 2^32 (mod p).
    // Just call gold_reduce128 twice via a loop: first reduce (acc_hi, acc_c) to a 64-bit-ish residue,
    // then combine with acc_lo.
    if (acc_c != 0ul) {
        // Reduce (acc_hi as low, acc_c as high) to a u64 in [0, p) then treat it as extra high contribution.
        ulong top = gold_reduce128(acc_hi, acc_c); // this is (acc_hi + acc_c*2^64) mod p, in [0,p)
        // Now value = acc_lo + top * 2^64.
        return gold_reduce128(acc_lo, top);
    }
    return gold_reduce128(acc_lo, acc_hi);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gmul(x, x);
    ulong x4 = gmul(x2, x2);
    ulong x3 = gmul(x2, x);
    return gmul(x4, x3);
}

// Lazy 3x3 MDS multiply with single reduction per row.
inline void mds3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                 const thread ulong M[9]) {
    ulong lo, hi, c;
    ulong n0, n1, n2;
    lo = 0; hi = 0; c = 0;
    mac128(M[0], s0, lo, hi, c);
    mac128(M[1], s1, lo, hi, c);
    mac128(M[2], s2, lo, hi, c);
    n0 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[3], s0, lo, hi, c);
    mac128(M[4], s1, lo, hi, c);
    mac128(M[5], s2, lo, hi, c);
    n1 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[6], s0, lo, hi, c);
    mac128(M[7], s1, lo, hi, c);
    mac128(M[8], s2, lo, hi, c);
    n2 = reduce_acc(lo, hi, c);

    s0 = n0; s1 = n1; s2 = n2;
}

inline void mds4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                 const thread ulong M[16]) {
    ulong lo, hi, c;
    ulong n0, n1, n2, n3;
    lo = 0; hi = 0; c = 0;
    mac128(M[0], s0, lo, hi, c);
    mac128(M[1], s1, lo, hi, c);
    mac128(M[2], s2, lo, hi, c);
    mac128(M[3], s3, lo, hi, c);
    n0 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[4], s0, lo, hi, c);
    mac128(M[5], s1, lo, hi, c);
    mac128(M[6], s2, lo, hi, c);
    mac128(M[7], s3, lo, hi, c);
    n1 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[8], s0, lo, hi, c);
    mac128(M[9], s1, lo, hi, c);
    mac128(M[10], s2, lo, hi, c);
    mac128(M[11], s3, lo, hi, c);
    n2 = reduce_acc(lo, hi, c);

    lo = 0; hi = 0; c = 0;
    mac128(M[12], s0, lo, hi, c);
    mac128(M[13], s1, lo, hi, c);
    mac128(M[14], s2, lo, hi, c);
    mac128(M[15], s3, lo, hi, c);
    n3 = reduce_acc(lo, hi, c);

    s0 = n0; s1 = n1; s2 = n2; s3 = n3;
}

inline void poseidon2_t3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[9],
                         const thread ulong D[3],
                         uint r_f, uint r_p)
{
    mds3(s0, s1, s2, M);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        mds3(s0, s1, s2, M);
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0, s1), s2);
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 3u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        mds3(s0, s1, s2, M);
    }
}

inline void poseidon2_t4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                         device const ulong *rc_ext,
                         device const ulong *rc_int,
                         const thread ulong M[16],
                         const thread ulong D[4],
                         uint r_f, uint r_p)
{
    mds4(s0, s1, s2, s3, M);

    uint half_f = r_f >> 1u;

    for (uint r = 0u; r < half_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        mds4(s0, s1, s2, s3, M);
    }

    for (uint r = 0u; r < r_p; ++r) {
        s0 = sbox7(gadd(s0, rc_int[r]));
        ulong sum = gadd(gadd(s0, s1), gadd(s2, s3));
        ulong n0 = gadd(sum, gmul(D[0], s0));
        ulong n1 = gadd(sum, gmul(D[1], s1));
        ulong n2 = gadd(sum, gmul(D[2], s2));
        ulong n3 = gadd(sum, gmul(D[3], s3));
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        uint b = r * 4u;
        s0 = sbox7(gadd(s0, rc_ext[b + 0u]));
        s1 = sbox7(gadd(s1, rc_ext[b + 1u]));
        s2 = sbox7(gadd(s2, rc_ext[b + 2u]));
        s3 = sbox7(gadd(s3, rc_ext[b + 3u]));
        mds4(s0, s1, s2, s3, M);
    }
}

inline void poseidon2_generic(thread ulong *state,
                              device const ulong *rc_ext,
                              device const ulong *rc_int,
                              device const ulong *ext_mds,
                              device const ulong *int_diag,
                              uint t, uint r_f, uint r_p)
{
    ulong tmp[4];

    for (uint i = 0u; i < t; ++i) {
        ulong lo = 0, hi = 0, c = 0;
        for (uint j = 0u; j < t; ++j) mac128(ext_mds[i * t + j], state[j], lo, hi, c);
        tmp[i] = reduce_acc(lo, hi, c);
    }
    for (uint i = 0u; i < t; ++i) state[i] = tmp[i];

    uint half_f = r_f >> 1u;
    for (uint r = 0u; r < half_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        for (uint i = 0u; i < t; ++i) {
            ulong lo = 0, hi = 0, c = 0;
            for (uint j = 0u; j < t; ++j) mac128(ext_mds[i * t + j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    for (uint r = 0u; r < r_p; ++r) {
        state[0] = sbox7(gadd(state[0], rc_int[r]));
        ulong s = 0ul;
        for (uint i = 0u; i < t; ++i) s = gadd(s, state[i]);
        for (uint i = 0u; i < t; ++i) tmp[i] = gadd(s, gmul(int_diag[i], state[i]));
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
    for (uint r = half_f; r < r_f; ++r) {
        for (uint i = 0u; i < t; ++i) state[i] = sbox7(gadd(state[i], rc_ext[r * t + i]));
        for (uint i = 0u; i < t; ++i) {
            ulong lo = 0, hi = 0, c = 0;
            for (uint j = 0u; j < t; ++j) mac128(ext_mds[i * t + j], state[j], lo, hi, c);
            tmp[i] = reduce_acc(lo, hi, c);
        }
        for (uint i = 0u; i < t; ++i) state[i] = tmp[i];
    }
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
    uint parent_count = (child_count + arity - 1u) / arity;
    if (p >= parent_count) return;

    uint base = p * arity;

    if (t == 3u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];

        ulong M[9];
        for (uint k = 0u; k < 9u; ++k) M[k] = ext_mds[k];
        ulong D[3];
        D[0] = int_diag[0]; D[1] = int_diag[1]; D[2] = int_diag[2];

        poseidon2_t3(s0, s1, s2, rc_ext, rc_int, M, D, r_f, r_p);

        tree[out_offset + p] = s0;
        return;
    }

    if (t == 4u) {
        ulong s0 = 0ul, s1 = 0ul, s2 = 0ul, s3 = 0ul;
        if (base + 0u < child_count) s0 = tree[in_offset + base + 0u];
        if (base + 1u < child_count) s1 = tree[in_offset + base + 1u];
        if (arity >= 3u && base + 2u < child_count) s2 = tree[in_offset + base + 2u];
        if (arity >= 4u && base + 3u < child_count) s3 = tree[in_offset + base + 3u];

        ulong M[16];
        for (uint k = 0u; k < 16u; ++k) M[k] = ext_mds[k];
        ulong D[4];
        D[0] = int_diag[0]; D[1] = int_diag[1]; D[2] = int_diag[2]; D[3] = int_diag[3];

        poseidon2_t4(s0, s1, s2, s3, rc_ext, rc_int, M, D, r_f, r_p);

        tree[out_offset + p] = s0;
        return;
    }

    ulong state[4] = {0ul, 0ul, 0ul, 0ul};
    for (uint i = 0u; i < arity; ++i) {
        uint src = base + i;
        if (src < child_count && i < t) state[i] = tree[in_offset + src];
    }
    poseidon2_generic(state, rc_ext, rc_int, ext_mds, int_diag, t, r_f, r_p);
    tree[out_offset + p] = state[0];
}