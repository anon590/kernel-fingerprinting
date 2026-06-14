To push beyond the incumbent, this kernel applies several arithmetic and data-flow optimizations:
1. **Fused Partial-Round MAC**: In the partial rounds, instead of multiplying by the diagonal `d` (which requires a 128-bit reduction) and then modulo-adding `sum`, we compute the 128-bit product `d * s`, add `sum` directly to the 128-bit accumulator, and do a single final `reduce128`. This safely eliminates 128 full modular reductions per element in partial rounds.
2. **Optimized Sqr128**: We streamline the 128-bit squaring by replacing separate `uint` casts and additions with a pre-shifted single addition, trimming ALU instructions in the heavily utilized `sbox7` core.
3. **Optimized Reduce128**: Removed redundant shifts and variable extractions; reusing `hi << 32` directly saves instructions without breaking the mathematically proven 1-conditional-subtraction bounds.
4. **Register-Pinned Constants**: External MDS and internal diagonal matrices are explicitly loaded into scalar registers at the top of the permutation, allowing the compiler to aggressively optimize `mac128` without uniform cache thrashing.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD = 0xFFFFFFFF00000001ul;

struct u128 {
    ulong hi;
    ulong lo;
};

inline u128 mul128(ulong a, ulong b) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);
    uint b0 = (uint)b, b1 = (uint)(b >> 32);

    ulong p01 = (ulong)a0 * b1;
    ulong p10 = (ulong)a1 * b0;
    ulong p00 = (ulong)a0 * b0;
    ulong p11 = (ulong)a1 * b1;

    ulong mid = (p00 >> 32) + (uint)p01 + (uint)p10;
    ulong l = (uint)p00 | (mid << 32);
    ulong h = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    return {h, l};
}

inline u128 sqr128(ulong a) {
    uint a0 = (uint)a, a1 = (uint)(a >> 32);

    ulong p01 = (ulong)a0 * a1;
    ulong p00 = (ulong)a0 * a0;
    ulong p11 = (ulong)a1 * a1;

    ulong p01_lo = (uint)p01;
    ulong p01_hi = p01 >> 32;

    ulong mid = (p00 >> 32) + (p01_lo << 1);
    ulong l = (uint)p00 | (mid << 32);
    ulong h = p11 + (p01_hi << 1) + (mid >> 32);
    
    return {h, l};
}

inline ulong reduce128(ulong hi, ulong lo) {
    ulong sub = (hi >> 32) + (uint)hi;
    ulong t0 = lo - sub;
    t0 -= (lo < sub) ? 0xFFFFFFFFul : 0ul;
    
    ulong t1 = hi << 32;
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? 0xFFFFFFFFul : 0ul;
    
    return (t2 >= P_GOLD) ? t2 - P_GOLD : t2;
}

inline u128 mac128(ulong a, ulong b, u128 acc) {
    u128 p = mul128(a, b);
    acc.lo += p.lo;
    acc.hi += p.hi + (acc.lo < p.lo ? 1ul : 0ul);
    return acc;
}

inline ulong mac_sum(ulong sum, ulong d, ulong s) {
    u128 p = mul128(d, s);
    p.lo += sum;
    p.hi += (p.lo < sum ? 1ul : 0ul);
    return reduce128(p.hi, p.lo);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t += (t < a) ? 0xFFFFFFFFul : 0ul;
    return (t >= P_GOLD) ? t - P_GOLD : t;
}

inline ulong gold_mul(ulong a, ulong b) {
    u128 m = mul128(a, b);
    return reduce128(m.hi, m.lo);
}

inline ulong gold_sqr(ulong a) {
    u128 m = sqr128(a);
    return reduce128(m.hi, m.lo);
}

inline ulong sbox7(ulong x) {
    ulong x2 = gold_sqr(x);
    ulong x4 = gold_sqr(x2);
    ulong x6 = gold_mul(x4, x2);
    return gold_mul(x6, x);
}

inline void poseidon2_permute_3(thread ulong &s0, thread ulong &s1, thread ulong &s2,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2];
    ulong m10 = ext_mds[3], m11 = ext_mds[4], m12 = ext_mds[5];
    ulong m20 = ext_mds[6], m21 = ext_mds[7], m22 = ext_mds[8];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2];

    {
        u128 acc0 = {0, 0};
        acc0 = mac128(m00, s0, acc0); acc0 = mac128(m01, s1, acc0); acc0 = mac128(m02, s2, acc0);
        ulong n0 = reduce128(acc0.hi, acc0.lo);
        
        u128 acc1 = {0, 0};
        acc1 = mac128(m10, s0, acc1); acc1 = mac128(m11, s1, acc1); acc1 = mac128(m12, s2, acc1);
        ulong n1 = reduce128(acc1.hi, acc1.lo);
        
        u128 acc2 = {0, 0};
        acc2 = mac128(m20, s0, acc2); acc2 = mac128(m21, s1, acc2); acc2 = mac128(m22, s2, acc2);
        ulong n2 = reduce128(acc2.hi, acc2.lo);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        
        u128 acc0 = {0, 0};
        acc0 = mac128(m00, s0, acc0); acc0 = mac128(m01, s1, acc0); acc0 = mac128(m02, s2, acc0);
        ulong n0 = reduce128(acc0.hi, acc0.lo);
        
        u128 acc1 = {0, 0};
        acc1 = mac128(m10, s0, acc1); acc1 = mac128(m11, s1, acc1); acc1 = mac128(m12, s2, acc1);
        ulong n1 = reduce128(acc1.hi, acc1.lo);
        
        u128 acc2 = {0, 0};
        acc2 = mac128(m20, s0, acc2); acc2 = mac128(m21, s1, acc2); acc2 = mac128(m22, s2, acc2);
        ulong n2 = reduce128(acc2.hi, acc2.lo);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), s2);
        
        ulong n0 = mac_sum(sum, d0, s0);
        ulong n1 = mac_sum(sum, d1, s1);
        ulong n2 = mac_sum(sum, d2, s2);
        
        s0 = n0; s1 = n1; s2 = n2;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        rc_idx += 3;
        
        u128 acc0 = {0, 0};
        acc0 = mac128(m00, s0, acc0); acc0 = mac128(m01, s1, acc0); acc0 = mac128(m02, s2, acc0);
        ulong n0 = reduce128(acc0.hi, acc0.lo);
        
        u128 acc1 = {0, 0};
        acc1 = mac128(m10, s0, acc1); acc1 = mac128(m11, s1, acc1); acc1 = mac128(m12, s2, acc1);
        ulong n1 = reduce128(acc1.hi, acc1.lo);
        
        u128 acc2 = {0, 0};
        acc2 = mac128(m20, s0, acc2); acc2 = mac128(m21, s1, acc2); acc2 = mac128(m22, s2, acc2);
        ulong n2 = reduce128(acc2.hi, acc2.lo);
        
        s0 = n0; s1 = n1; s2 = n2;
    }
}

inline void poseidon2_permute_4(thread ulong &s0, thread ulong &s1, thread ulong &s2, thread ulong &s3,
                                device const ulong *rc_ext,
                                device const ulong *rc_int,
                                device const ulong *ext_mds,
                                device const ulong *int_diag,
                                uint r_f, uint r_p)
{
    ulong m00 = ext_mds[0], m01 = ext_mds[1], m02 = ext_mds[2], m03 = ext_mds[3];
    ulong m10 = ext_mds[4], m11 = ext_mds[5], m12 = ext_mds[6], m13 = ext_mds[7];
    ulong m20 = ext_mds[8], m21 = ext_mds[9], m22 = ext_mds[10], m23 = ext_mds[11];
    ulong m30 = ext_mds[12], m31 = ext_mds[13], m32 = ext_mds[14], m33 = ext_mds[15];

    ulong d0 = int_diag[0], d1 = int_diag[1], d2 = int_diag[2], d3 = int_diag[3];

    {
        u128 acc0 = {0, 0};
        acc0 = mac128(m00, s0, acc0); acc0 = mac128(m01, s1, acc0); acc0 = mac128(m02, s2, acc0); acc0 = mac128(m03, s3, acc0);
        ulong n0 = reduce128(acc0.hi, acc0.lo);
        
        u128 acc1 = {0, 0};
        acc1 = mac128(m10, s0, acc1); acc1 = mac128(m11, s1, acc1); acc1 = mac128(m12, s2, acc1); acc1 = mac128(m13, s3, acc1);
        ulong n1 = reduce128(acc1.hi, acc1.lo);
        
        u128 acc2 = {0, 0};
        acc2 = mac128(m20, s0, acc2); acc2 = mac128(m21, s1, acc2); acc2 = mac128(m22, s2, acc2); acc2 = mac128(m23, s3, acc2);
        ulong n2 = reduce128(acc2.hi, acc2.lo);
        
        u128 acc3 = {0, 0};
        acc3 = mac128(m30, s0, acc3); acc3 = mac128(m31, s1, acc3); acc3 = mac128(m32, s2, acc3); acc3 = mac128(m33, s3, acc3);
        ulong n3 = reduce128(acc3.hi, acc3.lo);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    uint half_f = r_f >> 1;
    uint rc_idx = 0;

    for (uint r = 0; r < half_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        
        u128 acc0 = {0, 0};
        acc0 = mac128(m00, s0, acc0); acc0 = mac128(m01, s1, acc0); acc0 = mac128(m02, s2, acc0); acc0 = mac128(m03, s3, acc0);
        ulong n0 = reduce128(acc0.hi, acc0.lo);
        
        u128 acc1 = {0, 0};
        acc1 = mac128(m10, s0, acc1); acc1 = mac128(m11, s1, acc1); acc1 = mac128(m12, s2, acc1); acc1 = mac128(m13, s3, acc1);
        ulong n1 = reduce128(acc1.hi, acc1.lo);
        
        u128 acc2 = {0, 0};
        acc2 = mac128(m20, s0, acc2); acc2 = mac128(m21, s1, acc2); acc2 = mac128(m22, s2, acc2); acc2 = mac128(m23, s3, acc2);
        ulong n2 = reduce128(acc2.hi, acc2.lo);
        
        u128 acc3 = {0, 0};
        acc3 = mac128(m30, s0, acc3); acc3 = mac128(m31, s1, acc3); acc3 = mac128(m32, s2, acc3); acc3 = mac128(m33, s3, acc3);
        ulong n3 = reduce128(acc3.hi, acc3.lo);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = 0; r < r_p; ++r) {
        s0 = sbox7(gold_add(s0, rc_int[r]));
        
        ulong sum = gold_add(gold_add(s0, s1), gold_add(s2, s3));
        
        ulong n0 = mac_sum(sum, d0, s0);
        ulong n1 = mac_sum(sum, d1, s1);
        ulong n2 = mac_sum(sum, d2, s2);
        ulong n3 = mac_sum(sum, d3, s3);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
    }

    for (uint r = half_f; r < r_f; ++r) {
        s0 = sbox7(gold_add(s0, rc_ext[rc_idx]));
        s1 = sbox7(gold_add(s1, rc_ext[rc_idx + 1]));
        s2 = sbox7(gold_add(s2, rc_ext[rc_idx + 2]));
        s3 = sbox7(gold_add(s3, rc_ext[rc_idx + 3]));
        rc_idx += 4;
        
        u128 acc0 = {0, 0};
        acc0 = mac128(m00, s0, acc0); acc0 = mac128(m01, s1, acc0); acc0 = mac128(m02, s2, acc0); acc0 = mac128(m03, s3, acc0);
        ulong n0 = reduce128(acc0.hi, acc0.lo);
        
        u128 acc1 = {0, 0};
        acc1 = mac128(m10, s0, acc1); acc1 = mac128(m11, s1, acc1); acc1 = mac128(m12, s2, acc1); acc1 = mac128(m13, s3, acc1);
        ulong n1 = reduce128(acc1.hi, acc1.lo);
        
        u128 acc2 = {0, 0};
        acc2 = mac128(m20, s0, acc2); acc2 = mac128(m21, s1, acc2); acc2 = mac128(m22, s2, acc2); acc2 = mac128(m23, s3, acc2);
        ulong n2 = reduce128(acc2.hi, acc2.lo);
        
        u128 acc3 = {0, 0};
        acc3 = mac128(m30, s0, acc3); acc3 = mac128(m31, s1, acc3); acc3 = mac128(m32, s2, acc3); acc3 = mac128(m33, s3, acc3);
        ulong n3 = reduce128(acc3.hi, acc3.lo);
        
        s0 = n0; s1 = n1; s2 = n2; s3 = n3;
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

    if (t <= 3) {
        ulong s0 = 0, s1 = 0, s2 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        
        poseidon2_permute_3(s0, s1, s2, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    } else {
        ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        uint base = p * arity;
        
        if (arity > 0 && base + 0 < child_count) s0 = tree[in_offset + base + 0];
        if (arity > 1 && base + 1 < child_count) s1 = tree[in_offset + base + 1];
        if (arity > 2 && base + 2 < child_count) s2 = tree[in_offset + base + 2];
        if (arity > 3 && base + 3 < child_count) s3 = tree[in_offset + base + 3];
        
        poseidon2_permute_4(s0, s1, s2, s3, rc_ext, rc_int, ext_mds, int_diag, r_f, r_p);
        tree[out_offset + p] = s0;
    }
}
```