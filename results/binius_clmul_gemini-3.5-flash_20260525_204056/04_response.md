To achieve a significant performance improvement, we apply several key optimizations:

1. **2-bit Windowed `clmul32`**: We transition from 1-bit-at-a-time loop to a 2-bit window. Precomputing 3 registers of 32-bit multiples of $A$ allows us to process 2 bits of $B$ per iteration using 3 fast, branchless `select` instructions, halving loop iterations to 16 and drastically reducing instruction count.
2. **Inlined Karatsuba & Direct Unpacking**: We flattened `clmul64` directly into `clmul128_unreduced`. This eliminates the overhead of temporary `ulong2` structs and exposes all 9 `clmul32` calls directly to the compiler's register allocator and scheduler.
3. **Deferred GCM Reduction in Tower 1**: Since GCM reduction is linear, we defer it on the Karatsuba components. This reduces the number of expensive `gcm_reduce` calls per thread from 4 to 3.
4. **Optimized `gcm_reduce`**: Utilizing the linearity of `fold_low`, we combine `fold_low(t2) ^ fold_low(fold_high(t3))` into `fold_low(t2 ^ fold_high(t3))`, saving a whole `fold_low` call.

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong fold_low(ulong x) {
    return x ^ (x << 1u) ^ (x << 2u) ^ (x << 7u);
}

inline ulong fold_high(ulong x) {
    return (x >> 63u) ^ (x >> 62u) ^ (x >> 57u);
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong h3 = fold_high(t3);
    r_lo = t0 ^ fold_low(t2 ^ h3);
    r_hi = t1 ^ fold_low(t3) ^ fold_high(t2);
}

inline ulong clmul32(uint A, uint B)
{
    ulong accum = 0;
    ulong a1 = A;
    ulong a2 = a1 << 1;
    ulong a3 = a2 ^ a1;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint chunk = (B >> (2 * i)) & 3u;
        bool b0 = (chunk & 1u) != 0;
        bool b1 = (chunk & 2u) != 0;
        ulong res = select(select(0ul, a1, b0), select(a2, a3, b0), b1);
        accum ^= (res << (2 * i));
    }
    return accum;
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    uint a00 = (uint)a_lo;
    uint a01 = (uint)(a_lo >> 32);
    uint a10 = (uint)a_hi;
    uint a11 = (uint)(a_hi >> 32);

    uint b00 = (uint)b_lo;
    uint b01 = (uint)(b_lo >> 32);
    uint b10 = (uint)b_hi;
    uint b11 = (uint)(b_hi >> 32);

    ulong l0 = clmul32(a00, b00);
    ulong h0 = clmul32(a01, b01);
    ulong m0 = clmul32(a00 ^ a01, b00 ^ b01) ^ l0 ^ h0;
    ulong low_x = l0 ^ (m0 << 32);
    ulong low_y = h0 ^ (m0 >> 32);

    ulong l2 = clmul32(a10, b10);
    ulong h2 = clmul32(a11, b11);
    ulong m2 = clmul32(a10 ^ a11, b10 ^ b11) ^ l2 ^ h2;
    ulong high_x = l2 ^ (m2 << 32);
    ulong high_y = h2 ^ (m2 >> 32);

    ulong l1 = clmul32(a00 ^ a10, b00 ^ b10);
    ulong h1 = clmul32(a01 ^ a11, b01 ^ b11);
    ulong m1 = clmul32((a00 ^ a10) ^ (a01 ^ a11), (b00 ^ b10) ^ (b01 ^ b11)) ^ l1 ^ h1;
    ulong mid_x = l1 ^ (m1 << 32) ^ low_x ^ high_x;
    ulong mid_y = h1 ^ (m1 >> 32) ^ low_y ^ high_y;

    t0 = low_x;
    t1 = low_y ^ mid_x;
    t2 = high_x ^ mid_y;
    t3 = high_y;
}

kernel void binius_clmul(
    device const ulong *a         [[buffer(0)]],
    device const ulong *b         [[buffer(1)]],
    device       ulong *c         [[buffer(2)]],
    constant ulong     &alpha_lo  [[buffer(3)]],
    constant ulong     &alpha_hi  [[buffer(4)]],
    constant uint      &tower     [[buffer(5)]],
    constant uint      &batch     [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    if (tower == 0u) {
        device const ulong2 *a2 = (device const ulong2 *)a;
        device const ulong2 *b2 = (device const ulong2 *)b;
        device       ulong2 *c2 = (device       ulong2 *)c;

        ulong2 va = a2[idx];
        ulong2 vb = b2[idx];

        ulong t0, t1, t2, t3;
        clmul128_unreduced(va.x, va.y, vb.x, vb.y, t0, t1, t2, t3);

        ulong c_lo, c_hi;
        gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);

        c2[idx] = ulong2(c_lo, c_hi);
    } else {
        device const ulong4 *a4 = (device const ulong4 *)a;
        device const ulong4 *b4 = (device const ulong4 *)b;
        device       ulong4 *c4 = (device       ulong4 *)c;

        ulong4 va = a4[idx];
        ulong4 vb = b4[idx];

        ulong a0_lo = va.x, a0_hi = va.y;
        ulong a1_lo = va.z, a1_hi = va.w;
        ulong b0_lo = vb.x, b0_hi = vb.y;
        ulong b1_lo = vb.z, b1_hi = vb.w;

        ulong t00_0, t00_1, t00_2, t00_3;
        clmul128_unreduced(a0_lo, a0_hi, b0_lo, b0_hi, t00_0, t00_1, t00_2, t00_3);

        ulong t11_0, t11_1, t11_2, t11_3;
        clmul128_unreduced(a1_lo, a1_hi, b1_lo, b1_hi, t11_0, t11_1, t11_2, t11_3);

        ulong m11_lo, m11_hi;
        gcm_reduce(t11_0, t11_1, t11_2, t11_3, m11_lo, m11_hi);

        ulong tam_0, tam_1, tam_2, tam_3;
        clmul128_unreduced(alpha_lo, alpha_hi, m11_lo, m11_hi, tam_0, tam_1, tam_2, tam_3);

        ulong tadd_0, tadd_1, tadd_2, tadd_3;
        clmul128_unreduced(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, tadd_0, tadd_1, tadd_2, tadd_3);

        ulong tc0_0 = t00_0 ^ tam_0;
        ulong tc0_1 = t00_1 ^ tam_1;
        ulong tc0_2 = t00_2 ^ tam_2;
        ulong tc0_3 = t00_3 ^ tam_3;

        ulong tc1_0 = tadd_0 ^ t00_0;
        ulong tc1_1 = tadd_1 ^ t00_1;
        ulong tc1_2 = tadd_2 ^ t00_2;
        ulong tc1_3 = tadd_3 ^ t00_3;

        ulong c0_lo, c0_hi;
        gcm_reduce(tc0_0, tc0_1, tc0_2, tc0_3, c0_lo, c0_hi);

        ulong c1_lo, c1_hi;
        gcm_reduce(tc1_0, tc1_1, tc1_2, tc1_3, c1_lo, c1_hi);

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}
```