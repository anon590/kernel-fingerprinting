To optimize the carry-less multiplication on Apple Silicon, we make two key design shifts:

1. **Karatsuba-based `clmul64`**: The previous 4-bit window implementation suffered from heavy loop-carried dependencies due to simulating 128-bit shifts across limbs. By applying Karatsuba to split `clmul64` into three independent `clmul32` multiplications, we eliminate the 128-bit shift simulation entirely.
2. **ILP-Friendly `clmul32`**: Each 32-bit carry-less multiply is written as a fully unrolled loop of independent constant shifts and selects. This removes data dependencies between steps, enabling the GPU to maximize Instruction-Level Parallelism (ILP).
3. **Streamlined `gcm_reduce`**: We simplified the AES-GCM reduction by grouping shift-and-XOR operations into clean, reusable `fold_low` and `fold_high` helpers, reducing total instruction count.

These enhancements yield a highly parallel, low-latency implementation optimized for Apple's wide execution pipelines.

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
    r_lo = t0 ^ fold_low(t2) ^ fold_low(fold_high(t3));
    r_hi = t1 ^ fold_low(t3) ^ fold_high(t2);
}

inline ulong clmul32(uint A, uint B)
{
    ulong accum = 0;
    ulong a = A;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        accum ^= ((B & (1u << j)) ? (a << j) : 0ul);
    }
    return accum;
}

inline ulong2 clmul64(ulong A, ulong B)
{
    uint a_lo = (uint)A;
    uint a_hi = (uint)(A >> 32);
    uint b_lo = (uint)B;
    uint b_hi = (uint)(B >> 32);

    ulong low = clmul32(a_lo, b_lo);
    ulong high = clmul32(a_hi, b_hi);
    ulong mid = clmul32(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    ulong r_lo = low ^ (mid << 32);
    ulong r_hi = high ^ (mid >> 32);
    return ulong2(r_lo, r_hi);
}

inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong2 low = clmul64(a_lo, b_lo);
    ulong2 high = clmul64(a_hi, b_hi);
    ulong2 mid = clmul64(a_lo ^ a_hi, b_lo ^ b_hi);

    mid ^= low ^ high;

    t0 = low.x;
    t1 = low.y ^ mid.x;
    t2 = high.x ^ mid.y;
    t3 = high.y;
}

inline void gf128_mul(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &c_lo, thread ulong &c_hi)
{
    ulong t0, t1, t2, t3;
    clmul128_unreduced(a_lo, a_hi, b_lo, b_hi, t0, t1, t2, t3);
    gcm_reduce(t0, t1, t2, t3, c_lo, c_hi);
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

        ulong c_lo, c_hi;
        gf128_mul(va.x, va.y, vb.x, vb.y, c_lo, c_hi);

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

        ulong m00_lo, m00_hi;
        gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);

        ulong m11_lo, m11_hi;
        gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);

        ulong madd_lo, madd_hi;
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi, b0_lo ^ b1_lo, b0_hi ^ b1_hi, madd_lo, madd_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = madd_lo ^ m00_lo;
        ulong c1_hi = madd_hi ^ m00_hi;

        c4[idx] = ulong4(c0_lo, c0_hi, c1_lo, c1_hi);
    }
}
```