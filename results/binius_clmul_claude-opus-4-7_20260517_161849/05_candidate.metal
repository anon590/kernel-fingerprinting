#include <metal_stdlib>
using namespace metal;

// Bit-sliced carry-less multiply 64x64 -> 128.
// For each bit j of b: if bit set, XOR (a << j) into (lo, hi).
// Implemented branchlessly with a sign-mask so the compiler emits AND+XOR.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    ulong rl = 0ul;
    ulong rh = 0ul;

    // j = 0
    {
        ulong m = (ulong)0 - (b & 1ul);
        rl ^= a & m;
    }
    // j = 1..63
    ulong al = a;
    ulong ah = 0ul;
    // Unrolled loop
    #define STEP(j) do {                                                   \
        ulong _al = a << (j);                                              \
        ulong _ah = a >> (64u - (j));                                      \
        ulong _m  = (ulong)0 - ((b >> (j)) & 1ul);                         \
        rl ^= _al & _m;                                                    \
        rh ^= _ah & _m;                                                    \
    } while(0)

    STEP(1);  STEP(2);  STEP(3);  STEP(4);  STEP(5);  STEP(6);  STEP(7);  STEP(8);
    STEP(9);  STEP(10); STEP(11); STEP(12); STEP(13); STEP(14); STEP(15); STEP(16);
    STEP(17); STEP(18); STEP(19); STEP(20); STEP(21); STEP(22); STEP(23); STEP(24);
    STEP(25); STEP(26); STEP(27); STEP(28); STEP(29); STEP(30); STEP(31); STEP(32);
    STEP(33); STEP(34); STEP(35); STEP(36); STEP(37); STEP(38); STEP(39); STEP(40);
    STEP(41); STEP(42); STEP(43); STEP(44); STEP(45); STEP(46); STEP(47); STEP(48);
    STEP(49); STEP(50); STEP(51); STEP(52); STEP(53); STEP(54); STEP(55); STEP(56);
    STEP(57); STEP(58); STEP(59); STEP(60); STEP(61); STEP(62); STEP(63);

    #undef STEP

    (void)al; (void)ah;
    lo = rl;
    hi = rh;
}

// 128 x 128 -> 256 carry-less multiply via Karatsuba on 64-bit halves.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong p0_lo, p0_hi;
    ulong p1_lo, p1_hi;
    ulong pm_lo, pm_hi;
    clmul64(a_lo, b_lo, p0_lo, p0_hi);
    clmul64(a_hi, b_hi, p1_lo, p1_hi);
    clmul64(a_lo ^ a_hi, b_lo ^ b_hi, pm_lo, pm_hi);

    ulong mid_lo = pm_lo ^ p0_lo ^ p1_lo;
    ulong mid_hi = pm_hi ^ p0_hi ^ p1_hi;

    t0 = p0_lo;
    t1 = p0_hi ^ mid_lo;
    t2 = p1_lo ^ mid_hi;
    t3 = p1_hi;
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    ulong d_lo0 = t2 ^ (t2 << 1u) ^ (t2 << 2u) ^ (t2 << 7u);
    ulong d_lo1 = t3
                ^ ((t3 << 1u) | (t2 >> 63u))
                ^ ((t3 << 2u) | (t2 >> 62u))
                ^ ((t3 << 7u) | (t2 >> 57u));
    ulong d_hi  = (t3 >> 63u) ^ (t3 >> 62u) ^ (t3 >> 57u);

    t0 ^= d_lo0;
    t1 ^= d_lo1;
    t0 ^= d_hi ^ (d_hi << 1u) ^ (d_hi << 2u) ^ (d_hi << 7u);

    r_lo = t0;
    r_hi = t1;
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
        size_t base = (size_t)idx * (size_t)2;
        ulong a_lo = a[base + 0];
        ulong a_hi = a[base + 1];
        ulong b_lo = b[base + 0];
        ulong b_hi = b[base + 1];

        ulong c_lo, c_hi;
        gf128_mul(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi);

        c[base + 0] = c_lo;
        c[base + 1] = c_hi;
    } else {
        size_t base = (size_t)idx * (size_t)4;
        ulong a0_lo = a[base + 0], a0_hi = a[base + 1];
        ulong a1_lo = a[base + 2], a1_hi = a[base + 3];
        ulong b0_lo = b[base + 0], b0_hi = b[base + 1];
        ulong b1_lo = b[base + 2], b1_hi = b[base + 3];

        ulong m00_lo, m00_hi; gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong msum_lo, msum_hi;
        gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                  b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                  msum_lo, msum_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = msum_lo ^ m00_lo;
        ulong c1_hi = msum_hi ^ m00_hi;

        c[base + 0] = c0_lo;
        c[base + 1] = c0_hi;
        c[base + 2] = c1_lo;
        c[base + 3] = c1_hi;
    }
}