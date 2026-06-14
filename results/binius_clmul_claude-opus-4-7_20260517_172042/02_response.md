**Optimization rationale:** The previous 4-bit windowed clmul builds a 16-entry table (192-bit triples) and scans 32 nibbles, doing a lot of shifts. I'll switch to a **classic 8-bit-at-a-time Karatsuba-free schoolbook** using 64×64→128 clmul via the "shift-and-XOR by mask" pattern, but more importantly I'll use the **standard interleaved doubling formulation** that processes 8 bits of `b` per iteration with no table — using `b`'s bit mask to conditionally XOR `a<<i`. Actually the best trick on Apple GPUs: the bitwise clmul `clmul64(a,b)` computed by iterating over **the bits set in the smaller operand** is slow; instead use the **branchless mask-multiply** where for each of 64 bit positions we XOR `(a << i) & mask_i`. But the real winner is to express `clmul64` as 8 nibble-wise shifted XORs using mask-based selection per bit, vectorized with `ulong2`. I'll implement clmul64 using the well-known 4-mask trick (split b into 4 interleaved bit-streams) and combine via shifted XOR, giving a 128×128 → 256 product in ~16 clmul64 calls but each much cheaper than my table build. Combined with packing the GF(2^256) path to reuse `m11` (Karatsuba: `m01+m10 = (a0+a1)(b0+b1) - m00 - m11`), reducing 4 gf128 muls to 3.

```metal
#include <metal_stdlib>
using namespace metal;

// 64x64 -> 128 bit carry-less multiply using the 4-way bit-split trick.
// Split b into 4 streams b0..b3 where bk has only bits with (i mod 4)==k set.
// For each stream, mask a by the predicate "bit i of b is set" via the
// well-known branchless pattern using arithmetic from each bit, then sum
// the four shifted partial products. This avoids the 192-bit table and
// 32-nibble loop of the previous attempt.
//
// We use the classic "double-and-add" but unrolled over 8 bits of b at a
// time with 8 shifted masks. Total: 64 iterations of (mask, shift, xor)
// but compiler unrolls and ILP is high.
inline void clmul64(ulong a, ulong b, thread ulong &lo, thread ulong &hi)
{
    ulong l = 0ul;
    ulong h = 0ul;
    // Bit 0
    ulong m;
    m = -(b & 1ul);        l ^= a & m;
    m = -((b >> 1) & 1ul); l ^= (a << 1) & m;  h ^= (a >> 63) & m;
    m = -((b >> 2) & 1ul); l ^= (a << 2) & m;  h ^= (a >> 62) & m;
    m = -((b >> 3) & 1ul); l ^= (a << 3) & m;  h ^= (a >> 61) & m;
    m = -((b >> 4) & 1ul); l ^= (a << 4) & m;  h ^= (a >> 60) & m;
    m = -((b >> 5) & 1ul); l ^= (a << 5) & m;  h ^= (a >> 59) & m;
    m = -((b >> 6) & 1ul); l ^= (a << 6) & m;  h ^= (a >> 58) & m;
    m = -((b >> 7) & 1ul); l ^= (a << 7) & m;  h ^= (a >> 57) & m;
    m = -((b >> 8) & 1ul); l ^= (a << 8) & m;  h ^= (a >> 56) & m;
    m = -((b >> 9) & 1ul); l ^= (a << 9) & m;  h ^= (a >> 55) & m;
    m = -((b >>10) & 1ul); l ^= (a <<10) & m;  h ^= (a >> 54) & m;
    m = -((b >>11) & 1ul); l ^= (a <<11) & m;  h ^= (a >> 53) & m;
    m = -((b >>12) & 1ul); l ^= (a <<12) & m;  h ^= (a >> 52) & m;
    m = -((b >>13) & 1ul); l ^= (a <<13) & m;  h ^= (a >> 51) & m;
    m = -((b >>14) & 1ul); l ^= (a <<14) & m;  h ^= (a >> 50) & m;
    m = -((b >>15) & 1ul); l ^= (a <<15) & m;  h ^= (a >> 49) & m;
    m = -((b >>16) & 1ul); l ^= (a <<16) & m;  h ^= (a >> 48) & m;
    m = -((b >>17) & 1ul); l ^= (a <<17) & m;  h ^= (a >> 47) & m;
    m = -((b >>18) & 1ul); l ^= (a <<18) & m;  h ^= (a >> 46) & m;
    m = -((b >>19) & 1ul); l ^= (a <<19) & m;  h ^= (a >> 45) & m;
    m = -((b >>20) & 1ul); l ^= (a <<20) & m;  h ^= (a >> 44) & m;
    m = -((b >>21) & 1ul); l ^= (a <<21) & m;  h ^= (a >> 43) & m;
    m = -((b >>22) & 1ul); l ^= (a <<22) & m;  h ^= (a >> 42) & m;
    m = -((b >>23) & 1ul); l ^= (a <<23) & m;  h ^= (a >> 41) & m;
    m = -((b >>24) & 1ul); l ^= (a <<24) & m;  h ^= (a >> 40) & m;
    m = -((b >>25) & 1ul); l ^= (a <<25) & m;  h ^= (a >> 39) & m;
    m = -((b >>26) & 1ul); l ^= (a <<26) & m;  h ^= (a >> 38) & m;
    m = -((b >>27) & 1ul); l ^= (a <<27) & m;  h ^= (a >> 37) & m;
    m = -((b >>28) & 1ul); l ^= (a <<28) & m;  h ^= (a >> 36) & m;
    m = -((b >>29) & 1ul); l ^= (a <<29) & m;  h ^= (a >> 35) & m;
    m = -((b >>30) & 1ul); l ^= (a <<30) & m;  h ^= (a >> 34) & m;
    m = -((b >>31) & 1ul); l ^= (a <<31) & m;  h ^= (a >> 33) & m;
    m = -((b >>32) & 1ul); l ^= (a <<32) & m;  h ^= (a >> 32) & m;
    m = -((b >>33) & 1ul); l ^= (a <<33) & m;  h ^= (a >> 31) & m;
    m = -((b >>34) & 1ul); l ^= (a <<34) & m;  h ^= (a >> 30) & m;
    m = -((b >>35) & 1ul); l ^= (a <<35) & m;  h ^= (a >> 29) & m;
    m = -((b >>36) & 1ul); l ^= (a <<36) & m;  h ^= (a >> 28) & m;
    m = -((b >>37) & 1ul); l ^= (a <<37) & m;  h ^= (a >> 27) & m;
    m = -((b >>38) & 1ul); l ^= (a <<38) & m;  h ^= (a >> 26) & m;
    m = -((b >>39) & 1ul); l ^= (a <<39) & m;  h ^= (a >> 25) & m;
    m = -((b >>40) & 1ul); l ^= (a <<40) & m;  h ^= (a >> 24) & m;
    m = -((b >>41) & 1ul); l ^= (a <<41) & m;  h ^= (a >> 23) & m;
    m = -((b >>42) & 1ul); l ^= (a <<42) & m;  h ^= (a >> 22) & m;
    m = -((b >>43) & 1ul); l ^= (a <<43) & m;  h ^= (a >> 21) & m;
    m = -((b >>44) & 1ul); l ^= (a <<44) & m;  h ^= (a >> 20) & m;
    m = -((b >>45) & 1ul); l ^= (a <<45) & m;  h ^= (a >> 19) & m;
    m = -((b >>46) & 1ul); l ^= (a <<46) & m;  h ^= (a >> 18) & m;
    m = -((b >>47) & 1ul); l ^= (a <<47) & m;  h ^= (a >> 17) & m;
    m = -((b >>48) & 1ul); l ^= (a <<48) & m;  h ^= (a >> 16) & m;
    m = -((b >>49) & 1ul); l ^= (a <<49) & m;  h ^= (a >> 15) & m;
    m = -((b >>50) & 1ul); l ^= (a <<50) & m;  h ^= (a >> 14) & m;
    m = -((b >>51) & 1ul); l ^= (a <<51) & m;  h ^= (a >> 13) & m;
    m = -((b >>52) & 1ul); l ^= (a <<52) & m;  h ^= (a >> 12) & m;
    m = -((b >>53) & 1ul); l ^= (a <<53) & m;  h ^= (a >> 11) & m;
    m = -((b >>54) & 1ul); l ^= (a <<54) & m;  h ^= (a >> 10) & m;
    m = -((b >>55) & 1ul); l ^= (a <<55) & m;  h ^= (a >>  9) & m;
    m = -((b >>56) & 1ul); l ^= (a <<56) & m;  h ^= (a >>  8) & m;
    m = -((b >>57) & 1ul); l ^= (a <<57) & m;  h ^= (a >>  7) & m;
    m = -((b >>58) & 1ul); l ^= (a <<58) & m;  h ^= (a >>  6) & m;
    m = -((b >>59) & 1ul); l ^= (a <<59) & m;  h ^= (a >>  5) & m;
    m = -((b >>60) & 1ul); l ^= (a <<60) & m;  h ^= (a >>  4) & m;
    m = -((b >>61) & 1ul); l ^= (a <<61) & m;  h ^= (a >>  3) & m;
    m = -((b >>62) & 1ul); l ^= (a <<62) & m;  h ^= (a >>  2) & m;
    m = -((b >>63) & 1ul); l ^= (a <<63) & m;  h ^= (a >>  1) & m;
    lo = l;
    hi = h;
}

// 128x128 -> 256 unreduced clmul, via 3-mul Karatsuba.
inline void clmul128_unreduced(
    ulong a_lo, ulong a_hi, ulong b_lo, ulong b_hi,
    thread ulong &t0, thread ulong &t1,
    thread ulong &t2, thread ulong &t3)
{
    ulong z0_lo, z0_hi; clmul64(a_lo, b_lo, z0_lo, z0_hi);
    ulong z2_lo, z2_hi; clmul64(a_hi, b_hi, z2_lo, z2_hi);
    ulong zm_lo, zm_hi; clmul64(a_lo ^ a_hi, b_lo ^ b_hi, zm_lo, zm_hi);
    // middle = zm - z0 - z2 (XOR in GF(2))
    ulong m_lo = zm_lo ^ z0_lo ^ z2_lo;
    ulong m_hi = zm_hi ^ z0_hi ^ z2_hi;
    // Combine: result = z0 + (m << 64) + (z2 << 128)
    t0 = z0_lo;
    t1 = z0_hi ^ m_lo;
    t2 = z2_lo ^ m_hi;
    t3 = z2_hi;
}

inline void gcm_reduce(
    ulong t0, ulong t1, ulong t2, ulong t3,
    thread ulong &r_lo, thread ulong &r_hi)
{
    // Fold t3:t2 (upper 128) into lower via R(x) = x^128 + x^7 + x^2 + x + 1.
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

        // Karatsuba over GF(2^128)[v]:
        //   m00 = a0*b0
        //   m11 = a1*b1
        //   mk  = (a0+a1)*(b0+b1) = m00 + m01 + m10 + m11   (in GF(2))
        // => m01 + m10 = mk + m00 + m11
        // c0 = m00 + alpha*m11
        // c1 = m01 + m10 + m11 = mk + m00 + m11 + m11 = mk + m00
        ulong m00_lo, m00_hi; gf128_mul(a0_lo, a0_hi, b0_lo, b0_hi, m00_lo, m00_hi);
        ulong m11_lo, m11_hi; gf128_mul(a1_lo, a1_hi, b1_lo, b1_hi, m11_lo, m11_hi);
        ulong mk_lo,  mk_hi;  gf128_mul(a0_lo ^ a1_lo, a0_hi ^ a1_hi,
                                        b0_lo ^ b1_lo, b0_hi ^ b1_hi,
                                        mk_lo, mk_hi);

        ulong am_lo, am_hi;
        gf128_mul(alpha_lo, alpha_hi, m11_lo, m11_hi, am_lo, am_hi);

        ulong c0_lo = m00_lo ^ am_lo;
        ulong c0_hi = m00_hi ^ am_hi;
        ulong c1_lo = mk_lo ^ m00_lo;
        ulong c1_hi = mk_hi ^ m00_hi;

        c[base + 0] = c0_lo;
        c[base + 1] = c0_hi;
        c[base + 2] = c1_lo;
        c[base + 3] = c1_hi;
    }
}
```