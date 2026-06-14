**Optimization strategy:** The incumbent does one butterfly per dispatch with full global memory round-trips. Since the host fixes dispatch geometry to (N/2, 1, 1) and I cannot change it, I'll focus on making each butterfly as cheap as possible. Key optimizations:

1. **Faster 128-bit multiplication** using Metal's native `mulhi` intrinsic which compiles to a single hardware instruction on Apple GPUs, instead of the 4-way 32x32 split.
2. **Branchless arithmetic** using `select` to avoid divergence.
3. **Streamlined reduction** — fold the high-half decomposition more tightly.
4. **Hoist canonicalization** — the reduction already produces canonical output, so `gold_add`/`gold_sub` operate on canonical inputs and need minimal correction.

The `mulhi` intrinsic is the big win: on Apple Silicon, `mulhi((uint)a, (uint)b)` produces the high 32 bits of a 32×32 product in one instruction, and we can combine four such products plus the low-mul (just `a*b` for the low 64 bits) much more efficiently than four explicit `(ulong)u32*(ulong)u32` multiplies.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;  // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;  // 2^32 - 1

inline ulong gold_canonical(ulong x) {
    return select(x, x - P_GOLD, x >= P_GOLD);
}

inline ulong gold_add(ulong a, ulong b) {
    ulong t = a + b;
    t = select(t, t + EPSILON, t < a);
    return select(t, t - P_GOLD, t >= P_GOLD);
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    return select(t, t - EPSILON, t > a);
}

// 64x64 -> 128 multiply using Metal's native 32x32->high intrinsic.
// Low 64 bits come from a plain a*b. High 64 bits computed via mulhi
// on 32-bit halves plus cross products.
inline ulong umul_hi(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
    uint b1 = (uint)(b >> 32);

    // p00_hi = high 32 bits of (a0 * b0)
    uint  p00_hi = mulhi(a0, b0);
    ulong p01    = (ulong)a0 * (ulong)b1;   // 64-bit
    ulong p10    = (ulong)a1 * (ulong)b0;   // 64-bit
    ulong p11    = (ulong)a1 * (ulong)b1;   // 64-bit

    // mid = p00_hi + low32(p01) + low32(p10)  (fits in 34 bits)
    ulong mid = (ulong)p00_hi + (p01 & EPSILON) + (p10 & EPSILON);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return hi;
}

// Goldilocks multiplication: 64x64 -> reduce mod p = 2^64 - 2^32 + 1.
// Uses 2^64 ≡ 2^32 - 1 (mod p), i.e.
//   hi*2^64 + lo ≡ lo + hi_lo*(2^32 - 1) - hi_hi (mod p)
inline ulong gold_mul(ulong a, ulong b) {
    ulong lo = a * b;            // implicit low 64 bits
    ulong hi = umul_hi(a, b);

    ulong hi_lo = hi & EPSILON;
    ulong hi_hi = hi >> 32;

    // t0 = lo - hi_hi (mod 2^64), correct underflow by subtracting EPSILON
    ulong t0 = lo - hi_hi;
    t0 = select(t0, t0 - EPSILON, t0 > lo);

    // t1 = hi_lo * (2^32 - 1) = (hi_lo << 32) - hi_lo, fits in u64
    ulong t1 = (hi_lo << 32) - hi_lo;

    // t2 = t0 + t1, correct overflow by adding EPSILON
    ulong t2 = t0 + t1;
    t2 = select(t2, t2 + EPSILON, t2 < t0);

    return gold_canonical(t2);
}

kernel void goldilocks_ntt_stage(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    device const ulong *twiddles   [[buffer(2)]],
    constant uint      &stage_idx  [[buffer(3)]],
    constant uint      &log_N      [[buffer(4)]],
    uint k [[thread_position_in_grid]])
{
    uint lN     = log_N;
    uint half_N = 1u << (lN - 1u);
    if (k >= half_N) return;

    uint s         = stage_idx;
    uint span_half = 1u << s;            // 2^s
    uint mask      = span_half - 1u;

    uint r  = k & mask;
    uint j  = k >> s;
    uint jh = j << s;                    // j * 2^s

    uint i0 = jh + r;
    uint i1 = i0 + half_N;
    uint o0 = (jh << 1) + r;             // j * 2^(s+1) + r
    uint o1 = o0 + span_half;

    uint tw_shift = lN - 1u - s;
    ulong w = twiddles[(ulong)r << tw_shift];

    ulong u = in_data[i0];
    ulong v = gold_mul(in_data[i1], w);

    out_data[o0] = gold_add(u, v);
    out_data[o1] = gold_sub(u, v);
}
```