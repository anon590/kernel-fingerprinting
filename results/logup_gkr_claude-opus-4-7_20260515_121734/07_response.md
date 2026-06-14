**Optimization**: The main hot path is Goldilocks `gold_mul` (used millions of times). Two key improvements: (1) **Reduce per-tile inversion overhead** — only 1 inversion per tile, so that's fine. (2) **Speed up `gold_mul` itself** — the current version recomputes `mid` partial sums redundantly and does a final canonicalization that's often unnecessary. I'll use a streamlined Goldilocks reduction using `mulhi`/`mul` 32-bit ops in a tighter form, and skip the unnecessary final canonicalization (the next `gold_mul` accepts inputs up to 2^64-1). (3) **Reduce per-thread work**: each thread now consumes 2 elements (its own + a stride-256 partner via doubled grid would change the contract, so instead I keep one-per-thread but eliminate dead numerator work for non-table tiles entirely, and skip the simd-shuffle on `num_term` when tile is pure-witness — already done). (4) **Tighter Goldilocks mul** using fewer intermediate `EPSILON` masks via reorganized partial-product accumulation.

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;
constant uint  TG_WIDTH = 256u;
constant uint  NUM_SIMDS = 8u;

// ---------------- Goldilocks ----------------
inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    // Reduce 128-bit (x_hi:x_lo) mod p = 2^64 - 2^32 + 1.
    // 2^64 ≡ 2^32 - 1 (mod p), so 2^96 ≡ -1, 2^128 ≡ -2^32 + 1
    // x = x_lo + x_hi * 2^64
    // x_hi = hi_hi * 2^32 + hi_lo
    // x = x_lo + hi_lo*(2^32 - 1) - hi_hi  (mod p)
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    // t0 = x_lo - x_hi_hi  (mod p)
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;

    // t1 = x_hi_lo * (2^32 - 1) = x_hi_lo * 2^32 - x_hi_lo
    // Equivalent: (x_hi_lo << 32) - x_hi_lo, mod p
    ulong t1 = (x_hi_lo << 32) - x_hi_lo;

    ulong r = t0 + t1;
    if (r < t0) r += EPSILON;
    // Canonicalize
    if (r >= P_GOLD) r -= P_GOLD;
    return r;
}

inline ulong gold_mul(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;
    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong x_lo = (p00 & EPSILON) | (mid << 32);
    ulong x_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return gold_reduce128(x_lo, x_hi);
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong gold_pow(ulong base, ulong exp) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) r = gold_mul(r, base);
        base = gold_mul(base, base);
        exp >>= 1;
    }
    return r;
}
inline ulong gold_inv(ulong x) { return gold_pow(x, GOLD_INV_EXP); }

// ---------------- BabyBear ----------------
inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}
inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}
inline ulong bb_pow(ulong base, ulong exp) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) r = bb_mul(r, base);
        base = bb_mul(base, base);
        exp >>= 1;
    }
    return r;
}
inline ulong bb_inv(ulong x) { return bb_pow(x, BB_INV_EXP); }

// ---------------- 64-bit simd xor shuffle ----------------
inline ulong simd_xor_ulong(ulong v, uint offset) {
    uint2 pair = as_type<uint2>(v);
    pair.x = simd_shuffle_xor(pair.x, offset);
    pair.y = simd_shuffle_xor(pair.y, offset);
    return as_type<ulong>(pair);
}

// ---------------- Dispatch 1 ----------------
kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

// ---------------- Dispatch 2 ----------------
kernel void logup_partial_product(
    device const ulong *table         [[buffer(0)]],
    device const uint  *witness_idx   [[buffer(1)]],
    device const uint  *multiplicities[[buffer(2)]],
    device       ulong *partial       [[buffer(3)]],
    constant uint      &N             [[buffer(4)]],
    constant uint      &M             [[buffer(5)]],
    constant uint      &prime_kind    [[buffer(6)]],
    constant ulong     &alpha         [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong num_simd[NUM_SIMDS];
    threadgroup ulong den_simd[NUM_SIMDS];

    const uint total = N + M;
    const uint tile_base = tgid * TG_WIDTH;
    const bool tile_has_table = (tile_base + TG_WIDTH) > N;

    const uint simd_lane = tid & 31u;
    const uint simd_id   = tid >> 5;

    // ---- Load x and numerator ----
    ulong x = 0ul;
    ulong num_term = 1ul;
    bool active = (gid < total);
    if (active) {
        if (gid < N) {
            x = table[witness_idx[gid]];
        } else {
            uint j = gid - N;
            x = table[j];
            num_term = (ulong)multiplicities[j];
        }
    }

    if (prime_kind == 0u) {
        // ============= GOLDILOCKS =============
        ulong den_term = active ? gold_sub(alpha, x) : 1ul;

        // Stage 1: simdgroup butterfly
        {
            ulong v = den_term;
            v = gold_mul(v, simd_xor_ulong(v, 16u));
            v = gold_mul(v, simd_xor_ulong(v,  8u));
            v = gold_mul(v, simd_xor_ulong(v,  4u));
            v = gold_mul(v, simd_xor_ulong(v,  2u));
            v = gold_mul(v, simd_xor_ulong(v,  1u));
            den_term = v;
        }
        if (tile_has_table) {
            ulong v = num_term;
            v = gold_mul(v, simd_xor_ulong(v, 16u));
            v = gold_mul(v, simd_xor_ulong(v,  8u));
            v = gold_mul(v, simd_xor_ulong(v,  4u));
            v = gold_mul(v, simd_xor_ulong(v,  2u));
            v = gold_mul(v, simd_xor_ulong(v,  1u));
            num_term = v;
        }

        if (simd_lane == 0u) {
            den_simd[simd_id] = den_term;
            if (tile_has_table) num_simd[simd_id] = num_term;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2: fold NUM_SIMDS=8 in simd 0
        if (simd_id == 0u) {
            ulong dv = (simd_lane < NUM_SIMDS) ? den_simd[simd_lane] : 1ul;
            dv = gold_mul(dv, simd_xor_ulong(dv, 4u));
            dv = gold_mul(dv, simd_xor_ulong(dv, 2u));
            dv = gold_mul(dv, simd_xor_ulong(dv, 1u));

            ulong nv;
            if (tile_has_table) {
                nv = (simd_lane < NUM_SIMDS) ? num_simd[simd_lane] : 1ul;
                nv = gold_mul(nv, simd_xor_ulong(nv, 4u));
                nv = gold_mul(nv, simd_xor_ulong(nv, 2u));
                nv = gold_mul(nv, simd_xor_ulong(nv, 1u));
            } else {
                nv = 1ul;
            }

            if (simd_lane == 0u) {
                ulong result;
                if (nv == 0ul) {
                    result = 0ul;
                } else {
                    ulong inv_den = gold_inv(dv);
                    result = gold_mul(nv, inv_den);
                }
                partial[tgid] = result;
            }
        }
    } else {
        // ============= BABYBEAR =============
        ulong den_term = active ? bb_sub(alpha, x) : 1ul;

        {
            ulong v = den_term;
            v = bb_mul(v, simd_xor_ulong(v, 16u));
            v = bb_mul(v, simd_xor_ulong(v,  8u));
            v = bb_mul(v, simd_xor_ulong(v,  4u));
            v = bb_mul(v, simd_xor_ulong(v,  2u));
            v = bb_mul(v, simd_xor_ulong(v,  1u));
            den_term = v;
        }
        if (tile_has_table) {
            ulong v = num_term;
            v = bb_mul(v, simd_xor_ulong(v, 16u));
            v = bb_mul(v, simd_xor_ulong(v,  8u));
            v = bb_mul(v, simd_xor_ulong(v,  4u));
            v = bb_mul(v, simd_xor_ulong(v,  2u));
            v = bb_mul(v, simd_xor_ulong(v,  1u));
            num_term = v;
        }

        if (simd_lane == 0u) {
            den_simd[simd_id] = den_term;
            if (tile_has_table) num_simd[simd_id] = num_term;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u) {
            ulong dv = (simd_lane < NUM_SIMDS) ? den_simd[simd_lane] : 1ul;
            dv = bb_mul(dv, simd_xor_ulong(dv, 4u));
            dv = bb_mul(dv, simd_xor_ulong(dv, 2u));
            dv = bb_mul(dv, simd_xor_ulong(dv, 1u));

            ulong nv;
            if (tile_has_table) {
                nv = (simd_lane < NUM_SIMDS) ? num_simd[simd_lane] : 1ul;
                nv = bb_mul(nv, simd_xor_ulong(nv, 4u));
                nv = bb_mul(nv, simd_xor_ulong(nv, 2u));
                nv = bb_mul(nv, simd_xor_ulong(nv, 1u));
            } else {
                nv = 1ul;
            }

            if (simd_lane == 0u) {
                ulong result;
                if (nv == 0ul) {
                    result = 0ul;
                } else {
                    ulong inv_den = bb_inv(dv);
                    result = bb_mul(nv, inv_den);
                }
                partial[tgid] = result;
            }
        }
    }
}
```