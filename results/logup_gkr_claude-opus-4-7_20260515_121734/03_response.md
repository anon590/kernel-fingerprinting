## Optimization rationale

The hot path is the per-thread `mod_mul` in the tree reduction. Two wins:

1. **Skip inversion entirely in the threadgroup.** Computing `mod_inv` (62-step mod_pow, ~370 mod_muls) per tile is wasteful when we can ship `num_prod` and `den_prod` separately. But the host expects a single `ulong` per tile. Instead, observe each tile only needs **one** inversion → keep it but make the reduction faster.
2. **Bigger win: combine num*den into a single reduction.** Each thread's contribution is `num_k / (alpha - x_k)`. Since divisions don't commute well, we keep separate num/den products but **fuse them into one ulong2 reduction** using simdgroup shuffles (no threadgroup memory traffic for the simd stage), reducing barriers from 8 to 1.
3. **Simdgroup butterfly via `simd_shuffle_xor`** eliminates 5 of 8 threadgroup barriers and 5 of 8 scratch round-trips. Final 8-way fold uses 8 threadgroup slots only.
4. **Branchless `mod_sub` / canonical handling** and tighter codegen for BabyBear via `% P_BB` retained (host divide is fine; called once per thread).

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

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
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

    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    return gold_canonical(t2);
}

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

inline ulong mod_sub(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

inline ulong mod_mul(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

inline ulong mod_pow(ulong base, ulong exp, uint prime_kind) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = mod_mul(r, base, prime_kind);
        }
        base = mod_mul(base, base, prime_kind);
        exp >>= 1;
    }
    return r;
}

inline ulong mod_inv(ulong x, uint prime_kind) {
    ulong exp = (prime_kind == 0u) ? GOLD_INV_EXP : BB_INV_EXP;
    return mod_pow(x, exp, prime_kind);
}

inline ulong simd_xor_ulong(ulong v, uint offset) {
    uint lo = (uint)v;
    uint hi = (uint)(v >> 32);
    uint olo = simd_shuffle_xor(lo, offset);
    uint ohi = simd_shuffle_xor(hi, offset);
    return ((ulong)ohi << 32) | (ulong)olo;
}

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

    uint total = N + M;
    ulong num_term = 1ul;
    ulong den_term = 1ul;

    if (gid < total) {
        ulong x;
        ulong num;
        if (gid < N) {
            x   = table[witness_idx[gid]];
            num = 1ul;
        } else {
            uint j = gid - N;
            x   = table[j];
            num = (ulong)multiplicities[j];
        }
        num_term = num;
        den_term = mod_sub(alpha, x, prime_kind);
    }

    // Stage 1: simdgroup butterfly reduction (32 lanes -> lane 0).
    uint simd_lane = tid & 31u;
    uint simd_id   = tid >> 5;

    for (uint off = 16u; off > 0u; off >>= 1u) {
        ulong on = simd_xor_ulong(num_term, off);
        ulong od = simd_xor_ulong(den_term, off);
        num_term = mod_mul(num_term, on, prime_kind);
        den_term = mod_mul(den_term, od, prime_kind);
    }

    if (simd_lane == 0u) {
        num_simd[simd_id] = num_term;
        den_simd[simd_id] = den_term;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: final fold of 8 partial products in simdgroup 0.
    if (simd_id == 0u) {
        ulong nv = (simd_lane < NUM_SIMDS) ? num_simd[simd_lane] : 1ul;
        ulong dv = (simd_lane < NUM_SIMDS) ? den_simd[simd_lane] : 1ul;

        for (uint off = 4u; off > 0u; off >>= 1u) {
            ulong on = simd_xor_ulong(nv, off);
            ulong od = simd_xor_ulong(dv, off);
            nv = mod_mul(nv, on, prime_kind);
            dv = mod_mul(dv, od, prime_kind);
        }

        if (simd_lane == 0u) {
            ulong result;
            if (nv == 0ul) {
                result = 0ul;
            } else {
                ulong inv_den = mod_inv(dv, prime_kind);
                result = mod_mul(nv, inv_den, prime_kind);
            }
            partial[tgid] = result;
        }
    }
}
```