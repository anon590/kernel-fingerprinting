#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;
constant uint  TG_WIDTH = 256u;
constant uint  SIMD_W   = 32u;
constant uint  NUM_SIMDS = TG_WIDTH / SIMD_W; // 8

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;
    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

// BabyBear: inputs in [0, P_BB), p < 2^31, so product < 2^62 — fits in ulong.
// Reduction via Barrett-like trick: q = (x * mu) >> 62, mu = floor(2^62 / p).
// But simple `% P_BB` on ulong compiles to integer divide; let's use it
// but avoid in tight loops. Since p ~ 2^31 we can do explicit reduction.
inline ulong bb_reduce(ulong x) {
    // x < 2^62; subtract multiples of P_BB.
    // mu = floor(2^62 / P_BB) ~ 2147483647 (since P_BB ~ 2^31)
    // Use: q = x / P_BB (hw divide), or shift-based estimate.
    // For correctness, simple modulo is fine; just inline.
    return x % P_BB;
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

// Simdgroup product reduction for ulong (split into two uint halves).
inline ulong simd_product(ulong v, uint prime_kind) {
    // Butterfly reduction across 32 lanes.
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        uint lo = (uint)v;
        uint hi = (uint)(v >> 32);
        uint olo = simd_shuffle_xor(lo, offset);
        uint ohi = simd_shuffle_xor(hi, offset);
        ulong other = ((ulong)ohi << 32) | (ulong)olo;
        v = mod_mul(v, other, prime_kind);
    }
    return v;
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
            x   = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        num_term = num;
        den_term = mod_sub(alpha, x, prime_kind);
    }

    // Stage 1: simdgroup-wide product reduction (no threadgroup memory).
    uint simd_lane = tid & 31u;
    uint simd_id   = tid >> 5;

    ulong num_simd_prod = simd_product(num_term, prime_kind);
    ulong den_simd_prod = simd_product(den_term, prime_kind);

    if (simd_lane == 0u) {
        num_simd[simd_id] = num_simd_prod;
        den_simd[simd_id] = den_simd_prod;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: final fold of 8 simd products — done by simd 0.
    if (simd_id == 0u) {
        ulong nv, dv;
        if (simd_lane < NUM_SIMDS) {
            nv = num_simd[simd_lane];
            dv = den_simd[simd_lane];
        } else {
            nv = 1ul;
            dv = 1ul;
        }
        // Reduce across first 8 lanes (others are 1).
        for (uint offset = 4u; offset > 0u; offset >>= 1u) {
            uint lo = (uint)nv;
            uint hi = (uint)(nv >> 32);
            uint olo = simd_shuffle_xor(lo, offset);
            uint ohi = simd_shuffle_xor(hi, offset);
            ulong onv = ((ulong)ohi << 32) | (ulong)olo;
            nv = mod_mul(nv, onv, prime_kind);

            uint dlo = (uint)dv;
            uint dhi = (uint)(dv >> 32);
            uint odlo = simd_shuffle_xor(dlo, offset);
            uint odhi = simd_shuffle_xor(dhi, offset);
            ulong odv = ((ulong)odhi << 32) | (ulong)odlo;
            dv = mod_mul(dv, odv, prime_kind);
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