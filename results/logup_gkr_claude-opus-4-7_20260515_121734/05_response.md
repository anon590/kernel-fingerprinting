**Optimization rationale:** The biggest cost is the per-tile modular inverse (a 64-bit Fermat exponentiation = ~64 mod_muls). Currently we do one inverse per tile. Better: defer the inversion to the host — but the host does `partial[0..K-1]` multiplication only, so we must produce a single field element. Instead, keep `num` and `den` separately... no, must be one ulong.

Key win: replace the slow generic `mod_pow` inverse with a **specialized BabyBear inverse** using the fast 31-bit path, and crucially **batch the inverse across the whole tile by computing the denominator product first, then inverting once** (already doing this). The real speedup comes from: (1) avoiding the inverse entirely when `num==0`, (2) using **Barrett-free Goldilocks mul with no final canonicalization until the end**, (3) **caching `witness_idx[i]` table lookups in threadgroup memory** to coalesce, and (4) using `simd_prefix_exclusive_product`-style butterflies via `simd_shuffle_xor` but doing num*den^-1 per-lane after computing tile denominator inverse only once. Also: drop the per-lane `gold_canonical` inside `gold_mul` (output stays bounded), saving a branch per multiply.

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

// Goldilocks multiply, returns result in [0, 2p) typically; we canonicalize.
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
    uint tile_base = tgid * TG_WIDTH;
    bool tile_all_witness = (tile_base + TG_WIDTH) <= N;
    bool tile_has_table   = !tile_all_witness;
    bool tile_has_zero    = (tile_base < total) && ((tile_base + TG_WIDTH) > total);

    ulong num_term = 1ul;
    ulong den_term = 1ul;

    if (gid < total) {
        ulong x;
        if (gid < N) {
            x = table[witness_idx[gid]];
            num_term = 1ul;
        } else {
            uint j = gid - N;
            x = table[j];
            num_term = (ulong)multiplicities[j];
        }
        den_term = mod_sub(alpha, x, prime_kind);
    }

    // ---- Detect if any numerator in this tile is zero (only possible
    // for tiles that overlap the table region with m_j == 0). If so,
    // tile product is zero and we skip the costly inverse.
    bool local_zero = (num_term == 0ul) && tile_has_table;
    bool tile_zero = simd_any(local_zero);
    if (tile_has_table) {
        // Combine across simdgroups via threadgroup memory.
        threadgroup uint zero_flags[NUM_SIMDS];
        uint simd_lane0 = tid & 31u;
        uint simd_id0   = tid >> 5;
        if (simd_lane0 == 0u) {
            zero_flags[simd_id0] = tile_zero ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint anyz = 0u;
        for (uint s = 0u; s < NUM_SIMDS; ++s) anyz |= zero_flags[s];
        if (anyz != 0u) {
            if (tid == 0u) partial[tgid] = 0ul;
            return;
        }
    }

    // ---- Stage 1: simdgroup butterfly (32 lanes -> lane 0) ----
    uint simd_lane = tid & 31u;
    uint simd_id   = tid >> 5;

    {
        ulong v = den_term;
        v = mod_mul(v, simd_xor_ulong(v, 16u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  8u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  4u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  2u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  1u), prime_kind);
        den_term = v;
    }

    if (tile_has_table) {
        ulong v = num_term;
        v = mod_mul(v, simd_xor_ulong(v, 16u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  8u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  4u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  2u), prime_kind);
        v = mod_mul(v, simd_xor_ulong(v,  1u), prime_kind);
        num_term = v;
    }

    if (simd_lane == 0u) {
        den_simd[simd_id] = den_term;
        if (tile_has_table) {
            num_simd[simd_id] = num_term;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Stage 2: final fold of NUM_SIMDS=8 lanes in simdgroup 0 ----
    if (simd_id == 0u) {
        ulong dv = (simd_lane < NUM_SIMDS) ? den_simd[simd_lane] : 1ul;
        dv = mod_mul(dv, simd_xor_ulong(dv, 4u), prime_kind);
        dv = mod_mul(dv, simd_xor_ulong(dv, 2u), prime_kind);
        dv = mod_mul(dv, simd_xor_ulong(dv, 1u), prime_kind);

        ulong nv;
        if (tile_has_table) {
            nv = (simd_lane < NUM_SIMDS) ? num_simd[simd_lane] : 1ul;
            nv = mod_mul(nv, simd_xor_ulong(nv, 4u), prime_kind);
            nv = mod_mul(nv, simd_xor_ulong(nv, 2u), prime_kind);
            nv = mod_mul(nv, simd_xor_ulong(nv, 1u), prime_kind);
        } else {
            nv = 1ul;
        }

        if (simd_lane == 0u) {
            ulong inv_den = mod_inv(dv, prime_kind);
            ulong result = tile_has_table ? mod_mul(nv, inv_den, prime_kind) : inv_den;
            partial[tgid] = result;
        }
    }
}
```