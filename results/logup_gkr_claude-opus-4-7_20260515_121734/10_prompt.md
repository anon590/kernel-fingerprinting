## Task: logup_gkr

Batched LogUp lookup-argument core (Haebock 2022; back-end of Jolt / Lasso / Plonkish v2). Given a table T[M] and a witness column w[N] where each w_i := T[witness_idx[i]], compute (1) multiplicities m[j] = #{ i : witness_idx[i] == j } for j in [0, M), and (2) the running product
  P = prod_{i=0..N-1} 1/(alpha - w_i)
     * prod_{j=0..M-1} m_j / (alpha - T_j)   (mod p)
where alpha is a verifier challenge. The host fixes alpha so that alpha is outside the set of table values -- no zero denominators arise.

Combined fingerprint stream of length N + M:
  k <  N:   x_k = T[witness_idx[k]],  num_k = 1
  k >= N:   x_k = T[k - N],            num_k = m[k - N]

The host issues two dispatches in a single compute command encoder. Their serial ordering provides the implicit barrier so the second dispatch sees the first's atomic writes:
  Dispatch 1 (logup_count_mult): one thread per witness     row; atomically increments multiplicities[witness_idx[i]].
  Dispatch 2 (logup_partial_product): each threadgroup     of TG_WIDTH = 256 threads owns 256 consecutive indices     in [0, N+M). Each thread computes num_k * 1/(alpha -     x_k); threadgroup-cooperatively reduces the 256 terms     into one tile product written to partial[tgid]. Threads     with k >= N+M contribute the multiplicative identity     (1). The host then multiplies partial[0..K-1] (K =     ceil((N+M)/256)) on the CPU to obtain the final running     product (the sub-millisecond host fold is intentionally     untimed).

Field selection (constant prime_kind):
  0 = Goldilocks  p = 2^64 - 2^32 + 1
  1 = BabyBear    p = 2^31 - 2^27 + 1 = 2013265921
Both reductions are runtime-dispatched on prime_kind; a candidate that hardcodes the Goldilocks reduction macro, or assumes 64-bit limbs are needed, silently fails the held-out BabyBear probe.

All field elements (table, alpha, partial[]) are canonical uint64 in [0, p); a non-canonical output element is a correctness failure even if its residue class matches. Multiplicities are canonical uint32 counts (promoted to ulong only when used as the numerator).

## Required kernel signature(s)

```
kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]]);

kernel void logup_partial_product(
    device const ulong *table          [[buffer(0)]],
    device const uint  *witness_idx    [[buffer(1)]],
    device const uint  *multiplicities [[buffer(2)]],
    device       ulong *partial        [[buffer(3)]],
    constant uint      &N              [[buffer(4)]],
    constant uint      &M              [[buffer(5)]],
    constant uint      &prime_kind     [[buffer(6)]],
    constant ulong     &alpha          [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]);

Dispatch geometry (host-fixed):
  logup_count_mult:
    threadsPerGrid        = (N rounded up to TG width, 1, 1)
    threadsPerThreadgroup = (min(N, 256), 1, 1)
  logup_partial_product:
    threadsPerGrid        = (K * 256, 1, 1)   K = ceil((N+M)/256)
    threadsPerThreadgroup = (256, 1, 1)        // FIXED at TG_WIDTH=256

The 256-wide threadgroup is part of the host-kernel contract for logup_partial_product: K = ceil((N+M) / 256) is baked into the host-side partial[] allocation, so the kernel must emit exactly one tile-product per 256 consecutive indices. The host pre-zeroes multiplicities[M] before every dispatch. The two dispatches share a single MTLComputeCommandEncoder; the implicit cross-dispatch barrier in serial mode gives logup_partial_product a coherent view of multiplicities written by logup_count_mult.
```

## Your previous attempt

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
inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}
inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
    return t;
}
// Non-canonical multiply: returns value in [0, 2^64) congruent mod p.
// Safe to feed back into gold_mul_nc.
inline ulong gold_mul_nc(ulong a, ulong b) {
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
    return t2;
}
inline ulong gold_mul(ulong a, ulong b) {
    return gold_canonical(gold_mul_nc(a, b));
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

        // Stage 1: simdgroup butterfly — use non-canonical mul throughout
        {
            ulong v = den_term;
            v = gold_mul_nc(v, simd_xor_ulong(v, 16u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  8u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  4u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  2u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  1u));
            den_term = gold_canonical(v);
        }
        if (tile_has_table) {
            ulong v = num_term;
            v = gold_mul_nc(v, simd_xor_ulong(v, 16u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  8u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  4u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  2u));
            v = gold_mul_nc(v, simd_xor_ulong(v,  1u));
            num_term = gold_canonical(v);
        }

        if (simd_lane == 0u) {
            den_simd[simd_id] = den_term;
            if (tile_has_table) num_simd[simd_id] = num_term;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage 2: fold NUM_SIMDS=8 in simd 0
        if (simd_id == 0u) {
            ulong dv = (simd_lane < NUM_SIMDS) ? den_simd[simd_lane] : 1ul;
            dv = gold_mul_nc(dv, simd_xor_ulong(dv, 4u));
            dv = gold_mul_nc(dv, simd_xor_ulong(dv, 2u));
            dv = gold_mul_nc(dv, simd_xor_ulong(dv, 1u));
            dv = gold_canonical(dv);

            ulong nv;
            if (tile_has_table) {
                nv = (simd_lane < NUM_SIMDS) ? num_simd[simd_lane] : 1ul;
                nv = gold_mul_nc(nv, simd_xor_ulong(nv, 4u));
                nv = gold_mul_nc(nv, simd_xor_ulong(nv, 2u));
                nv = gold_mul_nc(nv, simd_xor_ulong(nv, 1u));
                nv = gold_canonical(nv);
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

Result of previous attempt:
          gold_M4K: correct, 0.08 ms, 0.8 Gmodmul/s (int64) (1.5% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 0.59 ms, 1.7 Gmodmul/s (int64) (3.1% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 3.02 ms, 5.2 Gmodmul/s (int64) (9.8% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0359

## Current best (incumbent)

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
            num_term = 1ul;
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

Incumbent result:
          gold_M4K: correct, 0.08 ms, 0.8 Gmodmul/s (int64) (1.5% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 0.30 ms, 3.3 Gmodmul/s (int64) (6.1% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 3.06 ms, 5.1 Gmodmul/s (int64) (9.6% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0446

## History

- iter  2: compile=OK | correct=True | score=0.004431301440059308
- iter  3: compile=OK | correct=True | score=0.003648669361099135
- iter  4: compile=OK | correct=True | score=0.004972451308839243
- iter  5: compile=OK | correct=True | score=0.004494865361908353
- iter  6: compile=OK | correct=True | score=0.04459558550252404
- iter  7: compile=OK | correct=True | score=0.040811012892945184
- iter  8: compile=OK | correct=True | score=0.039401419013110686
- iter  9: compile=OK | correct=True | score=0.035914970801963916

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
