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

constant uint  P_BB_U  = 2013265921u;
constant ulong P_BB    = 2013265921ul;
constant ulong BB_MU   = 9162596893ul; // floor(2^64 / 2013265921)

// ----------------------------------------------------------------------
// 64 x 64 -> 128 using 32-bit partial products.
// ----------------------------------------------------------------------

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)a;
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)b;
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

// ----------------------------------------------------------------------
// Goldilocks field.
// ----------------------------------------------------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) {
        t -= EPSILON;
    }
    return t;
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;

    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) {
        t0 -= EPSILON;
    }

    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) {
        t2 += EPSILON;
    }

    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 p = umul128(a, b);
    return gold_reduce128(p.x, p.y);
}

// Fixed exponent p - 2 = 0xFFFFFFFEFFFFFFFF.
// Bits after the top bit: 30 ones, one zero, 32 ones.
inline ulong gold_inv(ulong x) {
    ulong r = x;

    for (uint i = 0u; i < 30u; ++i) {
        r = gold_mul(r, r);
        r = gold_mul(r, x);
    }

    r = gold_mul(r, r); // zero bit

    for (uint i = 0u; i < 32u; ++i) {
        r = gold_mul(r, r);
        r = gold_mul(r, x);
    }

    return r;
}

// ----------------------------------------------------------------------
// BabyBear field.
// ----------------------------------------------------------------------

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + P_BB_U - b);
}

inline uint bb_reduce(ulong x) {
    ulong2 qprod = umul128(x, BB_MU);
    ulong q = qprod.y;
    ulong r = x - q * P_BB;
    if (r >= P_BB) {
        r -= P_BB;
    }
    return (uint)r;
}

inline uint bb_mul(uint a, uint b) {
    return bb_reduce((ulong)a * (ulong)b);
}

// Fixed exponent p - 2 = 0x77FFFFFF.
// Bits after the top bit: two ones, one zero, 27 ones.
inline uint bb_inv(uint x) {
    uint r = x;

    for (uint i = 0u; i < 2u; ++i) {
        r = bb_mul(r, r);
        r = bb_mul(r, x);
    }

    r = bb_mul(r, r); // zero bit

    for (uint i = 0u; i < 27u; ++i) {
        r = bb_mul(r, r);
        r = bb_mul(r, x);
    }

    return r;
}

// ----------------------------------------------------------------------
// SIMD helpers.
// ----------------------------------------------------------------------

inline uint2 pack_u64(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong unpack_u64(uint2 x) {
    return ((ulong)x.y << 32) | (ulong)x.x;
}

inline ulong simd_product_gold(ulong v) {
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)16)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)8)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)4)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)2)));
    v = gold_mul(v, unpack_u64(simd_shuffle_xor(pack_u64(v), (ushort)1)));
    return v;
}

inline uint simd_product_bb(uint v) {
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)16));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)8));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)4));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)2));
    v = bb_mul(v, simd_shuffle_xor(v, (ushort)1));
    return v;
}

// ----------------------------------------------------------------------
// Kernel A: multiplicity count.
// ----------------------------------------------------------------------

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) {
        return;
    }

    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

// ----------------------------------------------------------------------
// Kernel B: tile product.
// One inversion per 256-element tile:
//   prod(num_i / den_i) = prod(num_i) * inv(prod(den_i)).
// ----------------------------------------------------------------------

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
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong scratch_den[8];
    threadgroup ulong scratch_num[8];

    uint lane = tid & 31u;
    uint sg   = tid >> 5;

    ulong tile_start = (ulong)tgid * 256ul;
    bool witness_only_tile = (tile_start + 255ul) < (ulong)N;

    ulong x = 0ul;
    uint  num32 = 1u;
    bool  active = false;

    if (witness_only_tile) {
        x = table[witness_idx[gid]];
    } else {
        ulong total = (ulong)N + (ulong)M;
        active = ((ulong)gid < total);

        if (active) {
            if (gid < N) {
                x = table[witness_idx[gid]];
                num32 = 1u;
            } else {
                uint j = gid - N;
                x = table[j];
                num32 = multiplicities[j];
            }
        }
    }

    if (prime_kind == 0u) {
        if (witness_only_tile) {
            ulong den = gold_sub(alpha, x);
            ulong den_sg = simd_product_gold(den);

            if (lane == 0u) {
                scratch_den[sg] = den_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                ulong d = (lane < 8u) ? scratch_den[lane] : 1ul;
                d = simd_product_gold(d);

                ulong out = gold_inv(d);
                if (lane == 0u) {
                    partial[tgid] = out;
                }
            }
        } else {
            ulong den = active ? gold_sub(alpha, x) : 1ul;
            ulong num = active ? (ulong)num32 : 1ul;

            ulong den_sg = simd_product_gold(den);
            ulong num_sg = simd_product_gold(num);

            if (lane == 0u) {
                scratch_den[sg] = den_sg;
                scratch_num[sg] = num_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                ulong d = (lane < 8u) ? scratch_den[lane] : 1ul;
                ulong n = (lane < 8u) ? scratch_num[lane] : 1ul;

                d = simd_product_gold(d);
                n = simd_product_gold(n);

                if (n == 0ul) {
                    if (lane == 0u) {
                        partial[tgid] = 0ul;
                    }
                } else {
                    ulong out = gold_mul(n, gold_inv(d));
                    if (lane == 0u) {
                        partial[tgid] = out;
                    }
                }
            }
        }
    } else {
        uint alpha32 = (uint)alpha;

        if (witness_only_tile) {
            uint den = bb_sub(alpha32, (uint)x);
            uint den_sg = simd_product_bb(den);

            if (lane == 0u) {
                scratch_den[sg] = (ulong)den_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                uint d = (lane < 8u) ? (uint)scratch_den[lane] : 1u;
                d = simd_product_bb(d);

                uint out = bb_inv(d);
                if (lane == 0u) {
                    partial[tgid] = (ulong)out;
                }
            }
        } else {
            uint den = active ? bb_sub(alpha32, (uint)x) : 1u;
            uint num = active ? num32 : 1u;

            uint den_sg = simd_product_bb(den);
            uint num_sg = simd_product_bb(num);

            if (lane == 0u) {
                scratch_den[sg] = (ulong)den_sg;
                scratch_num[sg] = (ulong)num_sg;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0u) {
                uint d = (lane < 8u) ? (uint)scratch_den[lane] : 1u;
                uint n = (lane < 8u) ? (uint)scratch_num[lane] : 1u;

                d = simd_product_bb(d);
                n = simd_product_bb(n);

                if (n == 0u) {
                    if (lane == 0u) {
                        partial[tgid] = 0ul;
                    }
                } else {
                    uint out = bb_mul(n, bb_inv(d));
                    if (lane == 0u) {
                        partial[tgid] = (ulong)out;
                    }
                }
            }
        }
    }
}
```

Result of previous attempt:
          gold_M4K: correct, 0.08 ms, 0.8 Gmodmul/s (int64) (1.4% of 53 Gops/s (int64 mul, est))
         gold_M64K: correct, 0.58 ms, 1.7 Gmodmul/s (int64) (3.2% of 53 Gops/s (int64 mul, est))
          gold_M1M: correct, 2.95 ms, 5.3 Gmodmul/s (int64) (10.0% of 53 Gops/s (int64 mul, est))
  score (gmean of fraction): 0.0358

## History

- iter  0: compile=OK | correct=True | score=0.0010133018385154713
- iter  1: compile=OK | correct=True | score=0.03575542992013374

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
