## Task: wots_chain

Batched WOTS+ / SPHINCS+-style hash chains. Given ``n_chains`` independent ``n_bytes``-byte seeds, apply the Keccak-256 inner hash ``w`` times in sequence per chain (each digest truncated to ``n_bytes`` bytes before feeding into the next iteration) and write the chain tip to the output. The chains are embarrassingly parallel; the ``w``-step iteration along each chain is strictly sequential.

Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing -- rate = 136 bytes (17 lanes), capacity = 64 bytes, domain pad byte = 0x06. State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k = x + 5*y holds bytes 8*k .. 8*k + 7 of the sponge state in little-endian.

All test sizes have ``n_bytes < rate_bytes`` (in-distribution n_bytes=16, held-out n_bytes=32; rate_bytes=136), so every chain step collapses to a single-block absorb + single-block squeeze of ``n_lanes = n_bytes / 8`` state lanes:
  state                          := 0
  state[lane 0..n_lanes-1]       := previous_chunk
  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain
  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad
  state                          := Keccak-f1600(state)
  next_chunk                     := state[lane 0..n_lanes-1]

On the first chain step the absorb is the seed; on every subsequent step the absorb is the n_lanes-lane truncation of the previous Keccak-f1600 output. After ``w`` steps the first n_lanes state lanes are written to the output as the chain tip.

The kernel must read ``n_bytes`` and ``w`` from the bound device buffers rather than treating them as compile-time constants; both vary across the test sizes (``w`` in {16, 64, 256} in the in-distribution sweep, ``n_bytes`` 16 -> 32 between in-distribution and held-out). Hardcoding either value silently produces wrong output, not just slow output.

Correctness is bit-exact against ``hashlib.sha3_256`` iterated ``w`` times with ``n_bytes``-byte truncation; any mismatched output ulong rejects the candidate.

## Required kernel signature(s)

```
kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (n_chains, 1, 1)
  threadsPerThreadgroup = (min(n_chains, 64), 1, 1)
Each thread processes ONE chain end-to-end; guard against idx >= n_chains (the grid is rounded up to a multiple of the TG width). ``seeds`` is laid out as n_chains consecutive runs of ``n_bytes / 8`` ulongs; ``tips`` likewise. The external buffer layout above must be preserved and the per-chain sequential semantics honored: each chain's step ``j+1`` must read the digest produced by its own step ``j`` (cross-chain mixing of intermediate digests would be a correctness bug).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

inline uint compact_even32(uint x) {
    x &= 0x55555555u;
    x = (x | (x >> 1u)) & 0x33333333u;
    x = (x | (x >> 2u)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4u)) & 0x00FF00FFu;
    x = (x | (x >> 8u)) & 0x0000FFFFu;
    return x;
}

inline uint spread16_even(uint x) {
    x &= 0x0000FFFFu;
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

inline uint2 to_bi64(ulong x) {
    uint2 s = as_type<uint2>(x);
    uint e = compact_even32(s.x) | (compact_even32(s.y) << 16u);
    uint o = compact_even32(s.x >> 1u) | (compact_even32(s.y >> 1u) << 16u);
    return uint2(e, o);
}

inline ulong from_bi64(uint2 v) {
    uint lo = spread16_even(v.x) | (spread16_even(v.y) << 1u);
    uint hi = spread16_even(v.x >> 16u) | (spread16_even(v.y >> 16u) << 1u);
    return as_type<ulong>(uint2(lo, hi));
}

#define ROL32_NZ(x, S) (((x) << (S)) | ((x) >> (32u - (S))))
#define ROLI1(v) uint2(ROL32_NZ((v).y, 1u), (v).x)
#define ROLI_E(v, S) (((v) << (S)) | ((v) >> (32u - (S))))
#define ROLI_O(v, SP1, S) uint2(ROL32_NZ((v).y, (SP1)), ROL32_NZ((v).x, (S)))

#define KECCAK_CHI_IOTA_BI(RCE, RCO) { \
    uint2 t0; uint2 t1; \
    t0 = a0;  t1 = a1; \
    a0 = t0 ^ ((~t1) & a2); \
    a1 = t1 ^ ((~a2) & a3); \
    a2 = a2 ^ ((~a3) & a4); \
    a3 = a3 ^ ((~a4) & t0); \
    a4 = a4 ^ ((~t0) & t1); \
    t0 = a5;  t1 = a6; \
    a5 = t0 ^ ((~t1) & a7); \
    a6 = t1 ^ ((~a7) & a8); \
    a7 = a7 ^ ((~a8) & a9); \
    a8 = a8 ^ ((~a9) & t0); \
    a9 = a9 ^ ((~t0) & t1); \
    t0 = a10; t1 = a11; \
    a10 = t0 ^ ((~t1) & a12); \
    a11 = t1 ^ ((~a12) & a13); \
    a12 = a12 ^ ((~a13) & a14); \
    a13 = a13 ^ ((~a14) & t0); \
    a14 = a14 ^ ((~t0) & t1); \
    t0 = a15; t1 = a16; \
    a15 = t0 ^ ((~t1) & a17); \
    a16 = t1 ^ ((~a17) & a18); \
    a17 = a17 ^ ((~a18) & a19); \
    a18 = a18 ^ ((~a19) & t0); \
    a19 = a19 ^ ((~t0) & t1); \
    t0 = a20; t1 = a21; \
    a20 = t0 ^ ((~t1) & a22); \
    a21 = t1 ^ ((~a22) & a23); \
    a22 = a22 ^ ((~a23) & a24); \
    a23 = a23 ^ ((~a24) & t0); \
    a24 = a24 ^ ((~t0) & t1); \
    a0 ^= uint2((RCE), (RCO)); \
}

#define KECCAK_RHO_PI_CHI_IOTA_BI(RCE, RCO) { \
    uint2 t = a1; \
    a1  = ROLI_E(a6,  22u); \
    a6  = ROLI_E(a9,  10u); \
    a9  = ROLI_O(a22, 31u, 30u); \
    a22 = ROLI_O(a14, 20u, 19u); \
    a14 = ROLI_E(a20, 9u); \
    a20 = ROLI_E(a2,  31u); \
    a2  = ROLI_O(a12, 22u, 21u); \
    a12 = ROLI_O(a13, 13u, 12u); \
    a13 = ROLI_E(a19, 4u); \
    a19 = ROLI_E(a23, 28u); \
    a23 = ROLI_O(a15, 21u, 20u); \
    a15 = ROLI_O(a4,  14u, 13u); \
    a4  = ROLI_E(a24, 7u); \
    a24 = ROLI_E(a21, 1u); \
    a21 = ROLI_O(a8,  28u, 27u); \
    a8  = ROLI_O(a16, 23u, 22u); \
    a16 = ROLI_E(a5,  18u); \
    a5  = ROLI_E(a3,  14u); \
    a3  = ROLI_O(a18, 11u, 10u); \
    a18 = ROLI_O(a17, 8u,  7u); \
    a17 = ROLI_E(a11, 5u); \
    a11 = ROLI_E(a7,  3u); \
    a7  = ROLI_O(a10, 2u,  1u); \
    a10 = ROLI1(t); \
    KECCAK_CHI_IOTA_BI(RCE, RCO) \
}

#define KECCAK_ROUND_BI(RCE, RCO) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d = c4 ^ ROLI1(c1); \
    a0 ^= d; a5 ^= d; a10 ^= d; a15 ^= d; a20 ^= d; \
    d = c0 ^ ROLI1(c2); \
    a1 ^= d; a6 ^= d; a11 ^= d; a16 ^= d; a21 ^= d; \
    d = c1 ^ ROLI1(c3); \
    a2 ^= d; a7 ^= d; a12 ^= d; a17 ^= d; a22 ^= d; \
    d = c2 ^ ROLI1(c4); \
    a3 ^= d; a8 ^= d; a13 ^= d; a18 ^= d; a23 ^= d; \
    d = c3 ^ ROLI1(c0); \
    a4 ^= d; a9 ^= d; a14 ^= d; a19 ^= d; a24 ^= d; \
    KECCAK_RHO_PI_CHI_IOTA_BI(RCE, RCO) \
}

#define KECCAK_MIDDLE_1_TO_22_BI() \
    KECCAK_ROUND_BI(0x00000000u, 0x00000089u) \
    KECCAK_ROUND_BI(0x00000000u, 0x8000008Bu) \
    KECCAK_ROUND_BI(0x00000000u, 0x80008080u) \
    KECCAK_ROUND_BI(0x00000001u, 0x0000008Bu) \
    KECCAK_ROUND_BI(0x00000001u, 0x00008000u) \
    KECCAK_ROUND_BI(0x00000001u, 0x80008088u) \
    KECCAK_ROUND_BI(0x00000001u, 0x80000082u) \
    KECCAK_ROUND_BI(0x00000000u, 0x0000000Bu) \
    KECCAK_ROUND_BI(0x00000000u, 0x0000000Au) \
    KECCAK_ROUND_BI(0x00000001u, 0x00008082u) \
    KECCAK_ROUND_BI(0x00000000u, 0x00008003u) \
    KECCAK_ROUND_BI(0x00000001u, 0x0000808Bu) \
    KECCAK_ROUND_BI(0x00000001u, 0x8000000Bu) \
    KECCAK_ROUND_BI(0x00000001u, 0x8000008Au) \
    KECCAK_ROUND_BI(0x00000001u, 0x80000081u) \
    KECCAK_ROUND_BI(0x00000000u, 0x80000081u) \
    KECCAK_ROUND_BI(0x00000000u, 0x80000008u) \
    KECCAK_ROUND_BI(0x00000000u, 0x00000083u) \
    KECCAK_ROUND_BI(0x00000000u, 0x80008003u) \
    KECCAK_ROUND_BI(0x00000001u, 0x80008088u) \
    KECCAK_ROUND_BI(0x00000000u, 0x80000088u) \
    KECCAK_ROUND_BI(0x00000001u, 0x00008000u)

#define KECCAK_LAST2_BI(RCE, RCO) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROLI1(c1); \
    uint2 d1 = c0 ^ ROLI1(c2); \
    uint2 d2 = c1 ^ ROLI1(c3); \
    uint2 d3 = c2 ^ ROLI1(c4); \
    uint2 b0 = a0 ^ d0; \
    uint2 b1 = ROLI_E(a6  ^ d1, 22u); \
    uint2 b2 = ROLI_O(a12 ^ d2, 22u, 21u); \
    uint2 b3 = ROLI_O(a18 ^ d3, 11u, 10u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2((RCE), (RCO)); \
    a1 =  b1 ^ ((~b2) & b3); \
}

#define KECCAK_LAST4_BI(RCE, RCO) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROLI1(c1); \
    uint2 d1 = c0 ^ ROLI1(c2); \
    uint2 d2 = c1 ^ ROLI1(c3); \
    uint2 d3 = c2 ^ ROLI1(c4); \
    uint2 d4 = c3 ^ ROLI1(c0); \
    uint2 b0 = a0 ^ d0; \
    uint2 b1 = ROLI_E(a6  ^ d1, 22u); \
    uint2 b2 = ROLI_O(a12 ^ d2, 22u, 21u); \
    uint2 b3 = ROLI_O(a18 ^ d3, 11u, 10u); \
    uint2 b4 = ROLI_E(a24 ^ d4, 7u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2((RCE), (RCO)); \
    a1 =  b1 ^ ((~b2) & b3); \
    a2 =  b2 ^ ((~b3) & b4); \
    a3 =  b3 ^ ((~b4) & b0); \
}

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    uint n_lanes = n_bytes >> 3u;

    const uint2 DOM = uint2(0x00000002u, 0x00000001u);
    const uint2 PAD = uint2(0x00000000u, 0x80000000u);

    if (n_lanes == 2u) {
        uint base = idx << 1u;
        uint2 v0 = to_bi64(seeds[base + 0u]);
        uint2 v1 = to_bi64(seeds[base + 1u]);

        for (uint step = w; step != 0u; --step) {
            uint2 d0 = ROLI1(v1) ^ uint2(0x00000001u, 0x00000000u);
            uint2 d1 = v0 ^ uint2(0x00000002u, 0x00000002u);
            uint2 d2 = v1 ^ PAD;

            uint2 a0  = v0 ^ d0;
            uint2 a1  = ROLI_E(d1, 22u);
            uint2 a2  = ROLI_O(d2, 22u, 21u);
            uint2 a3  = ROLI_O(DOM, 11u, 10u);
            uint2 a4  = ROLI_O(v0, 8u, 7u);

            uint2 a5  = ROLI_E(DOM, 14u);
            uint2 a6  = ROLI_O(v0, 11u, 10u);
            uint2 a7  = ROLI_O(d0, 2u, 1u);
            uint2 a8  = ROLI_O(PAD ^ d1, 23u, 22u);
            uint2 a9  = ROLI_O(d2, 31u, 30u);

            uint2 a10 = ROLI1(v1 ^ d1);
            uint2 a11 = ROLI_E(d2, 3u);
            uint2 a12 = ROLI_O(DOM, 13u, 12u);
            uint2 a13 = ROLI_O(v0, 5u, 4u);
            uint2 a14 = ROLI_E(d0, 9u);

            uint2 a15 = ROLI_E(v0, 14u);
            uint2 a16 = ROLI_E(d0, 18u);
            uint2 a17 = ROLI_E(d1, 5u);
            uint2 a18 = ROLI_O(d2, 8u, 7u);
            uint2 a19 = ROLI_E(DOM, 28u);

            uint2 a20 = ROLI_E(DOM ^ d2, 31u);
            uint2 a21 = ROLI_O(DOM, 28u, 27u);
            uint2 a22 = ROLI_E(v0, 20u);
            uint2 a23 = ROLI_O(d0, 21u, 20u);
            uint2 a24 = ROLI_E(d1, 1u);

            KECCAK_CHI_IOTA_BI(0x00000001u, 0x00000000u)
            KECCAK_MIDDLE_1_TO_22_BI()
            KECCAK_LAST2_BI(0x00000000u, 0x80008082u)

            v0 = a0;
            v1 = a1;
        }

        tips[base + 0u] = from_bi64(v0);
        tips[base + 1u] = from_bi64(v1);
    } else {
        uint base = idx << 2u;
        uint2 v0 = to_bi64(seeds[base + 0u]);
        uint2 v1 = to_bi64(seeds[base + 1u]);
        uint2 v2 = to_bi64(seeds[base + 2u]);
        uint2 v3 = to_bi64(seeds[base + 3u]);

        for (uint step = w; step != 0u; --step) {
            uint2 d0 = ROLI1(v1) ^ uint2(0x00000003u, 0x00000001u);
            uint2 d1 = v0 ^ ROLI1(v2);
            uint2 d2 = v1 ^ PAD ^ ROLI1(v3);
            uint2 d3 = v2 ^ uint2(0x00000002u, 0x00000002u);
            uint2 d4 = v3 ^ ROLI1(v0);

            uint2 a0  = v0 ^ d0;
            uint2 a1  = ROLI_E(d1, 22u);
            uint2 a2  = ROLI_O(d2, 22u, 21u);
            uint2 a3  = ROLI_O(d3, 11u, 10u);
            uint2 a4  = ROLI_E(d4, 7u);

            uint2 a5  = ROLI_E(v3 ^ d3, 14u);
            uint2 a6  = ROLI_E(d4, 10u);
            uint2 a7  = ROLI_O(d0, 2u, 1u);
            uint2 a8  = ROLI_O(PAD ^ d1, 23u, 22u);
            uint2 a9  = ROLI_O(d2, 31u, 30u);

            uint2 a10 = ROLI1(v1 ^ d1);
            uint2 a11 = ROLI_E(d2, 3u);
            uint2 a12 = ROLI_O(d3, 13u, 12u);
            uint2 a13 = ROLI_E(d4, 4u);
            uint2 a14 = ROLI_E(d0, 9u);

            uint2 a15 = ROLI_O(DOM ^ d4, 14u, 13u);
            uint2 a16 = ROLI_E(d0, 18u);
            uint2 a17 = ROLI_E(d1, 5u);
            uint2 a18 = ROLI_O(d2, 8u, 7u);
            uint2 a19 = ROLI_E(d3, 28u);

            uint2 a20 = ROLI_E(v2 ^ d2, 31u);
            uint2 a21 = ROLI_O(d3, 28u, 27u);
            uint2 a22 = ROLI_O(d4, 20u, 19u);
            uint2 a23 = ROLI_O(d0, 21u, 20u);
            uint2 a24 = ROLI_E(d1, 1u);

            KECCAK_CHI_IOTA_BI(0x00000001u, 0x00000000u)
            KECCAK_MIDDLE_1_TO_22_BI()
            KECCAK_LAST4_BI(0x00000000u, 0x80008082u)

            v0 = a0;
            v1 = a1;
            v2 = a2;
            v3 = a3;
        }

        tips[base + 0u] = from_bi64(v0);
        tips[base + 1u] = from_bi64(v1);
        tips[base + 2u] = from_bi64(v2);
        tips[base + 3u] = from_bi64(v3);
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 4.28 ms, 911.4 Gbitops/s (u64) (81.0% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 16.48 ms, 946.8 Gbitops/s (u64) (84.2% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 65.68 ms, 950.2 Gbitops/s (u64) (84.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.8320

## History

- iter  3: compile=OK | correct=True | score=0.6542232756263334
- iter  4: compile=OK | correct=True | score=0.655223119722654
- iter  5: compile=OK | correct=True | score=0.6530115175447916
- iter  6: compile=OK | correct=True | score=0.649739455698701
- iter  7: compile=OK | correct=True | score=0.6550962151905839
- iter  8: compile=OK | correct=True | score=0.6510511348003015
- iter  9: compile=OK | correct=True | score=0.6374360037358607
- iter 10: compile=OK | correct=True | score=0.8319510863310737

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
