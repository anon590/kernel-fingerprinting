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

#define U64(HI, LO) ((ulong(HI) << 32u) | ulong(LO))
#define ROL64(x, N) (((x) << (N)) | ((x) >> (64u - (N))))

#define KECCAK_CHI_IOTA64(RC) { \
    ulong t0; ulong t1; \
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
    a0 ^= (RC); \
}

#define KECCAK_RHO_PI_CHI_IOTA64(RC) { \
    ulong t = a1; \
    a1  = ROL64(a6,  44u); \
    a6  = ROL64(a9,  20u); \
    a9  = ROL64(a22, 61u); \
    a22 = ROL64(a14, 39u); \
    a14 = ROL64(a20, 18u); \
    a20 = ROL64(a2,  62u); \
    a2  = ROL64(a12, 43u); \
    a12 = ROL64(a13, 25u); \
    a13 = ROL64(a19, 8u); \
    a19 = ROL64(a23, 56u); \
    a23 = ROL64(a15, 41u); \
    a15 = ROL64(a4,  27u); \
    a4  = ROL64(a24, 14u); \
    a24 = ROL64(a21, 2u); \
    a21 = ROL64(a8,  55u); \
    a8  = ROL64(a16, 45u); \
    a16 = ROL64(a5,  36u); \
    a5  = ROL64(a3,  28u); \
    a3  = ROL64(a18, 21u); \
    a18 = ROL64(a17, 15u); \
    a17 = ROL64(a11, 10u); \
    a11 = ROL64(a7,  6u); \
    a7  = ROL64(a10, 3u); \
    a10 = ROL64(t,   1u); \
    KECCAK_CHI_IOTA64(RC) \
}

#define KECCAK_ROUND64(RC) { \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    ulong d = c4 ^ ROL64(c1, 1u); \
    a0 ^= d; a5 ^= d; a10 ^= d; a15 ^= d; a20 ^= d; \
    d = c0 ^ ROL64(c2, 1u); \
    a1 ^= d; a6 ^= d; a11 ^= d; a16 ^= d; a21 ^= d; \
    d = c1 ^ ROL64(c3, 1u); \
    a2 ^= d; a7 ^= d; a12 ^= d; a17 ^= d; a22 ^= d; \
    d = c2 ^ ROL64(c4, 1u); \
    a3 ^= d; a8 ^= d; a13 ^= d; a18 ^= d; a23 ^= d; \
    d = c3 ^ ROL64(c0, 1u); \
    a4 ^= d; a9 ^= d; a14 ^= d; a19 ^= d; a24 ^= d; \
    KECCAK_RHO_PI_CHI_IOTA64(RC) \
}

#define KECCAK_MIDDLE_1_TO_22_64() \
    KECCAK_ROUND64(U64(0x00000000u, 0x00008082u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x0000808Au)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008000u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000808Bu)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x80000001u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008081u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008009u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000008Au)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x00000088u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x80008009u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x8000000Au)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x8000808Bu)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x0000008Bu)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008089u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008003u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008002u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00000080u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000800Au)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x8000000Au)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008081u)) \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008080u)) \
    KECCAK_ROUND64(U64(0x00000000u, 0x80000001u))

#define KECCAK_LAST2_64(RC) { \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    ulong d0 = c4 ^ ROL64(c1, 1u); \
    ulong d1 = c0 ^ ROL64(c2, 1u); \
    ulong d2 = c1 ^ ROL64(c3, 1u); \
    ulong d3 = c2 ^ ROL64(c4, 1u); \
    ulong b0 = a0 ^ d0; \
    ulong b1 = ROL64(a6  ^ d1, 44u); \
    ulong b2 = ROL64(a12 ^ d2, 43u); \
    ulong b3 = ROL64(a18 ^ d3, 21u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ (RC); \
    a1 =  b1 ^ ((~b2) & b3); \
}

#define KECCAK_LAST4_64(RC) { \
    ulong c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    ulong c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    ulong c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    ulong c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    ulong c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    ulong d0 = c4 ^ ROL64(c1, 1u); \
    ulong d1 = c0 ^ ROL64(c2, 1u); \
    ulong d2 = c1 ^ ROL64(c3, 1u); \
    ulong d3 = c2 ^ ROL64(c4, 1u); \
    ulong d4 = c3 ^ ROL64(c0, 1u); \
    ulong b0 = a0 ^ d0; \
    ulong b1 = ROL64(a6  ^ d1, 44u); \
    ulong b2 = ROL64(a12 ^ d2, 43u); \
    ulong b3 = ROL64(a18 ^ d3, 21u); \
    ulong b4 = ROL64(a24 ^ d4, 14u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ (RC); \
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

    const ulong DOM = ulong(0x00000006u);
    const ulong PAD = U64(0x80000000u, 0x00000000u);
    const ulong RC0 = U64(0x00000000u, 0x00000001u);
    const ulong RCL = U64(0x80000000u, 0x80008008u);

    if (n_bytes == 16u) {
        uint base = idx << 1u;
        ulong v0 = seeds[base + 0u];
        ulong v1 = seeds[base + 1u];

        for (uint step = w; step != 0u; --step) {
            ulong d0 = ROL64(v1, 1u) ^ ulong(0x00000001u);
            ulong d1 = v0 ^ ulong(0x0000000Cu);
            ulong d2 = v1 ^ PAD;

            ulong a0  = v0 ^ d0;
            ulong a1  = ROL64(d1, 44u);
            ulong a2  = ROL64(d2, 43u);
            ulong a3  = ROL64(DOM, 21u);
            ulong a4  = ROL64(v0, 15u);

            ulong a5  = ROL64(DOM, 28u);
            ulong a6  = ROL64(v0, 21u);
            ulong a7  = ROL64(d0, 3u);
            ulong a8  = ROL64(PAD ^ d1, 45u);
            ulong a9  = ROL64(d2, 61u);

            ulong a10 = ROL64(v1 ^ d1, 1u);
            ulong a11 = ROL64(d2, 6u);
            ulong a12 = ROL64(DOM, 25u);
            ulong a13 = ROL64(v0, 9u);
            ulong a14 = ROL64(d0, 18u);

            ulong a15 = ROL64(v0, 28u);
            ulong a16 = ROL64(d0, 36u);
            ulong a17 = ROL64(d1, 10u);
            ulong a18 = ROL64(d2, 15u);
            ulong a19 = ROL64(DOM, 56u);

            ulong a20 = ROL64(DOM ^ d2, 62u);
            ulong a21 = ROL64(DOM, 55u);
            ulong a22 = ROL64(v0, 40u);
            ulong a23 = ROL64(d0, 41u);
            ulong a24 = ROL64(d1, 2u);

            KECCAK_CHI_IOTA64(RC0)
            KECCAK_MIDDLE_1_TO_22_64()
            KECCAK_LAST2_64(RCL)

            v0 = a0;
            v1 = a1;
        }

        tips[base + 0u] = v0;
        tips[base + 1u] = v1;
    } else {
        uint base = idx << 2u;
        ulong v0 = seeds[base + 0u];
        ulong v1 = seeds[base + 1u];
        ulong v2 = seeds[base + 2u];
        ulong v3 = seeds[base + 3u];

        for (uint step = w; step != 0u; --step) {
            ulong d0 = ROL64(v1, 1u) ^ ulong(0x00000007u);
            ulong d1 = v0 ^ ROL64(v2, 1u);
            ulong d2 = v1 ^ PAD ^ ROL64(v3, 1u);
            ulong d3 = v2 ^ ulong(0x0000000Cu);
            ulong d4 = v3 ^ ROL64(v0, 1u);

            ulong a0  = v0 ^ d0;
            ulong a1  = ROL64(d1, 44u);
            ulong a2  = ROL64(d2, 43u);
            ulong a3  = ROL64(d3, 21u);
            ulong a4  = ROL64(d4, 14u);

            ulong a5  = ROL64(v3 ^ d3, 28u);
            ulong a6  = ROL64(d4, 20u);
            ulong a7  = ROL64(d0, 3u);
            ulong a8  = ROL64(PAD ^ d1, 45u);
            ulong a9  = ROL64(d2, 61u);

            ulong a10 = ROL64(v1 ^ d1, 1u);
            ulong a11 = ROL64(d2, 6u);
            ulong a12 = ROL64(d3, 25u);
            ulong a13 = ROL64(d4, 8u);
            ulong a14 = ROL64(d0, 18u);

            ulong a15 = ROL64(DOM ^ d4, 27u);
            ulong a16 = ROL64(d0, 36u);
            ulong a17 = ROL64(d1, 10u);
            ulong a18 = ROL64(d2, 15u);
            ulong a19 = ROL64(d3, 56u);

            ulong a20 = ROL64(v2 ^ d2, 62u);
            ulong a21 = ROL64(d3, 55u);
            ulong a22 = ROL64(d4, 39u);
            ulong a23 = ROL64(d0, 41u);
            ulong a24 = ROL64(d1, 2u);

            KECCAK_CHI_IOTA64(RC0)
            KECCAK_MIDDLE_1_TO_22_64()
            KECCAK_LAST4_64(RCL)

            v0 = a0;
            v1 = a1;
            v2 = a2;
            v3 = a3;
        }

        tips[base + 0u] = v0;
        tips[base + 1u] = v1;
        tips[base + 2u] = v2;
        tips[base + 3u] = v3;
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.51 ms, 708.2 Gbitops/s (u64) (62.9% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.42 ms, 728.6 Gbitops/s (u64) (64.8% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 87.32 ms, 714.7 Gbitops/s (u64) (63.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6374

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

inline uint2 split_u64(ulong x) {
    return as_type<uint2>(x);
}

inline ulong join_u64(uint2 x) {
    return as_type<ulong>(x);
}

#define ROL2_LO(v, N) uint2((((v).x << (N)) | ((v).y >> (32u - (N)))), (((v).y << (N)) | ((v).x >> (32u - (N)))))
#define ROL2_HI(v, N) uint2((((v).y << ((N) - 32u)) | ((v).x >> (64u - (N)))), (((v).x << ((N) - 32u)) | ((v).y >> (64u - (N)))))

#define KECCAK_CHI_IOTA(RCLO, RCHI) { \
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
    a0 ^= uint2((RCLO), (RCHI)); \
}

#define KECCAK_RHO_PI_CHI_IOTA(RCLO, RCHI) { \
    uint2 t = a1; \
    a1  = ROL2_HI(a6,  44u); \
    a6  = ROL2_LO(a9,  20u); \
    a9  = ROL2_HI(a22, 61u); \
    a22 = ROL2_HI(a14, 39u); \
    a14 = ROL2_LO(a20, 18u); \
    a20 = ROL2_HI(a2,  62u); \
    a2  = ROL2_HI(a12, 43u); \
    a12 = ROL2_LO(a13, 25u); \
    a13 = ROL2_LO(a19, 8u); \
    a19 = ROL2_HI(a23, 56u); \
    a23 = ROL2_HI(a15, 41u); \
    a15 = ROL2_LO(a4,  27u); \
    a4  = ROL2_LO(a24, 14u); \
    a24 = ROL2_LO(a21, 2u); \
    a21 = ROL2_HI(a8,  55u); \
    a8  = ROL2_HI(a16, 45u); \
    a16 = ROL2_HI(a5,  36u); \
    a5  = ROL2_LO(a3,  28u); \
    a3  = ROL2_LO(a18, 21u); \
    a18 = ROL2_LO(a17, 15u); \
    a17 = ROL2_LO(a11, 10u); \
    a11 = ROL2_LO(a7,  6u); \
    a7  = ROL2_LO(a10, 3u); \
    a10 = ROL2_LO(t,   1u); \
    KECCAK_CHI_IOTA(RCLO, RCHI) \
}

#define KECCAK_ROUND(RCLO, RCHI) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d = c4 ^ ROL2_LO(c1, 1u); \
    a0 ^= d; a5 ^= d; a10 ^= d; a15 ^= d; a20 ^= d; \
    d = c0 ^ ROL2_LO(c2, 1u); \
    a1 ^= d; a6 ^= d; a11 ^= d; a16 ^= d; a21 ^= d; \
    d = c1 ^ ROL2_LO(c3, 1u); \
    a2 ^= d; a7 ^= d; a12 ^= d; a17 ^= d; a22 ^= d; \
    d = c2 ^ ROL2_LO(c4, 1u); \
    a3 ^= d; a8 ^= d; a13 ^= d; a18 ^= d; a23 ^= d; \
    d = c3 ^ ROL2_LO(c0, 1u); \
    a4 ^= d; a9 ^= d; a14 ^= d; a19 ^= d; a24 ^= d; \
    KECCAK_RHO_PI_CHI_IOTA(RCLO, RCHI) \
}

#define KECCAK_MIDDLE_1_TO_22() \
    KECCAK_ROUND(0x00008082u, 0x00000000u) \
    KECCAK_ROUND(0x0000808Au, 0x80000000u) \
    KECCAK_ROUND(0x80008000u, 0x80000000u) \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u) \
    KECCAK_ROUND(0x80000001u, 0x00000000u) \
    KECCAK_ROUND(0x80008081u, 0x80000000u) \
    KECCAK_ROUND(0x00008009u, 0x80000000u) \
    KECCAK_ROUND(0x0000008Au, 0x00000000u) \
    KECCAK_ROUND(0x00000088u, 0x00000000u) \
    KECCAK_ROUND(0x80008009u, 0x00000000u) \
    KECCAK_ROUND(0x8000000Au, 0x00000000u) \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u) \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u) \
    KECCAK_ROUND(0x00008089u, 0x80000000u) \
    KECCAK_ROUND(0x00008003u, 0x80000000u) \
    KECCAK_ROUND(0x00008002u, 0x80000000u) \
    KECCAK_ROUND(0x00000080u, 0x80000000u) \
    KECCAK_ROUND(0x0000800Au, 0x00000000u) \
    KECCAK_ROUND(0x8000000Au, 0x80000000u) \
    KECCAK_ROUND(0x80008081u, 0x80000000u) \
    KECCAK_ROUND(0x00008080u, 0x80000000u) \
    KECCAK_ROUND(0x80000001u, 0x00000000u)

#define KECCAK_LAST2(RCLO, RCHI) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROL2_LO(c1, 1u); \
    uint2 d1 = c0 ^ ROL2_LO(c2, 1u); \
    uint2 d2 = c1 ^ ROL2_LO(c3, 1u); \
    uint2 d3 = c2 ^ ROL2_LO(c4, 1u); \
    uint2 b0 = a0 ^ d0; \
    uint2 b1 = ROL2_HI(a6  ^ d1, 44u); \
    uint2 b2 = ROL2_HI(a12 ^ d2, 43u); \
    uint2 b3 = ROL2_LO(a18 ^ d3, 21u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2((RCLO), (RCHI)); \
    a1 =  b1 ^ ((~b2) & b3); \
}

#define KECCAK_LAST4(RCLO, RCHI) { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROL2_LO(c1, 1u); \
    uint2 d1 = c0 ^ ROL2_LO(c2, 1u); \
    uint2 d2 = c1 ^ ROL2_LO(c3, 1u); \
    uint2 d3 = c2 ^ ROL2_LO(c4, 1u); \
    uint2 d4 = c3 ^ ROL2_LO(c0, 1u); \
    uint2 b0 = a0 ^ d0; \
    uint2 b1 = ROL2_HI(a6  ^ d1, 44u); \
    uint2 b2 = ROL2_HI(a12 ^ d2, 43u); \
    uint2 b3 = ROL2_LO(a18 ^ d3, 21u); \
    uint2 b4 = ROL2_LO(a24 ^ d4, 14u); \
    a0 = (b0 ^ ((~b1) & b2)) ^ uint2((RCLO), (RCHI)); \
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
    const uint2 DOM = uint2(0x00000006u, 0u);
    const uint2 PAD = uint2(0u, 0x80000000u);

    if (n_lanes == 2u) {
        uint base = idx << 1u;
        uint2 v0 = split_u64(seeds[base + 0u]);
        uint2 v1 = split_u64(seeds[base + 1u]);

        for (uint step = w; step != 0u; --step) {
            uint2 d0 = ROL2_LO(v1, 1u) ^ uint2(0x00000001u, 0u);
            uint2 d1 = v0 ^ uint2(0x0000000Cu, 0u);
            uint2 d2 = v1 ^ PAD;

            uint2 a0  = v0 ^ d0;
            uint2 a1  = ROL2_HI(d1, 44u);
            uint2 a2  = ROL2_HI(d2, 43u);
            uint2 a3  = ROL2_LO(DOM, 21u);
            uint2 a4  = ROL2_LO(v0, 15u);

            uint2 a5  = ROL2_LO(DOM, 28u);
            uint2 a6  = ROL2_LO(v0, 21u);
            uint2 a7  = ROL2_LO(d0, 3u);
            uint2 a8  = ROL2_HI(PAD ^ d1, 45u);
            uint2 a9  = ROL2_HI(d2, 61u);

            uint2 a10 = ROL2_LO(v1 ^ d1, 1u);
            uint2 a11 = ROL2_LO(d2, 6u);
            uint2 a12 = ROL2_LO(DOM, 25u);
            uint2 a13 = ROL2_LO(v0, 9u);
            uint2 a14 = ROL2_LO(d0, 18u);

            uint2 a15 = ROL2_LO(v0, 28u);
            uint2 a16 = ROL2_HI(d0, 36u);
            uint2 a17 = ROL2_LO(d1, 10u);
            uint2 a18 = ROL2_LO(d2, 15u);
            uint2 a19 = ROL2_HI(DOM, 56u);

            uint2 a20 = ROL2_HI(DOM ^ d2, 62u);
            uint2 a21 = ROL2_HI(DOM, 55u);
            uint2 a22 = ROL2_HI(v0, 40u);
            uint2 a23 = ROL2_HI(d0, 41u);
            uint2 a24 = ROL2_LO(d1, 2u);

            KECCAK_CHI_IOTA(0x00000001u, 0x00000000u)
            KECCAK_MIDDLE_1_TO_22()
            KECCAK_LAST2(0x80008008u, 0x80000000u)

            v0 = a0;
            v1 = a1;
        }

        tips[base + 0u] = join_u64(v0);
        tips[base + 1u] = join_u64(v1);
    } else {
        uint base = idx << 2u;
        uint2 v0 = split_u64(seeds[base + 0u]);
        uint2 v1 = split_u64(seeds[base + 1u]);
        uint2 v2 = split_u64(seeds[base + 2u]);
        uint2 v3 = split_u64(seeds[base + 3u]);

        for (uint step = w; step != 0u; --step) {
            uint2 d0 = ROL2_LO(v1, 1u) ^ uint2(0x00000007u, 0u);
            uint2 d1 = v0 ^ ROL2_LO(v2, 1u);
            uint2 d2 = v1 ^ PAD ^ ROL2_LO(v3, 1u);
            uint2 d3 = v2 ^ uint2(0x0000000Cu, 0u);
            uint2 d4 = v3 ^ ROL2_LO(v0, 1u);

            uint2 a0  = v0 ^ d0;
            uint2 a1  = ROL2_HI(d1, 44u);
            uint2 a2  = ROL2_HI(d2, 43u);
            uint2 a3  = ROL2_LO(d3, 21u);
            uint2 a4  = ROL2_LO(d4, 14u);

            uint2 a5  = ROL2_LO(v3 ^ d3, 28u);
            uint2 a6  = ROL2_LO(d4, 20u);
            uint2 a7  = ROL2_LO(d0, 3u);
            uint2 a8  = ROL2_HI(PAD ^ d1, 45u);
            uint2 a9  = ROL2_HI(d2, 61u);

            uint2 a10 = ROL2_LO(v1 ^ d1, 1u);
            uint2 a11 = ROL2_LO(d2, 6u);
            uint2 a12 = ROL2_LO(d3, 25u);
            uint2 a13 = ROL2_LO(d4, 8u);
            uint2 a14 = ROL2_LO(d0, 18u);

            uint2 a15 = ROL2_LO(DOM ^ d4, 27u);
            uint2 a16 = ROL2_HI(d0, 36u);
            uint2 a17 = ROL2_LO(d1, 10u);
            uint2 a18 = ROL2_LO(d2, 15u);
            uint2 a19 = ROL2_HI(d3, 56u);

            uint2 a20 = ROL2_HI(v2 ^ d2, 62u);
            uint2 a21 = ROL2_HI(d3, 55u);
            uint2 a22 = ROL2_HI(d4, 39u);
            uint2 a23 = ROL2_HI(d0, 41u);
            uint2 a24 = ROL2_LO(d1, 2u);

            KECCAK_CHI_IOTA(0x00000001u, 0x00000000u)
            KECCAK_MIDDLE_1_TO_22()
            KECCAK_LAST4(0x80008008u, 0x80000000u)

            v0 = a0;
            v1 = a1;
            v2 = a2;
            v3 = a3;
        }

        tips[base + 0u] = join_u64(v0);
        tips[base + 1u] = join_u64(v1);
        tips[base + 2u] = join_u64(v2);
        tips[base + 3u] = join_u64(v3);
    }
}
```

Incumbent result:
          w16_C64K: correct, 5.39 ms, 724.0 Gbitops/s (u64) (64.4% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 20.88 ms, 747.1 Gbitops/s (u64) (66.4% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 84.28 ms, 740.5 Gbitops/s (u64) (65.8% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6552

## History

- iter  2: compile=OK | correct=True | score=0.65292479588798
- iter  3: compile=OK | correct=True | score=0.6542232756263334
- iter  4: compile=OK | correct=True | score=0.655223119722654
- iter  5: compile=OK | correct=True | score=0.6530115175447916
- iter  6: compile=OK | correct=True | score=0.649739455698701
- iter  7: compile=OK | correct=True | score=0.6550962151905839
- iter  8: compile=OK | correct=True | score=0.6510511348003015
- iter  9: compile=OK | correct=True | score=0.6374360037358607

## Stagnation notice

Your last 3 correct attempts all scored within 15% of
the incumbent without surpassing it. You are circling a local
optimum. STOP making incremental edits to the previous kernel and
propose a STRUCTURALLY different approach.

A reworded version of the previous kernel will not break out of
this plateau.

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
