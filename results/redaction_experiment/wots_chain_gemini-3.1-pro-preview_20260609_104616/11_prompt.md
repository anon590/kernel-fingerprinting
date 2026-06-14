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

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define K_ROUND(a00, a01, a02, a03, a04, \
                a05, a06, a07, a08, a09, \
                a10, a11, a12, a13, a14, \
                a15, a16, a17, a18, a19, \
                a20, a21, a22, a23, a24, \
                e00, e01, e02, e03, e04, \
                e05, e06, e07, e08, e09, \
                e10, e11, e12, e13, e14, \
                e15, e16, e17, e18, e19, \
                e20, e21, e22, e23, e24, rc) \
do { \
    ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20; \
    ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21; \
    ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22; \
    ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23; \
    ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong b00 = a00 ^ D0; \
    ulong b01 = rotl64(a06 ^ D1, 44u); \
    ulong b02 = rotl64(a12 ^ D2, 43u); \
    ulong b03 = rotl64(a18 ^ D3, 21u); \
    ulong b04 = rotl64(a24 ^ D4, 14u); \
    e00 = b00 ^ (b02 & ~b01) ^ rc; \
    e01 = b01 ^ (b03 & ~b02); \
    e02 = b02 ^ (b04 & ~b03); \
    e03 = b03 ^ (b00 & ~b04); \
    e04 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a03 ^ D3, 28u); \
    b01 = rotl64(a09 ^ D4, 20u); \
    b02 = rotl64(a10 ^ D0, 3u); \
    b03 = rotl64(a16 ^ D1, 45u); \
    b04 = rotl64(a22 ^ D2, 61u); \
    e05 = b00 ^ (b02 & ~b01); \
    e06 = b01 ^ (b03 & ~b02); \
    e07 = b02 ^ (b04 & ~b03); \
    e08 = b03 ^ (b00 & ~b04); \
    e09 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a01 ^ D1, 1u); \
    b01 = rotl64(a07 ^ D2, 6u); \
    b02 = rotl64(a13 ^ D3, 25u); \
    b03 = rotl64(a19 ^ D4, 8u); \
    b04 = rotl64(a20 ^ D0, 18u); \
    e10 = b00 ^ (b02 & ~b01); \
    e11 = b01 ^ (b03 & ~b02); \
    e12 = b02 ^ (b04 & ~b03); \
    e13 = b03 ^ (b00 & ~b04); \
    e14 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a04 ^ D4, 27u); \
    b01 = rotl64(a05 ^ D0, 36u); \
    b02 = rotl64(a11 ^ D1, 10u); \
    b03 = rotl64(a17 ^ D2, 15u); \
    b04 = rotl64(a23 ^ D3, 56u); \
    e15 = b00 ^ (b02 & ~b01); \
    e16 = b01 ^ (b03 & ~b02); \
    e17 = b02 ^ (b04 & ~b03); \
    e18 = b03 ^ (b00 & ~b04); \
    e19 = b04 ^ (b01 & ~b00); \
    \
    b00 = rotl64(a02 ^ D2, 62u); \
    b01 = rotl64(a08 ^ D3, 55u); \
    b02 = rotl64(a14 ^ D4, 39u); \
    b03 = rotl64(a15 ^ D0, 41u); \
    b04 = rotl64(a21 ^ D1, 2u); \
    e20 = b00 ^ (b02 & ~b01); \
    e21 = b01 ^ (b03 & ~b02); \
    e22 = b02 ^ (b04 & ~b03); \
    e23 = b03 ^ (b00 & ~b04); \
    e24 = b04 ^ (b01 & ~b00); \
} while(0)

#define K_ROUND_PAIR(r) \
    K_ROUND(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, \
            e00, e01, e02, e03, e04, e05, e06, e07, e08, e09, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, KECCAK_RC[r]); \
    K_ROUND(e00, e01, e02, e03, e04, e05, e06, e07, e08, e09, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, \
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, KECCAK_RC[r+1])

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    uint n_lanes = n_bytes >> 3;
    uint w_val = w;

    ulong a00, a01, a02, a03, a04, a05, a06, a07, a08, a09;
    ulong a10, a11, a12, a13, a14, a15, a16, a17, a18, a19;
    ulong a20, a21, a22, a23, a24;
    
    ulong e00, e01, e02, e03, e04, e05, e06, e07, e08, e09;
    ulong e10, e11, e12, e13, e14, e15, e16, e17, e18, e19;
    ulong e20, e21, e22, e23, e24;

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        a00 = seed_val.x;
        a01 = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            a02 = 0x06ul;
            a03 = 0ul; a04 = 0ul; a05 = 0ul; a06 = 0ul;
            a07 = 0ul; a08 = 0ul; a09 = 0ul; a10 = 0ul;
            a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
            a15 = 0ul; a16 = 0x8000000000000000ul; a17 = 0ul;
            a18 = 0ul; a19 = 0ul; a20 = 0ul; a21 = 0ul;
            a22 = 0ul; a23 = 0ul; a24 = 0ul;
            
            K_ROUND_PAIR(0);
            K_ROUND_PAIR(2);
            K_ROUND_PAIR(4);
            K_ROUND_PAIR(6);
            K_ROUND_PAIR(8);
            K_ROUND_PAIR(10);
            K_ROUND_PAIR(12);
            K_ROUND_PAIR(14);
            K_ROUND_PAIR(16);
            K_ROUND_PAIR(18);
            K_ROUND_PAIR(20);
            K_ROUND_PAIR(22);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(a00, a01);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        a00 = seed_val.x;
        a01 = seed_val.y;
        a02 = seed_val.z;
        a03 = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            a04 = 0x06ul;
            a05 = 0ul; a06 = 0ul; a07 = 0ul; a08 = 0ul; a09 = 0ul;
            a10 = 0ul; a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
            a15 = 0ul; a16 = 0x8000000000000000ul; a17 = 0ul;
            a18 = 0ul; a19 = 0ul; a20 = 0ul; a21 = 0ul;
            a22 = 0ul; a23 = 0ul; a24 = 0ul;
            
            K_ROUND_PAIR(0);
            K_ROUND_PAIR(2);
            K_ROUND_PAIR(4);
            K_ROUND_PAIR(6);
            K_ROUND_PAIR(8);
            K_ROUND_PAIR(10);
            K_ROUND_PAIR(12);
            K_ROUND_PAIR(14);
            K_ROUND_PAIR(16);
            K_ROUND_PAIR(18);
            K_ROUND_PAIR(20);
            K_ROUND_PAIR(22);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(a00, a01, a02, a03);

    } else {
        uint base = idx * n_lanes;
        a00=0; a01=0; a02=0; a03=0; a04=0;
        a05=0; a06=0; a07=0; a08=0; a09=0;
        a10=0; a11=0; a12=0; a13=0; a14=0;
        a15=0; a16=0; a17=0; a18=0; a19=0;
        a20=0; a21=0; a22=0; a23=0; a24=0;
        
        if (n_lanes > 0) a00 = seeds[base + 0];
        if (n_lanes > 1) a01 = seeds[base + 1];
        if (n_lanes > 2) a02 = seeds[base + 2];
        if (n_lanes > 3) a03 = seeds[base + 3];
        if (n_lanes > 4) a04 = seeds[base + 4];
        if (n_lanes > 5) a05 = seeds[base + 5];
        if (n_lanes > 6) a06 = seeds[base + 6];
        if (n_lanes > 7) a07 = seeds[base + 7];
        
        for (uint step = 0u; step < w_val; ++step) {
            if (n_lanes < 16) a15 = 0;
            if (n_lanes < 15) a14 = 0;
            if (n_lanes < 14) a13 = 0;
            if (n_lanes < 13) a12 = 0;
            if (n_lanes < 12) a11 = 0;
            if (n_lanes < 11) a10 = 0;
            if (n_lanes < 10) a09 = 0;
            if (n_lanes < 9)  a08 = 0;
            if (n_lanes < 8)  a07 = 0;
            if (n_lanes < 7)  a06 = 0;
            if (n_lanes < 6)  a05 = 0;
            if (n_lanes < 5)  a04 = 0;
            if (n_lanes < 4)  a03 = 0;
            if (n_lanes < 3)  a02 = 0;
            if (n_lanes < 2)  a01 = 0;
            if (n_lanes < 1)  a00 = 0;
            
            a16 = 0x8000000000000000ul;
            a17 = 0; a18 = 0; a19 = 0; a20 = 0;
            a21 = 0; a22 = 0; a23 = 0; a24 = 0;
            
            switch(n_lanes) {
                case 0: a00 ^= 0x06ul; break;
                case 1: a01 ^= 0x06ul; break;
                case 2: a02 ^= 0x06ul; break;
                case 3: a03 ^= 0x06ul; break;
                case 4: a04 ^= 0x06ul; break;
                case 5: a05 ^= 0x06ul; break;
                case 6: a06 ^= 0x06ul; break;
                case 7: a07 ^= 0x06ul; break;
                case 8: a08 ^= 0x06ul; break;
                case 9: a09 ^= 0x06ul; break;
                case 10: a10 ^= 0x06ul; break;
                case 11: a11 ^= 0x06ul; break;
                case 12: a12 ^= 0x06ul; break;
                case 13: a13 ^= 0x06ul; break;
                case 14: a14 ^= 0x06ul; break;
                case 15: a15 ^= 0x06ul; break;
                case 16: a16 ^= 0x06ul; break;
            }
            
            K_ROUND_PAIR(0);
            K_ROUND_PAIR(2);
            K_ROUND_PAIR(4);
            K_ROUND_PAIR(6);
            K_ROUND_PAIR(8);
            K_ROUND_PAIR(10);
            K_ROUND_PAIR(12);
            K_ROUND_PAIR(14);
            K_ROUND_PAIR(16);
            K_ROUND_PAIR(18);
            K_ROUND_PAIR(20);
            K_ROUND_PAIR(22);
        }
        
        if (n_lanes > 0) tips[base + 0] = a00;
        if (n_lanes > 1) tips[base + 1] = a01;
        if (n_lanes > 2) tips[base + 2] = a02;
        if (n_lanes > 3) tips[base + 3] = a03;
        if (n_lanes > 4) tips[base + 4] = a04;
        if (n_lanes > 5) tips[base + 5] = a05;
        if (n_lanes > 6) tips[base + 6] = a06;
        if (n_lanes > 7) tips[base + 7] = a07;
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.59 ms, 697.6 Gbitops/s (u64) (62.0% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.66 ms, 720.4 Gbitops/s (u64) (64.0% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 87.59 ms, 712.5 Gbitops/s (u64) (63.3% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6312

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define ROUND(A, E, rc) do { \
    ulong C0 = (A[0]  ^ A[5])  ^ (A[10] ^ A[15]) ^ A[20]; \
    ulong C1 = (A[1]  ^ A[6])  ^ (A[11] ^ A[16]) ^ A[21]; \
    ulong C2 = (A[2]  ^ A[7])  ^ (A[12] ^ A[17]) ^ A[22]; \
    ulong C3 = (A[3]  ^ A[8])  ^ (A[13] ^ A[18]) ^ A[23]; \
    ulong C4 = (A[4]  ^ A[9])  ^ (A[14] ^ A[19]) ^ A[24]; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong b0 = A[0] ^ D0; \
    ulong b1 = rotl64(A[6] ^ D1, 44u); \
    ulong b2 = rotl64(A[12] ^ D2, 43u); \
    ulong b3 = rotl64(A[18] ^ D3, 21u); \
    ulong b4 = rotl64(A[24] ^ D4, 14u); \
    E[0] = b0 ^ (b2 & ~b1) ^ rc; \
    E[1] = b1 ^ (b3 & ~b2); \
    E[2] = b2 ^ (b4 & ~b3); \
    E[3] = b3 ^ (b0 & ~b4); \
    E[4] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[3] ^ D3, 28u); \
    b1 = rotl64(A[9] ^ D4, 20u); \
    b2 = rotl64(A[10] ^ D0, 3u); \
    b3 = rotl64(A[16] ^ D1, 45u); \
    b4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = b0 ^ (b2 & ~b1); \
    E[6] = b1 ^ (b3 & ~b2); \
    E[7] = b2 ^ (b4 & ~b3); \
    E[8] = b3 ^ (b0 & ~b4); \
    E[9] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[1] ^ D1, 1u); \
    b1 = rotl64(A[7] ^ D2, 6u); \
    b2 = rotl64(A[13] ^ D3, 25u); \
    b3 = rotl64(A[19] ^ D4, 8u); \
    b4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = b0 ^ (b2 & ~b1); \
    E[11] = b1 ^ (b3 & ~b2); \
    E[12] = b2 ^ (b4 & ~b3); \
    E[13] = b3 ^ (b0 & ~b4); \
    E[14] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[4] ^ D4, 27u); \
    b1 = rotl64(A[5] ^ D0, 36u); \
    b2 = rotl64(A[11] ^ D1, 10u); \
    b3 = rotl64(A[17] ^ D2, 15u); \
    b4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = b0 ^ (b2 & ~b1); \
    E[16] = b1 ^ (b3 & ~b2); \
    E[17] = b2 ^ (b4 & ~b3); \
    E[18] = b3 ^ (b0 & ~b4); \
    E[19] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[2] ^ D2, 62u); \
    b1 = rotl64(A[8] ^ D3, 55u); \
    b2 = rotl64(A[14] ^ D4, 39u); \
    b3 = rotl64(A[15] ^ D0, 41u); \
    b4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = b0 ^ (b2 & ~b1); \
    E[21] = b1 ^ (b3 & ~b2); \
    E[22] = b2 ^ (b4 & ~b3); \
    E[23] = b3 ^ (b0 & ~b4); \
    E[24] = b4 ^ (b1 & ~b0); \
} while(0)

struct KeccakState {
    ulong A[25];
};

inline __attribute__((always_inline))
KeccakState keccak_f1600(KeccakState s) {
    KeccakState e;
    #pragma unroll
    for (uint r = 0u; r < 24u; r += 2u) {
        ROUND(s.A, e.A, KECCAK_RC[r]);
        ROUND(e.A, s.A, KECCAK_RC[r+1]);
    }
    return s;
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

    uint n_lanes = n_bytes >> 3;
    uint w_val = w;

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        
        KeccakState s;
        s.A[0] = seed_val.x;
        s.A[1] = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            s.A[2]  = 0x06ul;
            s.A[3]  = 0ul; s.A[4]  = 0ul; s.A[5]  = 0ul; s.A[6]  = 0ul;
            s.A[7]  = 0ul; s.A[8]  = 0ul; s.A[9]  = 0ul; s.A[10] = 0ul;
            s.A[11] = 0ul; s.A[12] = 0ul; s.A[13] = 0ul; s.A[14] = 0ul;
            s.A[15] = 0ul; 
            s.A[16] = 0x8000000000000000ul;
            s.A[17] = 0ul; s.A[18] = 0ul; s.A[19] = 0ul; s.A[20] = 0ul;
            s.A[21] = 0ul; s.A[22] = 0ul; s.A[23] = 0ul; s.A[24] = 0ul;
            
            s = keccak_f1600(s);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(s.A[0], s.A[1]);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        
        KeccakState s;
        s.A[0] = seed_val.x;
        s.A[1] = seed_val.y;
        s.A[2] = seed_val.z;
        s.A[3] = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            s.A[4]  = 0x06ul;
            s.A[5]  = 0ul; s.A[6]  = 0ul; s.A[7]  = 0ul; s.A[8]  = 0ul; 
            s.A[9]  = 0ul; s.A[10] = 0ul; s.A[11] = 0ul; s.A[12] = 0ul; 
            s.A[13] = 0ul; s.A[14] = 0ul; s.A[15] = 0ul; 
            s.A[16] = 0x8000000000000000ul;
            s.A[17] = 0ul; s.A[18] = 0ul; s.A[19] = 0ul; s.A[20] = 0ul;
            s.A[21] = 0ul; s.A[22] = 0ul; s.A[23] = 0ul; s.A[24] = 0ul;
            
            s = keccak_f1600(s);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(s.A[0], s.A[1], s.A[2], s.A[3]);

    } else {
        uint base = idx * n_lanes;
        KeccakState s;
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            s.A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) s.A[i] = 0ul;
                }
            }
            s.A[n_lanes] ^= 0x06ul;
            s.A[16u]     ^= 0x8000000000000000ul;
            
            s = keccak_f1600(s);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = s.A[i];
        }
    }
}
```

Incumbent result:
          w16_C64K: correct, 5.53 ms, 705.7 Gbitops/s (u64) (62.7% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.50 ms, 725.7 Gbitops/s (u64) (64.5% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 86.44 ms, 722.0 Gbitops/s (u64) (64.2% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6380

## History

- iter  3: compile=OK | correct=True | score=0.6293248516727208
- iter  4: compile=OK | correct=True | score=0.6302812102022003
- iter  5: compile=OK | correct=True | score=0.6098559015825479
- iter  6: compile=OK | correct=True | score=0.6379862316008869
- iter  7: compile=OK | correct=True | score=0.6278056530485507
- iter  8: compile=OK | correct=True | score=0.6335814359777029
- iter  9: compile=OK | correct=True | score=0.6344030963479368
- iter 10: compile=OK | correct=True | score=0.6312090616392295

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
