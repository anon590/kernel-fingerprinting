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
    E[0] = b0 ^ ((~b1) & b2) ^ rc; \
    E[1] = b1 ^ ((~b2) & b3); \
    E[2] = b2 ^ ((~b3) & b4); \
    E[3] = b3 ^ ((~b4) & b0); \
    E[4] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[3] ^ D3, 28u); \
    b1 = rotl64(A[9] ^ D4, 20u); \
    b2 = rotl64(A[10] ^ D0, 3u); \
    b3 = rotl64(A[16] ^ D1, 45u); \
    b4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = b0 ^ ((~b1) & b2); \
    E[6] = b1 ^ ((~b2) & b3); \
    E[7] = b2 ^ ((~b3) & b4); \
    E[8] = b3 ^ ((~b4) & b0); \
    E[9] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[1] ^ D1, 1u); \
    b1 = rotl64(A[7] ^ D2, 6u); \
    b2 = rotl64(A[13] ^ D3, 25u); \
    b3 = rotl64(A[19] ^ D4, 8u); \
    b4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = b0 ^ ((~b1) & b2); \
    E[11] = b1 ^ ((~b2) & b3); \
    E[12] = b2 ^ ((~b3) & b4); \
    E[13] = b3 ^ ((~b4) & b0); \
    E[14] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[4] ^ D4, 27u); \
    b1 = rotl64(A[5] ^ D0, 36u); \
    b2 = rotl64(A[11] ^ D1, 10u); \
    b3 = rotl64(A[17] ^ D2, 15u); \
    b4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = b0 ^ ((~b1) & b2); \
    E[16] = b1 ^ ((~b2) & b3); \
    E[17] = b2 ^ ((~b3) & b4); \
    E[18] = b3 ^ ((~b4) & b0); \
    E[19] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[2] ^ D2, 62u); \
    b1 = rotl64(A[8] ^ D3, 55u); \
    b2 = rotl64(A[14] ^ D4, 39u); \
    b3 = rotl64(A[15] ^ D0, 41u); \
    b4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = b0 ^ ((~b1) & b2); \
    E[21] = b1 ^ ((~b2) & b3); \
    E[22] = b2 ^ ((~b3) & b4); \
    E[23] = b3 ^ ((~b4) & b0); \
    E[24] = b4 ^ ((~b0) & b1); \
} while(0)

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    ulong E[25];
    ROUND(A, E, 0x0000000000000001ul);
    ROUND(E, A, 0x0000000000008082ul);
    ROUND(A, E, 0x800000000000808Aul);
    ROUND(E, A, 0x8000000080008000ul);
    ROUND(A, E, 0x000000000000808Bul);
    ROUND(E, A, 0x0000000080000001ul);
    ROUND(A, E, 0x8000000080008081ul);
    ROUND(E, A, 0x8000000000008009ul);
    ROUND(A, E, 0x000000000000008Aul);
    ROUND(E, A, 0x0000000000000088ul);
    ROUND(A, E, 0x0000000080008009ul);
    ROUND(E, A, 0x000000008000000Aul);
    ROUND(A, E, 0x000000008000808Bul);
    ROUND(E, A, 0x800000000000008Bul);
    ROUND(A, E, 0x8000000000008089ul);
    ROUND(E, A, 0x8000000000008003ul);
    ROUND(A, E, 0x8000000000008002ul);
    ROUND(E, A, 0x8000000000000080ul);
    ROUND(A, E, 0x000000000000800Aul);
    ROUND(E, A, 0x800000008000000Aul);
    ROUND(A, E, 0x8000000080008081ul);
    ROUND(E, A, 0x8000000000008080ul);
    ROUND(A, E, 0x0000000080000001ul);
    ROUND(E, A, 0x8000000080008008ul);
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
        ulong A[25];
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        A[0] = seed_val.x;
        A[1] = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            A[2]  = 0x06ul;
            A[3]  = 0ul; A[4]  = 0ul; A[5]  = 0ul; A[6]  = 0ul;
            A[7]  = 0ul; A[8]  = 0ul; A[9]  = 0ul; A[10] = 0ul;
            A[11] = 0ul; A[12] = 0ul; A[13] = 0ul; A[14] = 0ul;
            A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(A[0], A[1]);

    } else if (n_lanes == 4u) {
        ulong A[25];
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        A[0] = seed_val.x;
        A[1] = seed_val.y;
        A[2] = seed_val.z;
        A[3] = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            A[4]  = 0x06ul;
            A[5]  = 0ul; A[6]  = 0ul; A[7]  = 0ul; A[8]  = 0ul; 
            A[9]  = 0ul; A[10] = 0ul; A[11] = 0ul; A[12] = 0ul; 
            A[13] = 0ul; A[14] = 0ul; A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(A[0], A[1], A[2], A[3]);

    } else {
        ulong A[25];
        uint base = idx * n_lanes;
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            A[i] = 0ul;
        }
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            if (i < n_lanes) A[i] = seeds[base + i];
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) A[i] = 0ul;
                }
            }
            #pragma unroll
            for (uint i = 0u; i < 25u; ++i) {
                if (i == n_lanes) A[i] ^= 0x06ul;
            }
            A[16u] ^= 0x8000000000000000ul;
            
            keccak_f1600(A);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = A[i];
        }
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.89 ms, 662.8 Gbitops/s (u64) (58.9% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 22.36 ms, 697.7 Gbitops/s (u64) (62.0% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 89.36 ms, 698.5 Gbitops/s (u64) (62.1% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6099

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
    E[0] = b0 ^ ((~b1) & b2) ^ rc; \
    E[1] = b1 ^ ((~b2) & b3); \
    E[2] = b2 ^ ((~b3) & b4); \
    E[3] = b3 ^ ((~b4) & b0); \
    E[4] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[3] ^ D3, 28u); \
    b1 = rotl64(A[9] ^ D4, 20u); \
    b2 = rotl64(A[10] ^ D0, 3u); \
    b3 = rotl64(A[16] ^ D1, 45u); \
    b4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = b0 ^ ((~b1) & b2); \
    E[6] = b1 ^ ((~b2) & b3); \
    E[7] = b2 ^ ((~b3) & b4); \
    E[8] = b3 ^ ((~b4) & b0); \
    E[9] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[1] ^ D1, 1u); \
    b1 = rotl64(A[7] ^ D2, 6u); \
    b2 = rotl64(A[13] ^ D3, 25u); \
    b3 = rotl64(A[19] ^ D4, 8u); \
    b4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = b0 ^ ((~b1) & b2); \
    E[11] = b1 ^ ((~b2) & b3); \
    E[12] = b2 ^ ((~b3) & b4); \
    E[13] = b3 ^ ((~b4) & b0); \
    E[14] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[4] ^ D4, 27u); \
    b1 = rotl64(A[5] ^ D0, 36u); \
    b2 = rotl64(A[11] ^ D1, 10u); \
    b3 = rotl64(A[17] ^ D2, 15u); \
    b4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = b0 ^ ((~b1) & b2); \
    E[16] = b1 ^ ((~b2) & b3); \
    E[17] = b2 ^ ((~b3) & b4); \
    E[18] = b3 ^ ((~b4) & b0); \
    E[19] = b4 ^ ((~b0) & b1); \
    \
    b0 = rotl64(A[2] ^ D2, 62u); \
    b1 = rotl64(A[8] ^ D3, 55u); \
    b2 = rotl64(A[14] ^ D4, 39u); \
    b3 = rotl64(A[15] ^ D0, 41u); \
    b4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = b0 ^ ((~b1) & b2); \
    E[21] = b1 ^ ((~b2) & b3); \
    E[22] = b2 ^ ((~b3) & b4); \
    E[23] = b3 ^ ((~b4) & b0); \
    E[24] = b4 ^ ((~b0) & b1); \
} while(0)

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    ulong E[25];
    #pragma unroll
    for (uint r = 0u; r < 24u; r += 2u) {
        ROUND(A, E, KECCAK_RC[r]);
        ROUND(E, A, KECCAK_RC[r+1]);
    }
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

    thread ulong A[25];

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        A[0] = seed_val.x;
        A[1] = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            A[2]  = 0x06ul;
            A[3]  = 0ul; A[4]  = 0ul; A[5]  = 0ul; A[6]  = 0ul;
            A[7]  = 0ul; A[8]  = 0ul; A[9]  = 0ul; A[10] = 0ul;
            A[11] = 0ul; A[12] = 0ul; A[13] = 0ul; A[14] = 0ul;
            A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(A[0], A[1]);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        A[0] = seed_val.x;
        A[1] = seed_val.y;
        A[2] = seed_val.z;
        A[3] = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            A[4]  = 0x06ul;
            A[5]  = 0ul; A[6]  = 0ul; A[7]  = 0ul; A[8]  = 0ul; 
            A[9]  = 0ul; A[10] = 0ul; A[11] = 0ul; A[12] = 0ul; 
            A[13] = 0ul; A[14] = 0ul; A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(A[0], A[1], A[2], A[3]);

    } else {
        uint base = idx * n_lanes;
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) A[i] = 0ul;
                }
            }
            A[n_lanes] ^= 0x06ul;
            A[16u]     ^= 0x8000000000000000ul;
            
            keccak_f1600(A);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = A[i];
        }
    }
}
```

Incumbent result:
          w16_C64K: correct, 5.57 ms, 699.9 Gbitops/s (u64) (62.2% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.53 ms, 724.8 Gbitops/s (u64) (64.4% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 88.81 ms, 702.8 Gbitops/s (u64) (62.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6303

## History

- iter  0: compile=OK | correct=True | score=0.03891053861246893
- iter  1: compile=OK | correct=True | score=0.05283807961281304
- iter  2: compile=OK | correct=True | score=0.06711178931910174
- iter  3: compile=OK | correct=True | score=0.6293248516727208
- iter  4: compile=OK | correct=True | score=0.6302812102022003
- iter  5: compile=OK | correct=True | score=0.6098559015825479

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
