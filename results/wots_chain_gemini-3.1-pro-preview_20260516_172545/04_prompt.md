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

constexpr constant uint  SHA3_RATE_LANES   = 17u;
constexpr constant ulong SHA3_DOMAIN_WORD  = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD    = 0x8000000000000000ul;

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul
};

inline ulong rotl64_opt(ulong x, uint k) {
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    #pragma unroll
    for (uint r = 0; r < 24; ++r) {
        ulong C[5];
        C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D[5];
        D[0] = C[4] ^ rotl64_opt(C[1], 1u);
        D[1] = C[0] ^ rotl64_opt(C[2], 1u);
        D[2] = C[1] ^ rotl64_opt(C[3], 1u);
        D[3] = C[2] ^ rotl64_opt(C[4], 1u);
        D[4] = C[3] ^ rotl64_opt(C[0], 1u);

        // Compute Theta, Rho, and Pi simultaneously. 
        // This completely breaks the dependency chain present in the in-place Pi cycle.
        ulong B[25];
        B[ 0] = A[ 0] ^ D[0];
        B[10] = rotl64_opt(A[ 1] ^ D[1],  1u);
        B[20] = rotl64_opt(A[ 2] ^ D[2], 62u);
        B[ 5] = rotl64_opt(A[ 3] ^ D[3], 28u);
        B[15] = rotl64_opt(A[ 4] ^ D[4], 27u);

        B[16] = rotl64_opt(A[ 5] ^ D[0], 36u);
        B[ 1] = rotl64_opt(A[ 6] ^ D[1], 44u);
        B[11] = rotl64_opt(A[ 7] ^ D[2],  6u);
        B[21] = rotl64_opt(A[ 8] ^ D[3], 55u);
        B[ 6] = rotl64_opt(A[ 9] ^ D[4], 20u);

        B[ 7] = rotl64_opt(A[10] ^ D[0],  3u);
        B[17] = rotl64_opt(A[11] ^ D[1], 10u);
        B[ 2] = rotl64_opt(A[12] ^ D[2], 43u);
        B[12] = rotl64_opt(A[13] ^ D[3], 25u);
        B[22] = rotl64_opt(A[14] ^ D[4], 39u);

        B[23] = rotl64_opt(A[15] ^ D[0], 41u);
        B[ 8] = rotl64_opt(A[16] ^ D[1], 45u);
        B[18] = rotl64_opt(A[17] ^ D[2], 15u);
        B[ 3] = rotl64_opt(A[18] ^ D[3], 21u);
        B[13] = rotl64_opt(A[19] ^ D[4],  8u);

        B[14] = rotl64_opt(A[20] ^ D[0], 18u);
        B[24] = rotl64_opt(A[21] ^ D[1],  2u);
        B[ 9] = rotl64_opt(A[22] ^ D[2], 61u);
        B[19] = rotl64_opt(A[23] ^ D[3], 56u);
        B[ 4] = rotl64_opt(A[24] ^ D[4], 14u);

        // Chi
        #pragma unroll
        for (uint y = 0; y < 25; y += 5) {
            A[y + 0] = B[y + 0] ^ (~B[y + 1] & B[y + 2]);
            A[y + 1] = B[y + 1] ^ (~B[y + 2] & B[y + 3]);
            A[y + 2] = B[y + 2] ^ (~B[y + 3] & B[y + 4]);
            A[y + 3] = B[y + 3] ^ (~B[y + 4] & B[y + 0]);
            A[y + 4] = B[y + 4] ^ (~B[y + 0] & B[y + 1]);
        }

        A[0] ^= KECCAK_RC[r];
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
    
    // Load local loop bound to help the compiler optimize memory loads
    uint chain_len = w;

    if (n_bytes == 16) {
        ulong A[25];
        uint base = idx * 2;
        
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        
        for (uint step = 0; step < chain_len; ++step) {
            #pragma unroll
            for (uint i = 2; i < 25; ++i) A[i] = 0ul;
            A[2] = SHA3_DOMAIN_WORD;
            A[16] = SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        tips[base + 0] = A[0];
        tips[base + 1] = A[1];
        
    } else if (n_bytes == 32) {
        ulong A[25];
        uint base = idx * 4;
        
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        A[2] = seeds[base + 2];
        A[3] = seeds[base + 3];
        
        for (uint step = 0; step < chain_len; ++step) {
            #pragma unroll
            for (uint i = 4; i < 25; ++i) A[i] = 0ul;
            A[4] = SHA3_DOMAIN_WORD;
            A[16] = SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        tips[base + 0] = A[0];
        tips[base + 1] = A[1];
        tips[base + 2] = A[2];
        tips[base + 3] = A[3];
        
    } else {
        uint n_lanes = n_bytes >> 3;
        ulong A[25];
        uint base = idx * n_lanes;
        
        for (uint k = 0; k < n_lanes; ++k) {
            A[k] = seeds[base + k];
        }
        
        for (uint step = 0; step < chain_len; ++step) {
            for (uint i = n_lanes; i < 25; ++i) A[i] = 0ul;
            A[n_lanes] ^= SHA3_DOMAIN_WORD;
            A[16] ^= SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        for (uint k = 0; k < n_lanes; ++k) {
            tips[base + k] = A[k];
        }
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.78 ms, 675.4 Gbitops/s (u64) (117.0% of 577 Gops/s (u64 bitop, est))
          w64_C64K: correct, 22.56 ms, 691.8 Gbitops/s (u64) (119.8% of 577 Gops/s (u64 bitop, est))
         w256_C64K: correct, 90.13 ms, 692.4 Gbitops/s (u64) (119.9% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 1.1889

## History

- iter  0: compile=OK | correct=True | score=0.07572205647034184
- iter  1: compile=OK | correct=True | score=0.08716234102716486
- iter  2: compile=FAIL | correct=False | score=N/A
- iter  3: compile=OK | correct=True | score=1.1888944742819019

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
