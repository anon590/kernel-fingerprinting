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
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// Sequence of indices for the in-place Pi permutation cycle
constant uint PI_LANE[24] = {
    1, 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6
};

inline ulong rotl64_opt(ulong x, uint k) {
    uint amt = k & 63u;
    return (amt == 0u) ? x : ((x << amt) | (x >> (64u - amt)));
}

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    #pragma unroll
    for (uint r = 0; r < 24; ++r) {
        ulong C[5];
        #pragma unroll
        for (uint x = 0; x < 5; ++x) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        
        ulong D[5];
        #pragma unroll
        for (uint x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rotl64_opt(C[(x + 1) % 5], 1u);
        }
        
        #pragma unroll
        for (uint y = 0; y < 5; ++y) {
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                A[x + 5 * y] ^= D[x];
            }
        }
        
        ulong current = A[1];
        #pragma unroll
        for (uint i = 0; i < 24; ++i) {
            uint src = PI_LANE[i];
            uint dest = PI_LANE[(i + 1) % 24];
            ulong temp = A[dest];
            A[dest] = rotl64_opt(current, KECCAK_RHO[src]);
            current = temp;
        }
        
        #pragma unroll
        for (uint y = 0; y < 5; ++y) {
            ulong T[5];
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                T[x] = A[x + 5 * y];
            }
            #pragma unroll
            for (uint x = 0; x < 5; ++x) {
                A[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
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

    // Specialize for expected cryptographic digest sizes to guarantee that
    // state array accesses use strictly compile-time constant indices. 
    // This perfectly scalarizes `A` into thread registers.
    if (n_bytes == 16) {
        ulong A[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * 2;
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        
        for (uint step = 0; step < w; ++step) {
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
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * 4;
        A[0] = seeds[base + 0];
        A[1] = seeds[base + 1];
        A[2] = seeds[base + 2];
        A[3] = seeds[base + 3];
        
        for (uint step = 0; step < w; ++step) {
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
        // Generic fallback for any other sizes
        uint n_lanes = n_bytes >> 3;
        ulong A[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0ul;
        
        uint base = idx * n_lanes;
        for (uint k = 0; k < n_lanes; ++k) {
            A[k] = seeds[base + k];
        }
        
        for (uint step = 0; step < w; ++step) {
            for (uint i = n_lanes; i < 25; ++i) A[i] = 0ul;
            A[n_lanes] ^= SHA3_DOMAIN_WORD;
            A[SHA3_RATE_LANES - 1u] ^= SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
        
        for (uint k = 0; k < n_lanes; ++k) {
            tips[base + k] = A[k];
        }
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 77.35 ms, 50.4 Gbitops/s (u64) (8.7% of 577 Gops/s (u64 bitop, est))
          w64_C64K: correct, 310.46 ms, 50.3 Gbitops/s (u64) (8.7% of 577 Gops/s (u64 bitop, est))
         w256_C64K: correct, 1240.78 ms, 50.3 Gbitops/s (u64) (8.7% of 577 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0872

## History

- iter  0: compile=OK | correct=True | score=0.07572205647034184
- iter  1: compile=OK | correct=True | score=0.08716234102716486

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
