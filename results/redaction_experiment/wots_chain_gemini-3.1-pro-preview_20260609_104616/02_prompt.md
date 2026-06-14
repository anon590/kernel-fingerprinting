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

constexpr constant uint  SHA3_RATE_LANES  = 17u;
constexpr constant ulong SHA3_DOMAIN_WORD = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD   = 0x8000000000000000ul;

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

constexpr constant uint KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (k == 0u) ? x : ((x << k) | (x >> (64u - k)));
}

inline __attribute__((always_inline))
void keccak_f1600(thread ulong *A) {
    #pragma unroll
    for (uint r = 0u; r < 24u; ++r) {
        ulong C[5];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }

        ulong D[5];
        #pragma unroll
        for (uint x = 0u; x < 5u; ++x) {
            D[x] = C[(x + 4u) % 5u] ^ rotl64(C[(x + 1u) % 5u], 1u);
        }

        ulong B[25];
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            uint x = i % 5u;
            uint y = i / 5u;
            uint dest_x = y;
            uint dest_y = (2u * x + 3u * y) % 5u;
            // Directly inline A^D step into rotation
            B[dest_x + 5u * dest_y] = rotl64(A[i] ^ D[x], KECCAK_RHO[i]);
        }

        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            uint x = i % 5u;
            uint y = i / 5u;
            uint i1 = ((x + 1u) % 5u) + 5u * y;
            uint i2 = ((x + 2u) % 5u) + 5u * y;
            A[i] = B[i] ^ ((~B[i1]) & B[i2]);
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

    uint n_lanes = n_bytes >> 3;
    uint base = idx * n_lanes;

    thread ulong A[25];
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) {
        A[i] = 0ul;
    }

    // Explicit manual indexing to force full compiler scalarization
    if (n_lanes == 2u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
    } else if (n_lanes == 4u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
        A[2] = seeds[base + 2u];
        A[3] = seeds[base + 3u];
    } else {
        #pragma unroll
        for (uint k = 0u; k < 17u; ++k) {
            if (k < n_lanes) A[k] = seeds[base + k];
        }
    }

    // Specialize branches by expected sizes. Hard-coding zeroes as pure constants allows 
    // LLVM to fold them deeply into the first round of the Keccak core.
    if (n_lanes == 2u) {
        for (uint step = 0u; step < w; ++step) {
            #pragma unroll
            for (uint i = 2u; i < 25u; ++i) {
                if (i == 2u) A[i] = SHA3_DOMAIN_WORD;
                else if (i == 16u) A[i] = SHA3_FINAL_PAD;
                else A[i] = 0ul;
            }
            keccak_f1600(A);
        }
    } else if (n_lanes == 4u) {
        for (uint step = 0u; step < w; ++step) {
            #pragma unroll
            for (uint i = 4u; i < 25u; ++i) {
                if (i == 4u) A[i] = SHA3_DOMAIN_WORD;
                else if (i == 16u) A[i] = SHA3_FINAL_PAD;
                else A[i] = 0ul;
            }
            keccak_f1600(A);
        }
    } else {
        for (uint step = 0u; step < w; ++step) {
            #pragma unroll
            for (uint i = 0u; i < 25u; ++i) {
                if (i >= n_lanes) A[i] = 0ul;
            }
            #pragma unroll
            for (uint i = 0u; i < 25u; ++i) {
                if (i == n_lanes) A[i] ^= SHA3_DOMAIN_WORD;
            }
            A[16u] ^= SHA3_FINAL_PAD;
            keccak_f1600(A);
        }
    }

    if (n_lanes == 2u) {
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];
    } else if (n_lanes == 4u) {
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];
        tips[base + 2u] = A[2];
        tips[base + 3u] = A[3];
    } else {
        #pragma unroll
        for (uint k = 0u; k < 17u; ++k) {
            if (k < n_lanes) tips[base + k] = A[k];
        }
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 65.56 ms, 59.5 Gbitops/s (u64) (5.3% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 267.09 ms, 58.4 Gbitops/s (u64) (5.2% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 1032.76 ms, 60.4 Gbitops/s (u64) (5.4% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0528

## History

- iter  0: compile=OK | correct=True | score=0.03891053861246893
- iter  1: compile=OK | correct=True | score=0.05283807961281304

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
