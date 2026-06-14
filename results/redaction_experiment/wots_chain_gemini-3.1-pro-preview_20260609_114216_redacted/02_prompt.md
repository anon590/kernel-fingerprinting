## Task: wots_chain

Batched WOTS+ / SPHINCS+-style hash chains. Given ``n_chains`` independent ``n_bytes``-byte seeds, apply the Keccak-256 inner hash ``w`` times in sequence per chain (each digest truncated to ``n_bytes`` bytes before feeding into the next iteration) and write the chain tip to the output. The chains are embarrassingly parallel; the ``w``-step iteration along each chain is strictly sequential.

Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing -- rate = 136 bytes (17 lanes), capacity = 64 bytes, domain pad byte = 0x06. State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k = x + 5*y holds bytes 8*k .. 8*k + 7 of the sponge state in little-endian.

All test sizes have ``n_bytes < rate_bytes`` (``n_bytes`` is bound at runtime and varies across the configurations the kernel is scored on; rate_bytes=136), so every chain step collapses to a single-block absorb + single-block squeeze of ``n_lanes = n_bytes / 8`` state lanes:
  state                          := 0
  state[lane 0..n_lanes-1]       := previous_chunk
  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain
  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad
  state                          := Keccak-f1600(state)
  next_chunk                     := state[lane 0..n_lanes-1]

On the first chain step the absorb is the seed; on every subsequent step the absorb is the n_lanes-lane truncation of the previous Keccak-f1600 output. After ``w`` steps the first n_lanes state lanes are written to the output as the chain tip.

The kernel must read ``n_bytes`` and ``w`` from the bound device buffers rather than treating them as compile-time constants; both vary across the test sizes (``w`` in {16, 64, 256} among the baseline sizes shown; both ``w`` and ``n_bytes`` are bound at runtime and vary across the configurations the kernel is scored on). Hardcoding either value silently produces wrong output, not just slow output.

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

inline ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline __attribute__((always_inline)) void keccak_f1600(thread ulong *A) {
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 24u; ++r) {
        // Theta
        ulong C[5];
        #pragma clang loop unroll(full)
        for (uint x = 0u; x < 5u; ++x) {
            C[x] = A[x] ^ A[x + 5u] ^ A[x + 10u] ^ A[x + 15u] ^ A[x + 20u];
        }
        
        ulong D[5];
        #pragma clang loop unroll(full)
        for (uint x = 0u; x < 5u; ++x) {
            D[x] = C[(x + 4u) % 5u] ^ rotl64(C[(x + 1u) % 5u], 1u);
        }
        
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 25u; ++i) {
            A[i] ^= D[i % 5u];
        }

        // Rho and Pi (in-place cycle)
        ulong current = A[1];
        uint x_pos = 1u;
        uint y_pos = 0u;
        
        #pragma clang loop unroll(full)
        for (uint t = 0u; t < 24u; ++t) {
            uint src = x_pos + 5u * y_pos;
            uint x_new = y_pos;
            uint y_new = (2u * x_pos + 3u * y_pos) % 5u;
            uint dst = x_new + 5u * y_new;
            
            ulong temp = A[dst];
            A[dst] = rotl64(current, KECCAK_RHO[src]);
            current = temp;
            
            x_pos = x_new;
            y_pos = y_new;
        }

        // Chi
        #pragma clang loop unroll(full)
        for (uint y = 0u; y < 5u; ++y) {
            ulong T[5];
            #pragma clang loop unroll(full)
            for (uint x = 0u; x < 5u; ++x) {
                T[x] = A[x + 5u * y];
            }
            #pragma clang loop unroll(full)
            for (uint x = 0u; x < 5u; ++x) {
                A[x + 5u * y] = T[x] ^ ((~T[(x + 1u) % 5u]) & T[(x + 2u) % 5u]);
            }
        }

        // Iota
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

    // Allocate state array cleanly to avoid uninitialized values
    thread ulong A[25] = {0};
    
    // Initial absorb (avoid dynamic indexing to keep A purely in registers)
    uint base = idx * n_lanes;
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 17u; ++k) {
        if (k < n_lanes) A[k] = seeds[base + k];
    }

    for (uint step = 0u; step < w; ++step) {
        // Zeroing and domain padding via unrolled conditional execution
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 25u; ++i) {
            if (i >= n_lanes) A[i] = 0ul;
            if (i == n_lanes) A[i] ^= SHA3_DOMAIN_WORD;
        }
        
        A[16] ^= SHA3_FINAL_PAD; // SHA3_RATE_LANES - 1u == 16
        keccak_f1600(A);
    }

    // Export chain tip via static indexing bounds check
    #pragma clang loop unroll(full)
    for (uint k = 0u; k < 17u; ++k) {
        if (k < n_lanes) tips[base + k] = A[k];
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 42.66 ms, 91.4 Gbitops/s (u64) (8.1% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 172.62 ms, 90.4 Gbitops/s (u64) (8.0% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 679.52 ms, 91.8 Gbitops/s (u64) (8.2% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0811

## History

- iter  0: compile=OK | correct=True | score=0.03917601338347657
- iter  1: compile=OK | correct=True | score=0.08108350649983091

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
