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

inline __attribute__((always_inline)) ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define K_ROUND(rc) \
    do { \
        ulong C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
        ulong C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
        ulong C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
        ulong C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
        ulong C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
        \
        ulong D0 = C4 ^ rotl64(C1, 1u); \
        ulong D1 = C0 ^ rotl64(C2, 1u); \
        ulong D2 = C1 ^ rotl64(C3, 1u); \
        ulong D3 = C2 ^ rotl64(C4, 1u); \
        ulong D4 = C3 ^ rotl64(C0, 1u); \
        \
        ulong B00 = A00 ^ D0; \
        ulong B10 = rotl64(A11 ^ D1, 44u); \
        ulong B20 = rotl64(A22 ^ D2, 43u); \
        ulong B30 = rotl64(A33 ^ D3, 21u); \
        ulong B40 = rotl64(A44 ^ D4, 14u); \
        \
        ulong B01 = rotl64(A30 ^ D3, 28u); \
        ulong B11 = rotl64(A41 ^ D4, 20u); \
        ulong B21 = rotl64(A02 ^ D0,  3u); \
        ulong B31 = rotl64(A13 ^ D1, 45u); \
        ulong B41 = rotl64(A24 ^ D2, 61u); \
        \
        ulong B02 = rotl64(A10 ^ D1,  1u); \
        ulong B12 = rotl64(A21 ^ D2, 10u); \
        ulong B22 = rotl64(A32 ^ D3, 25u); \
        ulong B32 = rotl64(A43 ^ D4,  8u); \
        ulong B42 = rotl64(A04 ^ D0, 18u); \
        \
        ulong B03 = rotl64(A40 ^ D4, 27u); \
        ulong B13 = rotl64(A01 ^ D0, 36u); \
        ulong B23 = rotl64(A12 ^ D1,  6u); \
        ulong B33 = rotl64(A23 ^ D2, 15u); \
        ulong B43 = rotl64(A34 ^ D3, 56u); \
        \
        ulong B04 = rotl64(A20 ^ D2, 62u); \
        ulong B14 = rotl64(A31 ^ D3, 55u); \
        ulong B24 = rotl64(A42 ^ D4, 39u); \
        ulong B34 = rotl64(A03 ^ D0, 41u); \
        ulong B44 = rotl64(A14 ^ D1,  2u); \
        \
        A00 = B00 ^ (~B10 & B20); A00 ^= rc; \
        A10 = B10 ^ (~B20 & B30); \
        A20 = B20 ^ (~B30 & B40); \
        A30 = B30 ^ (~B40 & B00); \
        A40 = B40 ^ (~B00 & B10); \
        \
        A01 = B01 ^ (~B11 & B21); \
        A11 = B11 ^ (~B21 & B31); \
        A21 = B21 ^ (~B31 & B41); \
        A31 = B31 ^ (~B41 & B01); \
        A41 = B41 ^ (~B01 & B11); \
        \
        A02 = B02 ^ (~B12 & B22); \
        A12 = B12 ^ (~B22 & B32); \
        A22 = B22 ^ (~B32 & B42); \
        A32 = B32 ^ (~B42 & B02); \
        A42 = B42 ^ (~B02 & B12); \
        \
        A03 = B03 ^ (~B13 & B23); \
        A13 = B13 ^ (~B23 & B33); \
        A23 = B23 ^ (~B33 & B43); \
        A33 = B33 ^ (~B43 & B03); \
        A43 = B43 ^ (~B03 & B13); \
        \
        A04 = B04 ^ (~B14 & B24); \
        A14 = B14 ^ (~B24 & B34); \
        A24 = B24 ^ (~B34 & B44); \
        A34 = B34 ^ (~B44 & B04); \
        A44 = B44 ^ (~B04 & B14); \
    } while (0)


#define LOAD(S, idx) if (idx < n_lanes) S = seeds[base + idx]

#define PADA(S, idx) \
    S = select(S, 0ul, (bool)(idx >= n_lanes)); \
    S ^= select(0ul, 0x06ul, (bool)(idx == n_lanes))

#define STORE(S, idx) if (idx < n_lanes) tips[base + idx] = S


kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_chains) return;

    uint n_lanes = n_bytes >> 3;
    uint base = tid * n_lanes;

    ulong A00 = 0, A10 = 0, A20 = 0, A30 = 0, A40 = 0;
    ulong A01 = 0, A11 = 0, A21 = 0, A31 = 0, A41 = 0;
    ulong A02 = 0, A12 = 0, A22 = 0, A32 = 0, A42 = 0;
    ulong A03 = 0, A13 = 0, A23 = 0, A33 = 0, A43 = 0;
    ulong A04 = 0, A14 = 0, A24 = 0, A34 = 0, A44 = 0;

    LOAD(A00, 0); LOAD(A10, 1); LOAD(A20, 2); LOAD(A30, 3); LOAD(A40, 4);
    LOAD(A01, 5); LOAD(A11, 6); LOAD(A21, 7); LOAD(A31, 8); LOAD(A41, 9);
    LOAD(A02, 10); LOAD(A12, 11); LOAD(A22, 12); LOAD(A32, 13); LOAD(A42, 14);
    LOAD(A03, 15); LOAD(A13, 16); 

    for (uint step = 0u; step < w; ++step) {
        
        PADA(A00, 0); PADA(A10, 1); PADA(A20, 2); PADA(A30, 3); PADA(A40, 4);
        PADA(A01, 5); PADA(A11, 6); PADA(A21, 7); PADA(A31, 8); PADA(A41, 9);
        PADA(A02, 10); PADA(A12, 11); PADA(A22, 12); PADA(A32, 13); PADA(A42, 14);
        PADA(A03, 15); PADA(A13, 16); PADA(A23, 17); PADA(A33, 18); PADA(A43, 19);
        PADA(A04, 20); PADA(A14, 21); PADA(A24, 22); PADA(A34, 23); PADA(A44, 24);

        A13 ^= 0x8000000000000000ul;

        K_ROUND(0x0000000000000001ul);
        K_ROUND(0x0000000000008082ul);
        K_ROUND(0x800000000000808Aul);
        K_ROUND(0x8000000080008000ul);
        K_ROUND(0x000000000000808Bul);
        K_ROUND(0x0000000080000001ul);
        K_ROUND(0x8000000080008081ul);
        K_ROUND(0x8000000000008009ul);
        K_ROUND(0x000000000000008Aul);
        K_ROUND(0x0000000000000088ul);
        K_ROUND(0x0000000080008009ul);
        K_ROUND(0x000000008000000Aul);
        K_ROUND(0x000000008000808Bul);
        K_ROUND(0x800000000000008Bul);
        K_ROUND(0x8000000000008089ul);
        K_ROUND(0x8000000000008003ul);
        K_ROUND(0x8000000000008002ul);
        K_ROUND(0x8000000000000080ul);
        K_ROUND(0x000000000000800Aul);
        K_ROUND(0x800000008000000Aul);
        K_ROUND(0x8000000080008081ul);
        K_ROUND(0x8000000000008080ul);
        K_ROUND(0x0000000080000001ul);
        K_ROUND(0x8000000080008008ul);
    }

    STORE(A00, 0); STORE(A10, 1); STORE(A20, 2); STORE(A30, 3); STORE(A40, 4);
    STORE(A01, 5); STORE(A11, 6); STORE(A21, 7); STORE(A31, 8); STORE(A41, 9);
    STORE(A02, 10); STORE(A12, 11); STORE(A22, 12); STORE(A32, 13); STORE(A42, 14);
    STORE(A03, 15); STORE(A13, 16);
}
```

Result of previous attempt:
          w16_C64K: INCORRECT (bit_exact=131072)
  fail_reason: correctness failed at size w16_C64K: bit_exact=131072

## Current best (incumbent)

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

Incumbent result:
          w16_C64K: correct, 42.66 ms, 91.4 Gbitops/s (u64) (8.1% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 172.62 ms, 90.4 Gbitops/s (u64) (8.0% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 679.52 ms, 91.8 Gbitops/s (u64) (8.2% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.0811

## History

- iter  0: compile=OK | correct=True | score=0.03917601338347657
- iter  1: compile=OK | correct=True | score=0.08108350649983091
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
