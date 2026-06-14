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

constexpr uint morton_compress(uint x) {
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

constexpr uint2 to_interleaved(ulong V) {
    uint lo = (uint)V;
    uint hi = (uint)(V >> 32);
    uint e = morton_compress(lo) | (morton_compress(hi) << 16);
    uint o = morton_compress(lo >> 1) | (morton_compress(hi >> 1) << 16);
    return uint2(e, o);
}

constexpr uint morton_expand(uint x) {
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

constexpr ulong from_interleaved(uint2 V) {
    uint e = V.x;
    uint o = V.y;
    uint lo = morton_expand(e) | (morton_expand(o) << 1);
    uint hi = morton_expand(e >> 16) | (morton_expand(o >> 16) << 1);
    return ((ulong)hi << 32) | lo;
}

#define ROTL32(x, k) (((k) == 0) ? (x) : (((x) << (k)) | ((x) >> (32 - (k)))))

#define ROTL_UINT2(V, k) \
    (((k) & 1) ? uint2(ROTL32((V).y, (((k) / 2) + 1) & 31), ROTL32((V).x, ((k) / 2) & 31)) \
               : uint2(ROTL32((V).x, ((k) / 2) & 31),     ROTL32((V).y, ((k) / 2) & 31)))

#define K_ROUND(rc) \
    do { \
        constexpr uint2 RC = to_interleaved(rc); \
        \
        uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
        uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
        uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
        uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
        uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
        \
        uint2 D0 = C4 ^ ROTL_UINT2(C1, 1); \
        uint2 D1 = C0 ^ ROTL_UINT2(C2, 1); \
        uint2 D2 = C1 ^ ROTL_UINT2(C3, 1); \
        uint2 D3 = C2 ^ ROTL_UINT2(C4, 1); \
        uint2 D4 = C3 ^ ROTL_UINT2(C0, 1); \
        \
        uint2 B00 = A00 ^ D0; \
        uint2 B10 = ROTL_UINT2(A11 ^ D1, 44); \
        uint2 B20 = ROTL_UINT2(A22 ^ D2, 43); \
        uint2 B30 = ROTL_UINT2(A33 ^ D3, 21); \
        uint2 B40 = ROTL_UINT2(A44 ^ D4, 14); \
        \
        uint2 B01 = ROTL_UINT2(A30 ^ D3, 28); \
        uint2 B11 = ROTL_UINT2(A41 ^ D4, 20); \
        uint2 B21 = ROTL_UINT2(A02 ^ D0,  3); \
        uint2 B31 = ROTL_UINT2(A13 ^ D1, 45); \
        uint2 B41 = ROTL_UINT2(A24 ^ D2, 61); \
        \
        uint2 B02 = ROTL_UINT2(A10 ^ D1,  1); \
        uint2 B12 = ROTL_UINT2(A21 ^ D2,  6); \
        uint2 B22 = ROTL_UINT2(A32 ^ D3, 25); \
        uint2 B32 = ROTL_UINT2(A43 ^ D4,  8); \
        uint2 B42 = ROTL_UINT2(A04 ^ D0, 18); \
        \
        uint2 B03 = ROTL_UINT2(A40 ^ D4, 27); \
        uint2 B13 = ROTL_UINT2(A01 ^ D0, 36); \
        uint2 B23 = ROTL_UINT2(A12 ^ D1, 10); \
        uint2 B33 = ROTL_UINT2(A23 ^ D2, 15); \
        uint2 B43 = ROTL_UINT2(A34 ^ D3, 56); \
        \
        uint2 B04 = ROTL_UINT2(A20 ^ D2, 62); \
        uint2 B14 = ROTL_UINT2(A31 ^ D3, 55); \
        uint2 B24 = ROTL_UINT2(A42 ^ D4, 39); \
        uint2 B34 = ROTL_UINT2(A03 ^ D0, 41); \
        uint2 B44 = ROTL_UINT2(A14 ^ D1,  2); \
        \
        A00 = B00 ^ (~B10 & B20) ^ RC; \
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

    uint2 A00 = uint2(0); uint2 A10 = uint2(0); uint2 A20 = uint2(0); uint2 A30 = uint2(0); uint2 A40 = uint2(0);
    uint2 A01 = uint2(0); uint2 A11 = uint2(0); uint2 A21 = uint2(0); uint2 A31 = uint2(0); uint2 A41 = uint2(0);
    uint2 A02 = uint2(0); uint2 A12 = uint2(0); uint2 A22 = uint2(0); uint2 A32 = uint2(0); uint2 A42 = uint2(0);
    uint2 A03 = uint2(0); uint2 A13 = uint2(0); uint2 A23 = uint2(0); uint2 A33 = uint2(0); uint2 A43 = uint2(0);
    uint2 A04 = uint2(0); uint2 A14 = uint2(0); uint2 A24 = uint2(0); uint2 A34 = uint2(0); uint2 A44 = uint2(0);

    if (0 < n_lanes) A00 = to_interleaved(seeds[base +  0]);
    if (1 < n_lanes) A10 = to_interleaved(seeds[base +  1]);
    if (2 < n_lanes) A20 = to_interleaved(seeds[base +  2]);
    if (3 < n_lanes) A30 = to_interleaved(seeds[base +  3]);
    if (4 < n_lanes) A40 = to_interleaved(seeds[base +  4]);

    if (5 < n_lanes) A01 = to_interleaved(seeds[base +  5]);
    if (6 < n_lanes) A11 = to_interleaved(seeds[base +  6]);
    if (7 < n_lanes) A21 = to_interleaved(seeds[base +  7]);
    if (8 < n_lanes) A31 = to_interleaved(seeds[base +  8]);
    if (9 < n_lanes) A41 = to_interleaved(seeds[base +  9]);

    if (10 < n_lanes) A02 = to_interleaved(seeds[base + 10]);
    if (11 < n_lanes) A12 = to_interleaved(seeds[base + 11]);
    if (12 < n_lanes) A22 = to_interleaved(seeds[base + 12]);
    if (13 < n_lanes) A32 = to_interleaved(seeds[base + 13]);
    if (14 < n_lanes) A42 = to_interleaved(seeds[base + 14]);

    if (15 < n_lanes) A03 = to_interleaved(seeds[base + 15]);

    constexpr uint2 PAD06 = to_interleaved(0x06ul);
    constexpr uint2 PAD16_16 = to_interleaved(0x8000000000000006ul);
    constexpr uint2 PAD16_OTHERS = to_interleaved(0x8000000000000000ul);
    uint2 V_PAD16 = (n_lanes == 16) ? PAD16_16 : PAD16_OTHERS;

    for (uint step = 0u; step < w; ++step) {
        
        A00 = select(uint2(0), A00, n_lanes >  0);
        A10 = select(uint2(0), A10, n_lanes >  1);
        A20 = select(uint2(0), A20, n_lanes >  2);
        A30 = select(uint2(0), A30, n_lanes >  3);
        A40 = select(uint2(0), A40, n_lanes >  4);

        A01 = select(uint2(0), A01, n_lanes >  5);
        A11 = select(uint2(0), A11, n_lanes >  6);
        A21 = select(uint2(0), A21, n_lanes >  7);
        A31 = select(uint2(0), A31, n_lanes >  8);
        A41 = select(uint2(0), A41, n_lanes >  9);

        A02 = select(uint2(0), A02, n_lanes > 10);
        A12 = select(uint2(0), A12, n_lanes > 11);
        A22 = select(uint2(0), A22, n_lanes > 12);
        A32 = select(uint2(0), A32, n_lanes > 13);
        A42 = select(uint2(0), A42, n_lanes > 14);

        A03 = select(uint2(0), A03, n_lanes > 15);

        switch (n_lanes) {
            case  0: A00 = PAD06; break;
            case  1: A10 = PAD06; break;
            case  2: A20 = PAD06; break;
            case  3: A30 = PAD06; break;
            case  4: A40 = PAD06; break;
            case  5: A01 = PAD06; break;
            case  6: A11 = PAD06; break;
            case  7: A21 = PAD06; break;
            case  8: A31 = PAD06; break;
            case  9: A41 = PAD06; break;
            case 10: A02 = PAD06; break;
            case 11: A12 = PAD06; break;
            case 12: A22 = PAD06; break;
            case 13: A32 = PAD06; break;
            case 14: A42 = PAD06; break;
            case 15: A03 = PAD06; break;
        }

        A13 = V_PAD16;
        
        A23 = uint2(0); A33 = uint2(0); A43 = uint2(0);
        A04 = uint2(0); A14 = uint2(0); A24 = uint2(0);
        A34 = uint2(0); A44 = uint2(0);

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

    if (0 < n_lanes) tips[base +  0] = from_interleaved(A00);
    if (1 < n_lanes) tips[base +  1] = from_interleaved(A10);
    if (2 < n_lanes) tips[base +  2] = from_interleaved(A20);
    if (3 < n_lanes) tips[base +  3] = from_interleaved(A30);
    if (4 < n_lanes) tips[base +  4] = from_interleaved(A40);

    if (5 < n_lanes) tips[base +  5] = from_interleaved(A01);
    if (6 < n_lanes) tips[base +  6] = from_interleaved(A11);
    if (7 < n_lanes) tips[base +  7] = from_interleaved(A21);
    if (8 < n_lanes) tips[base +  8] = from_interleaved(A31);
    if (9 < n_lanes) tips[base +  9] = from_interleaved(A41);

    if (10 < n_lanes) tips[base + 10] = from_interleaved(A02);
    if (11 < n_lanes) tips[base + 11] = from_interleaved(A12);
    if (12 < n_lanes) tips[base + 12] = from_interleaved(A22);
    if (13 < n_lanes) tips[base + 13] = from_interleaved(A32);
    if (14 < n_lanes) tips[base + 14] = from_interleaved(A42);

    if (15 < n_lanes) tips[base + 15] = from_interleaved(A03);
}
```

Result of previous attempt:
          w16_C64K: correct, 4.60 ms, 848.5 Gbitops/s (u64) (75.4% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 17.55 ms, 889.2 Gbitops/s (u64) (79.0% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 70.50 ms, 885.3 Gbitops/s (u64) (78.7% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.7770

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr uint morton_compress(uint x) {
    x &= 0x55555555;
    x = (x ^ (x >> 1)) & 0x33333333;
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

constexpr uint2 to_interleaved(ulong V) {
    uint lo = (uint)V;
    uint hi = (uint)(V >> 32);
    uint e = morton_compress(lo) | (morton_compress(hi) << 16);
    uint o = morton_compress(lo >> 1) | (morton_compress(hi >> 1) << 16);
    return uint2(e, o);
}

constexpr uint morton_expand(uint x) {
    x &= 0x0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f;
    x = (x ^ (x << 2)) & 0x33333333;
    x = (x ^ (x << 1)) & 0x55555555;
    return x;
}

constexpr ulong from_interleaved(uint e, uint o) {
    uint lo = morton_expand(e) | (morton_expand(o) << 1);
    uint hi = morton_expand(e >> 16) | (morton_expand(o >> 16) << 1);
    return ((ulong)hi << 32) | lo;
}

#define ROTL32(x, k) (((k) == 0) ? (x) : (((x) << (k)) | ((x) >> (32 - (k)))))

#define R_E(E, O, k) (((k) & 1) ? ROTL32(O, (((k) / 2) + 1) & 31) : ROTL32(E, ((k) / 2) & 31))
#define R_O(E, O, k) (((k) & 1) ? ROTL32(E, ((k) / 2) & 31)       : ROTL32(O, ((k) / 2) & 31))

#define K_ROUND(rc) \
    do { \
        constexpr uint2 RC = to_interleaved(rc); \
        uint rc_E = RC.x; \
        uint rc_O = RC.y; \
        \
        uint C0_E = A00_E ^ A01_E ^ A02_E ^ A03_E ^ A04_E; \
        uint C0_O = A00_O ^ A01_O ^ A02_O ^ A03_O ^ A04_O; \
        uint C1_E = A10_E ^ A11_E ^ A12_E ^ A13_E ^ A14_E; \
        uint C1_O = A10_O ^ A11_O ^ A12_O ^ A13_O ^ A14_O; \
        uint C2_E = A20_E ^ A21_E ^ A22_E ^ A23_E ^ A24_E; \
        uint C2_O = A20_O ^ A21_O ^ A22_O ^ A23_O ^ A24_O; \
        uint C3_E = A30_E ^ A31_E ^ A32_E ^ A33_E ^ A34_E; \
        uint C3_O = A30_O ^ A31_O ^ A32_O ^ A33_O ^ A34_O; \
        uint C4_E = A40_E ^ A41_E ^ A42_E ^ A43_E ^ A44_E; \
        uint C4_O = A40_O ^ A41_O ^ A42_O ^ A43_O ^ A44_O; \
        \
        uint D0_E = C4_E ^ R_E(C1_E, C1_O, 1); \
        uint D0_O = C4_O ^ R_O(C1_E, C1_O, 1); \
        uint D1_E = C0_E ^ R_E(C2_E, C2_O, 1); \
        uint D1_O = C0_O ^ R_O(C2_E, C2_O, 1); \
        uint D2_E = C1_E ^ R_E(C3_E, C3_O, 1); \
        uint D2_O = C1_O ^ R_O(C3_E, C3_O, 1); \
        uint D3_E = C2_E ^ R_E(C4_E, C4_O, 1); \
        uint D3_O = C2_O ^ R_O(C4_E, C4_O, 1); \
        uint D4_E = C3_E ^ R_E(C0_E, C0_O, 1); \
        uint D4_O = C3_O ^ R_O(C0_E, C0_O, 1); \
        \
        uint B00_E = A00_E ^ D0_E; \
        uint B00_O = A00_O ^ D0_O; \
        \
        uint T_E, T_O; \
        T_E = A11_E ^ D1_E; T_O = A11_O ^ D1_O; \
        uint B10_E = R_E(T_E, T_O, 44); uint B10_O = R_O(T_E, T_O, 44); \
        T_E = A22_E ^ D2_E; T_O = A22_O ^ D2_O; \
        uint B20_E = R_E(T_E, T_O, 43); uint B20_O = R_O(T_E, T_O, 43); \
        T_E = A33_E ^ D3_E; T_O = A33_O ^ D3_O; \
        uint B30_E = R_E(T_E, T_O, 21); uint B30_O = R_O(T_E, T_O, 21); \
        T_E = A44_E ^ D4_E; T_O = A44_O ^ D4_O; \
        uint B40_E = R_E(T_E, T_O, 14); uint B40_O = R_O(T_E, T_O, 14); \
        \
        T_E = A30_E ^ D3_E; T_O = A30_O ^ D3_O; \
        uint B01_E = R_E(T_E, T_O, 28); uint B01_O = R_O(T_E, T_O, 28); \
        T_E = A41_E ^ D4_E; T_O = A41_O ^ D4_O; \
        uint B11_E = R_E(T_E, T_O, 20); uint B11_O = R_O(T_E, T_O, 20); \
        T_E = A02_E ^ D0_E; T_O = A02_O ^ D0_O; \
        uint B21_E = R_E(T_E, T_O,  3); uint B21_O = R_O(T_E, T_O,  3); \
        T_E = A13_E ^ D1_E; T_O = A13_O ^ D1_O; \
        uint B31_E = R_E(T_E, T_O, 45); uint B31_O = R_O(T_E, T_O, 45); \
        T_E = A24_E ^ D2_E; T_O = A24_O ^ D2_O; \
        uint B41_E = R_E(T_E, T_O, 61); uint B41_O = R_O(T_E, T_O, 61); \
        \
        T_E = A10_E ^ D1_E; T_O = A10_O ^ D1_O; \
        uint B02_E = R_E(T_E, T_O,  1); uint B02_O = R_O(T_E, T_O,  1); \
        T_E = A21_E ^ D2_E; T_O = A21_O ^ D2_O; \
        uint B12_E = R_E(T_E, T_O,  6); uint B12_O = R_O(T_E, T_O,  6); \
        T_E = A32_E ^ D3_E; T_O = A32_O ^ D3_O; \
        uint B22_E = R_E(T_E, T_O, 25); uint B22_O = R_O(T_E, T_O, 25); \
        T_E = A43_E ^ D4_E; T_O = A43_O ^ D4_O; \
        uint B32_E = R_E(T_E, T_O,  8); uint B32_O = R_O(T_E, T_O,  8); \
        T_E = A04_E ^ D0_E; T_O = A04_O ^ D0_O; \
        uint B42_E = R_E(T_E, T_O, 18); uint B42_O = R_O(T_E, T_O, 18); \
        \
        T_E = A40_E ^ D4_E; T_O = A40_O ^ D4_O; \
        uint B03_E = R_E(T_E, T_O, 27); uint B03_O = R_O(T_E, T_O, 27); \
        T_E = A01_E ^ D0_E; T_O = A01_O ^ D0_O; \
        uint B13_E = R_E(T_E, T_O, 36); uint B13_O = R_O(T_E, T_O, 36); \
        T_E = A12_E ^ D1_E; T_O = A12_O ^ D1_O; \
        uint B23_E = R_E(T_E, T_O, 10); uint B23_O = R_O(T_E, T_O, 10); \
        T_E = A23_E ^ D2_E; T_O = A23_O ^ D2_O; \
        uint B33_E = R_E(T_E, T_O, 15); uint B33_O = R_O(T_E, T_O, 15); \
        T_E = A34_E ^ D3_E; T_O = A34_O ^ D3_O; \
        uint B43_E = R_E(T_E, T_O, 56); uint B43_O = R_O(T_E, T_O, 56); \
        \
        T_E = A20_E ^ D2_E; T_O = A20_O ^ D2_O; \
        uint B04_E = R_E(T_E, T_O, 62); uint B04_O = R_O(T_E, T_O, 62); \
        T_E = A31_E ^ D3_E; T_O = A31_O ^ D3_O; \
        uint B14_E = R_E(T_E, T_O, 55); uint B14_O = R_O(T_E, T_O, 55); \
        T_E = A42_E ^ D4_E; T_O = A42_O ^ D4_O; \
        uint B24_E = R_E(T_E, T_O, 39); uint B24_O = R_O(T_E, T_O, 39); \
        T_E = A03_E ^ D0_E; T_O = A03_O ^ D0_O; \
        uint B34_E = R_E(T_E, T_O, 41); uint B34_O = R_O(T_E, T_O, 41); \
        T_E = A14_E ^ D1_E; T_O = A14_O ^ D1_O; \
        uint B44_E = R_E(T_E, T_O,  2); uint B44_O = R_O(T_E, T_O,  2); \
        \
        A00_E = B00_E ^ (~B10_E & B20_E); A00_E ^= rc_E; \
        A00_O = B00_O ^ (~B10_O & B20_O); A00_O ^= rc_O; \
        A10_E = B10_E ^ (~B20_E & B30_E); A10_O = B10_O ^ (~B20_O & B30_O); \
        A20_E = B20_E ^ (~B30_E & B40_E); A20_O = B20_O ^ (~B30_O & B40_O); \
        A30_E = B30_E ^ (~B40_E & B00_E); A30_O = B30_O ^ (~B40_O & B00_O); \
        A40_E = B40_E ^ (~B00_E & B10_E); A40_O = B40_O ^ (~B00_O & B10_O); \
        \
        A01_E = B01_E ^ (~B11_E & B21_E); A01_O = B01_O ^ (~B11_O & B21_O); \
        A11_E = B11_E ^ (~B21_E & B31_E); A11_O = B11_O ^ (~B21_O & B31_O); \
        A21_E = B21_E ^ (~B31_E & B41_E); A21_O = B21_O ^ (~B31_O & B41_O); \
        A31_E = B31_E ^ (~B41_E & B01_E); A31_O = B31_O ^ (~B41_O & B01_O); \
        A41_E = B41_E ^ (~B01_E & B11_E); A41_O = B41_O ^ (~B01_O & B11_O); \
        \
        A02_E = B02_E ^ (~B12_E & B22_E); A02_O = B02_O ^ (~B12_O & B22_O); \
        A12_E = B12_E ^ (~B22_E & B32_E); A12_O = B12_O ^ (~B22_O & B32_O); \
        A22_E = B22_E ^ (~B32_E & B42_E); A22_O = B22_O ^ (~B32_O & B42_O); \
        A32_E = B32_E ^ (~B42_E & B02_E); A32_O = B32_O ^ (~B42_O & B02_O); \
        A42_E = B42_E ^ (~B02_E & B12_E); A42_O = B42_O ^ (~B02_O & B12_O); \
        \
        A03_E = B03_E ^ (~B13_E & B23_E); A03_O = B03_O ^ (~B13_O & B23_O); \
        A13_E = B13_E ^ (~B23_E & B33_E); A13_O = B13_O ^ (~B23_O & B33_O); \
        A23_E = B23_E ^ (~B33_E & B43_E); A23_O = B23_O ^ (~B33_O & B43_O); \
        A33_E = B33_E ^ (~B43_E & B03_E); A33_O = B33_O ^ (~B43_O & B03_O); \
        A43_E = B43_E ^ (~B03_E & B13_E); A43_O = B43_O ^ (~B03_O & B13_O); \
        \
        A04_E = B04_E ^ (~B14_E & B24_E); A04_O = B04_O ^ (~B14_O & B24_O); \
        A14_E = B14_E ^ (~B24_E & B34_E); A14_O = B14_O ^ (~B24_O & B34_O); \
        A24_E = B24_E ^ (~B34_E & B44_E); A24_O = B24_O ^ (~B34_O & B44_O); \
        A34_E = B34_E ^ (~B44_E & B04_E); A34_O = B34_O ^ (~B44_O & B04_O); \
        A44_E = B44_E ^ (~B04_E & B14_E); A44_O = B44_O ^ (~B04_O & B14_O); \
    } while (0)

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

    uint A00_E = 0, A00_O = 0; uint A10_E = 0, A10_O = 0; uint A20_E = 0, A20_O = 0; uint A30_E = 0, A30_O = 0; uint A40_E = 0, A40_O = 0;
    uint A01_E = 0, A01_O = 0; uint A11_E = 0, A11_O = 0; uint A21_E = 0, A21_O = 0; uint A31_E = 0, A31_O = 0; uint A41_E = 0, A41_O = 0;
    uint A02_E = 0, A02_O = 0; uint A12_E = 0, A12_O = 0; uint A22_E = 0, A22_O = 0; uint A32_E = 0, A32_O = 0; uint A42_E = 0, A42_O = 0;
    uint A03_E = 0, A03_O = 0; uint A13_E = 0, A13_O = 0; uint A23_E = 0, A23_O = 0; uint A33_E = 0, A33_O = 0; uint A43_E = 0, A43_O = 0;
    uint A04_E = 0, A04_O = 0; uint A14_E = 0, A14_O = 0; uint A24_E = 0, A24_O = 0; uint A34_E = 0, A34_O = 0; uint A44_E = 0, A44_O = 0;

    if (0 < n_lanes) { uint2 v = to_interleaved(seeds[base + 0]); A00_E = v.x; A00_O = v.y; }
    if (1 < n_lanes) { uint2 v = to_interleaved(seeds[base + 1]); A10_E = v.x; A10_O = v.y; }
    if (2 < n_lanes) { uint2 v = to_interleaved(seeds[base + 2]); A20_E = v.x; A20_O = v.y; }
    if (3 < n_lanes) { uint2 v = to_interleaved(seeds[base + 3]); A30_E = v.x; A30_O = v.y; }
    if (4 < n_lanes) { uint2 v = to_interleaved(seeds[base + 4]); A40_E = v.x; A40_O = v.y; }

    if (5 < n_lanes) { uint2 v = to_interleaved(seeds[base + 5]); A01_E = v.x; A01_O = v.y; }
    if (6 < n_lanes) { uint2 v = to_interleaved(seeds[base + 6]); A11_E = v.x; A11_O = v.y; }
    if (7 < n_lanes) { uint2 v = to_interleaved(seeds[base + 7]); A21_E = v.x; A21_O = v.y; }
    if (8 < n_lanes) { uint2 v = to_interleaved(seeds[base + 8]); A31_E = v.x; A31_O = v.y; }
    if (9 < n_lanes) { uint2 v = to_interleaved(seeds[base + 9]); A41_E = v.x; A41_O = v.y; }

    if (10 < n_lanes) { uint2 v = to_interleaved(seeds[base + 10]); A02_E = v.x; A02_O = v.y; }
    if (11 < n_lanes) { uint2 v = to_interleaved(seeds[base + 11]); A12_E = v.x; A12_O = v.y; }
    if (12 < n_lanes) { uint2 v = to_interleaved(seeds[base + 12]); A22_E = v.x; A22_O = v.y; }
    if (13 < n_lanes) { uint2 v = to_interleaved(seeds[base + 13]); A32_E = v.x; A32_O = v.y; }
    if (14 < n_lanes) { uint2 v = to_interleaved(seeds[base + 14]); A42_E = v.x; A42_O = v.y; }

    if (15 < n_lanes) { uint2 v = to_interleaved(seeds[base + 15]); A03_E = v.x; A03_O = v.y; }

    constexpr uint2 PAD06 = to_interleaved(0x06ul);
    uint pad06_E = PAD06.x;
    uint pad06_O = PAD06.y;

    constexpr uint2 PAD16_16 = to_interleaved(0x8000000000000006ul);
    constexpr uint2 PAD16_OTHERS = to_interleaved(0x8000000000000000ul);
    uint pad16_E = (n_lanes == 16) ? PAD16_16.x : PAD16_OTHERS.x;
    uint pad16_O = (n_lanes == 16) ? PAD16_16.y : PAD16_OTHERS.y;

    uint M00 = (n_lanes > 0) ? ~0u : 0u; uint P00_E = (n_lanes == 0) ? pad06_E : 0u; uint P00_O = (n_lanes == 0) ? pad06_O : 0u;
    uint M10 = (n_lanes > 1) ? ~0u : 0u; uint P10_E = (n_lanes == 1) ? pad06_E : 0u; uint P10_O = (n_lanes == 1) ? pad06_O : 0u;
    uint M20 = (n_lanes > 2) ? ~0u : 0u; uint P20_E = (n_lanes == 2) ? pad06_E : 0u; uint P20_O = (n_lanes == 2) ? pad06_O : 0u;
    uint M30 = (n_lanes > 3) ? ~0u : 0u; uint P30_E = (n_lanes == 3) ? pad06_E : 0u; uint P30_O = (n_lanes == 3) ? pad06_O : 0u;
    uint M40 = (n_lanes > 4) ? ~0u : 0u; uint P40_E = (n_lanes == 4) ? pad06_E : 0u; uint P40_O = (n_lanes == 4) ? pad06_O : 0u;

    uint M01 = (n_lanes > 5) ? ~0u : 0u; uint P01_E = (n_lanes == 5) ? pad06_E : 0u; uint P01_O = (n_lanes == 5) ? pad06_O : 0u;
    uint M11 = (n_lanes > 6) ? ~0u : 0u; uint P11_E = (n_lanes == 6) ? pad06_E : 0u; uint P11_O = (n_lanes == 6) ? pad06_O : 0u;
    uint M21 = (n_lanes > 7) ? ~0u : 0u; uint P21_E = (n_lanes == 7) ? pad06_E : 0u; uint P21_O = (n_lanes == 7) ? pad06_O : 0u;
    uint M31 = (n_lanes > 8) ? ~0u : 0u; uint P31_E = (n_lanes == 8) ? pad06_E : 0u; uint P31_O = (n_lanes == 8) ? pad06_O : 0u;
    uint M41 = (n_lanes > 9) ? ~0u : 0u; uint P41_E = (n_lanes == 9) ? pad06_E : 0u; uint P41_O = (n_lanes == 9) ? pad06_O : 0u;

    uint M02 = (n_lanes > 10) ? ~0u : 0u; uint P02_E = (n_lanes == 10) ? pad06_E : 0u; uint P02_O = (n_lanes == 10) ? pad06_O : 0u;
    uint M12 = (n_lanes > 11) ? ~0u : 0u; uint P12_E = (n_lanes == 11) ? pad06_E : 0u; uint P12_O = (n_lanes == 11) ? pad06_O : 0u;
    uint M22 = (n_lanes > 12) ? ~0u : 0u; uint P22_E = (n_lanes == 12) ? pad06_E : 0u; uint P22_O = (n_lanes == 12) ? pad06_O : 0u;
    uint M32 = (n_lanes > 13) ? ~0u : 0u; uint P32_E = (n_lanes == 13) ? pad06_E : 0u; uint P32_O = (n_lanes == 13) ? pad06_O : 0u;
    uint M42 = (n_lanes > 14) ? ~0u : 0u; uint P42_E = (n_lanes == 14) ? pad06_E : 0u; uint P42_O = (n_lanes == 14) ? pad06_O : 0u;

    uint M03 = (n_lanes > 15) ? ~0u : 0u; uint P03_E = (n_lanes == 15) ? pad06_E : 0u; uint P03_O = (n_lanes == 15) ? pad06_O : 0u;

    for (uint step = 0u; step < w; ++step) {
        
        A00_E = (A00_E & M00) | P00_E; A00_O = (A00_O & M00) | P00_O;
        A10_E = (A10_E & M10) | P10_E; A10_O = (A10_O & M10) | P10_O;
        A20_E = (A20_E & M20) | P20_E; A20_O = (A20_O & M20) | P20_O;
        A30_E = (A30_E & M30) | P30_E; A30_O = (A30_O & M30) | P30_O;
        A40_E = (A40_E & M40) | P40_E; A40_O = (A40_O & M40) | P40_O;

        A01_E = (A01_E & M01) | P01_E; A01_O = (A01_O & M01) | P01_O;
        A11_E = (A11_E & M11) | P11_E; A11_O = (A11_O & M11) | P11_O;
        A21_E = (A21_E & M21) | P21_E; A21_O = (A21_O & M21) | P21_O;
        A31_E = (A31_E & M31) | P31_E; A31_O = (A31_O & M31) | P31_O;
        A41_E = (A41_E & M41) | P41_E; A41_O = (A41_O & M41) | P41_O;

        A02_E = (A02_E & M02) | P02_E; A02_O = (A02_O & M02) | P02_O;
        A12_E = (A12_E & M12) | P12_E; A12_O = (A12_O & M12) | P12_O;
        A22_E = (A22_E & M22) | P22_E; A22_O = (A22_O & M22) | P22_O;
        A32_E = (A32_E & M32) | P32_E; A32_O = (A32_O & M32) | P32_O;
        A42_E = (A42_E & M42) | P42_E; A42_O = (A42_O & M42) | P42_O;

        A03_E = (A03_E & M03) | P03_E; A03_O = (A03_O & M03) | P03_O;

        A13_E = pad16_E; A13_O = pad16_O;
        
        A23_E = 0; A23_O = 0;
        A33_E = 0; A33_O = 0;
        A43_E = 0; A43_O = 0;
        A04_E = 0; A04_O = 0;
        A14_E = 0; A14_O = 0;
        A24_E = 0; A24_O = 0;
        A34_E = 0; A34_O = 0;
        A44_E = 0; A44_O = 0;

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

    if (0 < n_lanes) tips[base + 0] = from_interleaved(A00_E, A00_O);
    if (1 < n_lanes) tips[base + 1] = from_interleaved(A10_E, A10_O);
    if (2 < n_lanes) tips[base + 2] = from_interleaved(A20_E, A20_O);
    if (3 < n_lanes) tips[base + 3] = from_interleaved(A30_E, A30_O);
    if (4 < n_lanes) tips[base + 4] = from_interleaved(A40_E, A40_O);

    if (5 < n_lanes) tips[base + 5] = from_interleaved(A01_E, A01_O);
    if (6 < n_lanes) tips[base + 6] = from_interleaved(A11_E, A11_O);
    if (7 < n_lanes) tips[base + 7] = from_interleaved(A21_E, A21_O);
    if (8 < n_lanes) tips[base + 8] = from_interleaved(A31_E, A31_O);
    if (9 < n_lanes) tips[base + 9] = from_interleaved(A41_E, A41_O);

    if (10 < n_lanes) tips[base + 10] = from_interleaved(A02_E, A02_O);
    if (11 < n_lanes) tips[base + 11] = from_interleaved(A12_E, A12_O);
    if (12 < n_lanes) tips[base + 12] = from_interleaved(A22_E, A22_O);
    if (13 < n_lanes) tips[base + 13] = from_interleaved(A32_E, A32_O);
    if (14 < n_lanes) tips[base + 14] = from_interleaved(A42_E, A42_O);

    if (15 < n_lanes) tips[base + 15] = from_interleaved(A03_E, A03_O);
}
```

Incumbent result:
          w16_C64K: correct, 4.54 ms, 858.4 Gbitops/s (u64) (76.3% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 17.28 ms, 902.8 Gbitops/s (u64) (80.3% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 70.13 ms, 889.9 Gbitops/s (u64) (79.1% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.7853

## History

- iter  1: compile=OK | correct=True | score=0.08108350649983091
- iter  2: compile=OK | correct=False | score=N/A
- iter  3: compile=OK | correct=False | score=N/A
- iter  4: compile=OK | correct=True | score=0.6089068857986767
- iter  5: compile=OK | correct=True | score=0.7853289798086907
- iter  6: compile=OK | correct=True | score=0.7677162525388368
- iter  7: compile=OK | correct=True | score=0.6065688367450467
- iter  8: compile=OK | correct=True | score=0.7770153272833022

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
