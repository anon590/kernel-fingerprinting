To improve performance over the incumbent and resolve the previous correctness failure, this solution completely scalarizes the Keccak-f1600 state matrix into 25 independent local registers (`A00` through `A44`) rather than relying on arrays. Previous array-based explicit unrolling likely suffered from the Metal compiler incorrectly handling large unrolled array states with dynamic branch dependencies, leading to correctness bugs or register spilling. 

By explicitly declaring all 25 variables and using a macro (`K_ROUND`) to fully inline the 24 permutations, we completely eliminate memory indexing overhead. Padding logic is applied branchlessly via `select` (compiling natively to `csel` conditional-select instructions). This strictly forces all data to remain within the rich scalar register file of the Apple GPU.

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