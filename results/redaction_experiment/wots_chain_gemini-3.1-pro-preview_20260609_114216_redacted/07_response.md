We switch the Keccak state from a 32-bit interleaved representation to a native 64-bit `ulong` representation. Since Apple Silicon GPUs feature full-throughput 64-bit bitwise ALUs, this eliminates the expensive `morton_compress`/`expand` steps at the start and end of the chain, and halves the number of logical operations (XOR/AND) per round. We also replace the incumbent's `A = (A & M) | P` arithmetic padding with `A = (n_lanes > i) ? A : P`. This compiles to branchless `csel` instructions, reducing loop instructions and halving the register pressure required for mask caching.

```metal
#include <metal_stdlib>
using namespace metal;

#define ROTL64(x, y) (((y) == 0) ? (x) : (((x) << (y)) | ((x) >> (64 - (y)))))

#define K_ROUND(rc) \
    do { \
        ulong C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
        ulong C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
        ulong C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
        ulong C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
        ulong C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
        \
        ulong D0 = C4 ^ ROTL64(C1, 1); \
        ulong D1 = C0 ^ ROTL64(C2, 1); \
        ulong D2 = C1 ^ ROTL64(C3, 1); \
        ulong D3 = C2 ^ ROTL64(C4, 1); \
        ulong D4 = C3 ^ ROTL64(C0, 1); \
        \
        ulong B00 = A00 ^ D0; \
        ulong B10 = ROTL64(A11 ^ D1, 44); \
        ulong B20 = ROTL64(A22 ^ D2, 43); \
        ulong B30 = ROTL64(A33 ^ D3, 21); \
        ulong B40 = ROTL64(A44 ^ D4, 14); \
        \
        ulong B01 = ROTL64(A30 ^ D3, 28); \
        ulong B11 = ROTL64(A41 ^ D4, 20); \
        ulong B21 = ROTL64(A02 ^ D0, 3); \
        ulong B31 = ROTL64(A13 ^ D1, 45); \
        ulong B41 = ROTL64(A24 ^ D2, 61); \
        \
        ulong B02 = ROTL64(A10 ^ D1, 1); \
        ulong B12 = ROTL64(A21 ^ D2, 6); \
        ulong B22 = ROTL64(A32 ^ D3, 25); \
        ulong B32 = ROTL64(A43 ^ D4, 8); \
        ulong B42 = ROTL64(A04 ^ D0, 18); \
        \
        ulong B03 = ROTL64(A40 ^ D4, 27); \
        ulong B13 = ROTL64(A01 ^ D0, 36); \
        ulong B23 = ROTL64(A12 ^ D1, 10); \
        ulong B33 = ROTL64(A23 ^ D2, 15); \
        ulong B43 = ROTL64(A34 ^ D3, 56); \
        \
        ulong B04 = ROTL64(A20 ^ D2, 62); \
        ulong B14 = ROTL64(A31 ^ D3, 55); \
        ulong B24 = ROTL64(A42 ^ D4, 39); \
        ulong B34 = ROTL64(A03 ^ D0, 41); \
        ulong B44 = ROTL64(A14 ^ D1, 2); \
        \
        A00 = B00 ^ (~B10 & B20) ^ rc; \
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

    ulong A00 = 0, A10 = 0, A20 = 0, A30 = 0, A40 = 0;
    ulong A01 = 0, A11 = 0, A21 = 0, A31 = 0, A41 = 0;
    ulong A02 = 0, A12 = 0, A22 = 0, A32 = 0, A42 = 0;
    ulong A03 = 0, A13 = 0, A23 = 0, A33 = 0, A43 = 0;
    ulong A04 = 0, A14 = 0, A24 = 0, A34 = 0, A44 = 0;

    if (0 < n_lanes) A00 = seeds[base + 0];
    if (1 < n_lanes) A10 = seeds[base + 1];
    if (2 < n_lanes) A20 = seeds[base + 2];
    if (3 < n_lanes) A30 = seeds[base + 3];
    if (4 < n_lanes) A40 = seeds[base + 4];

    if (5 < n_lanes) A01 = seeds[base + 5];
    if (6 < n_lanes) A11 = seeds[base + 6];
    if (7 < n_lanes) A21 = seeds[base + 7];
    if (8 < n_lanes) A31 = seeds[base + 8];
    if (9 < n_lanes) A41 = seeds[base + 9];

    if (10 < n_lanes) A02 = seeds[base + 10];
    if (11 < n_lanes) A12 = seeds[base + 11];
    if (12 < n_lanes) A22 = seeds[base + 12];
    if (13 < n_lanes) A32 = seeds[base + 13];
    if (14 < n_lanes) A42 = seeds[base + 14];

    if (15 < n_lanes) A03 = seeds[base + 15];

    ulong P00 = (n_lanes == 0) ? 0x06ul : 0ul;
    ulong P10 = (n_lanes == 1) ? 0x06ul : 0ul;
    ulong P20 = (n_lanes == 2) ? 0x06ul : 0ul;
    ulong P30 = (n_lanes == 3) ? 0x06ul : 0ul;
    ulong P40 = (n_lanes == 4) ? 0x06ul : 0ul;

    ulong P01 = (n_lanes == 5) ? 0x06ul : 0ul;
    ulong P11 = (n_lanes == 6) ? 0x06ul : 0ul;
    ulong P21 = (n_lanes == 7) ? 0x06ul : 0ul;
    ulong P31 = (n_lanes == 8) ? 0x06ul : 0ul;
    ulong P41 = (n_lanes == 9) ? 0x06ul : 0ul;

    ulong P02 = (n_lanes == 10) ? 0x06ul : 0ul;
    ulong P12 = (n_lanes == 11) ? 0x06ul : 0ul;
    ulong P22 = (n_lanes == 12) ? 0x06ul : 0ul;
    ulong P32 = (n_lanes == 13) ? 0x06ul : 0ul;
    ulong P42 = (n_lanes == 14) ? 0x06ul : 0ul;

    ulong P03 = (n_lanes == 15) ? 0x06ul : 0ul;

    ulong pad16 = (n_lanes == 16) ? 0x8000000000000006ul : 0x8000000000000000ul;

    for (uint step = 0u; step < w; ++step) {
        
        A00 = (n_lanes > 0) ? A00 : P00;
        A10 = (n_lanes > 1) ? A10 : P10;
        A20 = (n_lanes > 2) ? A20 : P20;
        A30 = (n_lanes > 3) ? A30 : P30;
        A40 = (n_lanes > 4) ? A40 : P40;

        A01 = (n_lanes > 5) ? A01 : P01;
        A11 = (n_lanes > 6) ? A11 : P11;
        A21 = (n_lanes > 7) ? A21 : P21;
        A31 = (n_lanes > 8) ? A31 : P31;
        A41 = (n_lanes > 9) ? A41 : P41;

        A02 = (n_lanes > 10) ? A02 : P02;
        A12 = (n_lanes > 11) ? A12 : P12;
        A22 = (n_lanes > 12) ? A22 : P22;
        A32 = (n_lanes > 13) ? A32 : P32;
        A42 = (n_lanes > 14) ? A42 : P42;

        A03 = (n_lanes > 15) ? A03 : P03;

        A13 = pad16;
        
        A23 = 0; A33 = 0; A43 = 0;
        A04 = 0; A14 = 0; A24 = 0; A34 = 0; A44 = 0;

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

    if (0 < n_lanes) tips[base + 0] = A00;
    if (1 < n_lanes) tips[base + 1] = A10;
    if (2 < n_lanes) tips[base + 2] = A20;
    if (3 < n_lanes) tips[base + 3] = A30;
    if (4 < n_lanes) tips[base + 4] = A40;

    if (5 < n_lanes) tips[base + 5] = A01;
    if (6 < n_lanes) tips[base + 6] = A11;
    if (7 < n_lanes) tips[base + 7] = A21;
    if (8 < n_lanes) tips[base + 8] = A31;
    if (9 < n_lanes) tips[base + 9] = A41;

    if (10 < n_lanes) tips[base + 10] = A02;
    if (11 < n_lanes) tips[base + 11] = A12;
    if (12 < n_lanes) tips[base + 12] = A22;
    if (13 < n_lanes) tips[base + 13] = A32;
    if (14 < n_lanes) tips[base + 14] = A42;

    if (15 < n_lanes) tips[base + 15] = A03;
}
```