To optimize the bit-interleaved Keccak implementation further, we can significantly reduce register pressure by replacing the explicit pre-computation of state masks and padding lanes with Metal's native `select` built-in and a uniform `switch` statement. The incumbent allocates over 48 scalar registers just to hold `M` (mask) and `P` (pad) variables for the lane-truncation logic. By using `select(uint2(0), A, n_lanes > k)`, the compiler leverages the GPU's `csel` (conditional select) instruction, which consumes no mask registers. A perfectly uniform `switch` then assigns the domain pad directly via immediates. Furthermore, vectorizing the E/O pairs into `uint2` structures shrinks the code size drastically and gives the compiler explicit vector-pairing semantics, improving the scheduling of the 32-bit instructions on Apple Silicon's scalar ALUs.

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