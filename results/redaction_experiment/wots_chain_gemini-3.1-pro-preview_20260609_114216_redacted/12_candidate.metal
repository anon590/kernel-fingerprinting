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
    uint lo = morton_expand(V.x) | (morton_expand(V.y) << 1);
    uint hi = morton_expand(V.x >> 16) | (morton_expand(V.y >> 16) << 1);
    return ((ulong)hi << 32) | lo;
}

#define ROTL32(x, k) (((k) == 0) ? (x) : (((x) << (k)) | ((x) >> (32 - (k)))))

#define R2(X, k) (((k) & 1) ? \
    uint2(ROTL32((X).y, (((k) / 2) + 1) & 31), ROTL32((X).x, ((k) / 2) & 31)) : \
    uint2(ROTL32((X).x, ((k) / 2) & 31), ROTL32((X).y, ((k) / 2) & 31)))

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
        uint2 D0 = C4 ^ R2(C1, 1); \
        uint2 D1 = C0 ^ R2(C2, 1); \
        uint2 D2 = C1 ^ R2(C3, 1); \
        uint2 D3 = C2 ^ R2(C4, 1); \
        uint2 D4 = C3 ^ R2(C0, 1); \
        \
        uint2 B00 = A00 ^ D0; \
        uint2 B10 = R2(A11 ^ D1, 44); \
        uint2 B20 = R2(A22 ^ D2, 43); \
        uint2 B30 = R2(A33 ^ D3, 21); \
        uint2 B40 = R2(A44 ^ D4, 14); \
        \
        uint2 B01 = R2(A30 ^ D3, 28); \
        uint2 B11 = R2(A41 ^ D4, 20); \
        uint2 B21 = R2(A02 ^ D0,  3); \
        uint2 B31 = R2(A13 ^ D1, 45); \
        uint2 B41 = R2(A24 ^ D2, 61); \
        \
        uint2 B02 = R2(A10 ^ D1,  1); \
        uint2 B12 = R2(A21 ^ D2,  6); \
        uint2 B22 = R2(A32 ^ D3, 25); \
        uint2 B32 = R2(A43 ^ D4,  8); \
        uint2 B42 = R2(A04 ^ D0, 18); \
        \
        uint2 B03 = R2(A40 ^ D4, 27); \
        uint2 B13 = R2(A01 ^ D0, 36); \
        uint2 B23 = R2(A12 ^ D1, 10); \
        uint2 B33 = R2(A23 ^ D2, 15); \
        uint2 B43 = R2(A34 ^ D3, 56); \
        \
        uint2 B04 = R2(A20 ^ D2, 62); \
        uint2 B14 = R2(A31 ^ D3, 55); \
        uint2 B24 = R2(A42 ^ D4, 39); \
        uint2 B34 = R2(A03 ^ D0, 41); \
        uint2 B44 = R2(A14 ^ D1,  2); \
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

    if (0 < n_lanes)  A00 = to_interleaved(seeds[base + 0]);
    if (1 < n_lanes)  A10 = to_interleaved(seeds[base + 1]);
    if (2 < n_lanes)  A20 = to_interleaved(seeds[base + 2]);
    if (3 < n_lanes)  A30 = to_interleaved(seeds[base + 3]);
    if (4 < n_lanes)  A40 = to_interleaved(seeds[base + 4]);

    if (5 < n_lanes)  A01 = to_interleaved(seeds[base + 5]);
    if (6 < n_lanes)  A11 = to_interleaved(seeds[base + 6]);
    if (7 < n_lanes)  A21 = to_interleaved(seeds[base + 7]);
    if (8 < n_lanes)  A31 = to_interleaved(seeds[base + 8]);
    if (9 < n_lanes)  A41 = to_interleaved(seeds[base + 9]);

    if (10 < n_lanes) A02 = to_interleaved(seeds[base + 10]);
    if (11 < n_lanes) A12 = to_interleaved(seeds[base + 11]);
    if (12 < n_lanes) A22 = to_interleaved(seeds[base + 12]);
    if (13 < n_lanes) A32 = to_interleaved(seeds[base + 13]);
    if (14 < n_lanes) A42 = to_interleaved(seeds[base + 14]);

    if (15 < n_lanes) A03 = to_interleaved(seeds[base + 15]);

    constexpr uint2 PAD06 = to_interleaved(0x06ul);
    constexpr uint2 PAD16_16 = to_interleaved(0x8000000000000006ul);
    constexpr uint2 PAD16_OTHERS = to_interleaved(0x8000000000000000ul);
    uint2 pad16 = select(PAD16_OTHERS, PAD16_16, bool2(n_lanes == 16));

    bool c00 = n_lanes > 0;  uint2 P00 = (n_lanes == 0)  ? PAD06 : uint2(0);
    bool c10 = n_lanes > 1;  uint2 P10 = (n_lanes == 1)  ? PAD06 : uint2(0);
    bool c20 = n_lanes > 2;  uint2 P20 = (n_lanes == 2)  ? PAD06 : uint2(0);
    bool c30 = n_lanes > 3;  uint2 P30 = (n_lanes == 3)  ? PAD06 : uint2(0);
    bool c40 = n_lanes > 4;  uint2 P40 = (n_lanes == 4)  ? PAD06 : uint2(0);

    bool c01 = n_lanes > 5;  uint2 P01 = (n_lanes == 5)  ? PAD06 : uint2(0);
    bool c11 = n_lanes > 6;  uint2 P11 = (n_lanes == 6)  ? PAD06 : uint2(0);
    bool c21 = n_lanes > 7;  uint2 P21 = (n_lanes == 7)  ? PAD06 : uint2(0);
    bool c31 = n_lanes > 8;  uint2 P31 = (n_lanes == 8)  ? PAD06 : uint2(0);
    bool c41 = n_lanes > 9;  uint2 P41 = (n_lanes == 9)  ? PAD06 : uint2(0);

    bool c02 = n_lanes > 10; uint2 P02 = (n_lanes == 10) ? PAD06 : uint2(0);
    bool c12 = n_lanes > 11; uint2 P12 = (n_lanes == 11) ? PAD06 : uint2(0);
    bool c22 = n_lanes > 12; uint2 P22 = (n_lanes == 12) ? PAD06 : uint2(0);
    bool c32 = n_lanes > 13; uint2 P32 = (n_lanes == 13) ? PAD06 : uint2(0);
    bool c42 = n_lanes > 14; uint2 P42 = (n_lanes == 14) ? PAD06 : uint2(0);

    bool c03 = n_lanes > 15; uint2 P03 = (n_lanes == 15) ? PAD06 : uint2(0);

    for (uint step = 0u; step < w; ++step) {
        
        A00 = select(P00, A00, bool2(c00));
        A10 = select(P10, A10, bool2(c10));
        A20 = select(P20, A20, bool2(c20));
        A30 = select(P30, A30, bool2(c30));
        A40 = select(P40, A40, bool2(c40));

        A01 = select(P01, A01, bool2(c01));
        A11 = select(P11, A11, bool2(c11));
        A21 = select(P21, A21, bool2(c21));
        A31 = select(P31, A31, bool2(c31));
        A41 = select(P41, A41, bool2(c41));

        A02 = select(P02, A02, bool2(c02));
        A12 = select(P12, A12, bool2(c12));
        A22 = select(P22, A22, bool2(c22));
        A32 = select(P32, A32, bool2(c32));
        A42 = select(P42, A42, bool2(c42));

        A03 = select(P03, A03, bool2(c03));

        A13 = pad16;
        
        A23 = uint2(0);
        A33 = uint2(0);
        A43 = uint2(0);
        A04 = uint2(0);
        A14 = uint2(0);
        A24 = uint2(0);
        A34 = uint2(0);
        A44 = uint2(0);

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

    if (0 < n_lanes)  tips[base + 0] = from_interleaved(A00);
    if (1 < n_lanes)  tips[base + 1] = from_interleaved(A10);
    if (2 < n_lanes)  tips[base + 2] = from_interleaved(A20);
    if (3 < n_lanes)  tips[base + 3] = from_interleaved(A30);
    if (4 < n_lanes)  tips[base + 4] = from_interleaved(A40);

    if (5 < n_lanes)  tips[base + 5] = from_interleaved(A01);
    if (6 < n_lanes)  tips[base + 6] = from_interleaved(A11);
    if (7 < n_lanes)  tips[base + 7] = from_interleaved(A21);
    if (8 < n_lanes)  tips[base + 8] = from_interleaved(A31);
    if (9 < n_lanes)  tips[base + 9] = from_interleaved(A41);

    if (10 < n_lanes) tips[base + 10] = from_interleaved(A02);
    if (11 < n_lanes) tips[base + 11] = from_interleaved(A12);
    if (12 < n_lanes) tips[base + 12] = from_interleaved(A22);
    if (13 < n_lanes) tips[base + 13] = from_interleaved(A32);
    if (14 < n_lanes) tips[base + 14] = from_interleaved(A42);

    if (15 < n_lanes) tips[base + 15] = from_interleaved(A03);
}