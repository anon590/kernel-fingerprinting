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
        A00_E = B00_E ^ (~B10_E & B20_E); if (RC.x) A00_E ^= RC.x; \
        A00_O = B00_O ^ (~B10_O & B20_O); if (RC.y) A00_O ^= RC.y; \
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
    uint pad16_E = select(PAD16_OTHERS.x, PAD16_16.x, n_lanes == 16);
    uint pad16_O = select(PAD16_OTHERS.y, PAD16_16.y, n_lanes == 16);

    bool c00 = n_lanes > 0;  uint P00_E = (n_lanes == 0)  ? pad06_E : 0u; uint P00_O = (n_lanes == 0)  ? pad06_O : 0u;
    bool c10 = n_lanes > 1;  uint P10_E = (n_lanes == 1)  ? pad06_E : 0u; uint P10_O = (n_lanes == 1)  ? pad06_O : 0u;
    bool c20 = n_lanes > 2;  uint P20_E = (n_lanes == 2)  ? pad06_E : 0u; uint P20_O = (n_lanes == 2)  ? pad06_O : 0u;
    bool c30 = n_lanes > 3;  uint P30_E = (n_lanes == 3)  ? pad06_E : 0u; uint P30_O = (n_lanes == 3)  ? pad06_O : 0u;
    bool c40 = n_lanes > 4;  uint P40_E = (n_lanes == 4)  ? pad06_E : 0u; uint P40_O = (n_lanes == 4)  ? pad06_O : 0u;

    bool c01 = n_lanes > 5;  uint P01_E = (n_lanes == 5)  ? pad06_E : 0u; uint P01_O = (n_lanes == 5)  ? pad06_O : 0u;
    bool c11 = n_lanes > 6;  uint P11_E = (n_lanes == 6)  ? pad06_E : 0u; uint P11_O = (n_lanes == 6)  ? pad06_O : 0u;
    bool c21 = n_lanes > 7;  uint P21_E = (n_lanes == 7)  ? pad06_E : 0u; uint P21_O = (n_lanes == 7)  ? pad06_O : 0u;
    bool c31 = n_lanes > 8;  uint P31_E = (n_lanes == 8)  ? pad06_E : 0u; uint P31_O = (n_lanes == 8)  ? pad06_O : 0u;
    bool c41 = n_lanes > 9;  uint P41_E = (n_lanes == 9)  ? pad06_E : 0u; uint P41_O = (n_lanes == 9)  ? pad06_O : 0u;

    bool c02 = n_lanes > 10; uint P02_E = (n_lanes == 10) ? pad06_E : 0u; uint P02_O = (n_lanes == 10) ? pad06_O : 0u;
    bool c12 = n_lanes > 11; uint P12_E = (n_lanes == 11) ? pad06_E : 0u; uint P12_O = (n_lanes == 11) ? pad06_O : 0u;
    bool c22 = n_lanes > 12; uint P22_E = (n_lanes == 12) ? pad06_E : 0u; uint P22_O = (n_lanes == 12) ? pad06_O : 0u;
    bool c32 = n_lanes > 13; uint P32_E = (n_lanes == 13) ? pad06_E : 0u; uint P32_O = (n_lanes == 13) ? pad06_O : 0u;
    bool c42 = n_lanes > 14; uint P42_E = (n_lanes == 14) ? pad06_E : 0u; uint P42_O = (n_lanes == 14) ? pad06_O : 0u;

    bool c03 = n_lanes > 15; uint P03_E = (n_lanes == 15) ? pad06_E : 0u; uint P03_O = (n_lanes == 15) ? pad06_O : 0u;

    for (uint step = 0u; step < w; ++step) {
        
        A00_E = select(P00_E, A00_E, c00); A00_O = select(P00_O, A00_O, c00);
        A10_E = select(P10_E, A10_E, c10); A10_O = select(P10_O, A10_O, c10);
        A20_E = select(P20_E, A20_E, c20); A20_O = select(P20_O, A20_O, c20);
        A30_E = select(P30_E, A30_E, c30); A30_O = select(P30_O, A30_O, c30);
        A40_E = select(P40_E, A40_E, c40); A40_O = select(P40_O, A40_O, c40);

        A01_E = select(P01_E, A01_E, c01); A01_O = select(P01_O, A01_O, c01);
        A11_E = select(P11_E, A11_E, c11); A11_O = select(P11_O, A11_O, c11);
        A21_E = select(P21_E, A21_E, c21); A21_O = select(P21_O, A21_O, c21);
        A31_E = select(P31_E, A31_E, c31); A31_O = select(P31_O, A31_O, c31);
        A41_E = select(P41_E, A41_E, c41); A41_O = select(P41_O, A41_O, c41);

        A02_E = select(P02_E, A02_E, c02); A02_O = select(P02_O, A02_O, c02);
        A12_E = select(P12_E, A12_E, c12); A12_O = select(P12_O, A12_O, c12);
        A22_E = select(P22_E, A22_E, c22); A22_O = select(P22_O, A22_O, c22);
        A32_E = select(P32_E, A32_E, c32); A32_O = select(P32_O, A32_O, c32);
        A42_E = select(P42_E, A42_E, c42); A42_O = select(P42_O, A42_O, c42);

        A03_E = select(P03_E, A03_E, c03); A03_O = select(P03_O, A03_O, c03);

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