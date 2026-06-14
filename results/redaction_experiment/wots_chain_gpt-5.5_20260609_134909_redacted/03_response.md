I’m replacing rotation macros with small inline constant-rotation helpers so rotated expressions are evaluated once before shifting, reducing redundant XOR/temporary work if the compiler fails to fully CSE macro expansions. I also copy runtime constants to locals and use countdown loops, which slightly reduces loop-control overhead while preserving the fast specialized paths for the common WOTS sizes.

```metal
#include <metal_stdlib>
using namespace metal;

#define Z2 uint2(0u, 0u)
#define D2 uint2(0x00000006u, 0u)
#define F2 uint2(0u, 0x80000000u)

inline uint2 load64le(ulong x) {
    return as_type<uint2>(x);
}

inline ulong store64le(uint2 v) {
    return as_type<ulong>(v);
}

#define DEF_ROL_LT(NAME,K) inline uint2 NAME(uint2 v) { \
    return uint2((v.x << (K)) | (v.y >> (32u - (K))), \
                 (v.y << (K)) | (v.x >> (32u - (K)))); \
}

#define DEF_ROL_GT(NAME,K) inline uint2 NAME(uint2 v) { \
    return uint2((v.y << ((K) - 32u)) | (v.x >> (64u - (K))), \
                 (v.x << ((K) - 32u)) | (v.y >> (64u - (K)))); \
}

DEF_ROL_LT(rotl01,  1u)
DEF_ROL_LT(rotl02,  2u)
DEF_ROL_LT(rotl03,  3u)
DEF_ROL_LT(rotl06,  6u)
DEF_ROL_LT(rotl08,  8u)
DEF_ROL_LT(rotl10, 10u)
DEF_ROL_LT(rotl14, 14u)
DEF_ROL_LT(rotl15, 15u)
DEF_ROL_LT(rotl18, 18u)
DEF_ROL_LT(rotl20, 20u)
DEF_ROL_LT(rotl21, 21u)
DEF_ROL_LT(rotl25, 25u)
DEF_ROL_LT(rotl27, 27u)
DEF_ROL_LT(rotl28, 28u)

DEF_ROL_GT(rotl36, 36u)
DEF_ROL_GT(rotl39, 39u)
DEF_ROL_GT(rotl41, 41u)
DEF_ROL_GT(rotl43, 43u)
DEF_ROL_GT(rotl44, 44u)
DEF_ROL_GT(rotl45, 45u)
DEF_ROL_GT(rotl55, 55u)
DEF_ROL_GT(rotl56, 56u)
DEF_ROL_GT(rotl61, 61u)
DEF_ROL_GT(rotl62, 62u)

#define KECCAK_ROUND(RCLO,RCHI) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ rotl01(C1); \
    uint2 D1v = C0 ^ rotl01(C2); \
    uint2 D2v = C1 ^ rotl01(C3); \
    uint2 D3v = C2 ^ rotl01(C4); \
    uint2 D4v = C3 ^ rotl01(C0); \
    uint2 B00 = A00 ^ D0v; \
    uint2 B10 = rotl44(A11 ^ D1v); \
    uint2 B20 = rotl43(A22 ^ D2v); \
    uint2 B30 = rotl21(A33 ^ D3v); \
    uint2 B40 = rotl14(A44 ^ D4v); \
    uint2 B01 = rotl28(A30 ^ D3v); \
    uint2 B11 = rotl20(A41 ^ D4v); \
    uint2 B21 = rotl03(A02 ^ D0v); \
    uint2 B31 = rotl45(A13 ^ D1v); \
    uint2 B41 = rotl61(A24 ^ D2v); \
    uint2 B02 = rotl01(A10 ^ D1v); \
    uint2 B12 = rotl06(A21 ^ D2v); \
    uint2 B22 = rotl25(A32 ^ D3v); \
    uint2 B32 = rotl08(A43 ^ D4v); \
    uint2 B42 = rotl18(A04 ^ D0v); \
    uint2 B03 = rotl27(A40 ^ D4v); \
    uint2 B13 = rotl36(A01 ^ D0v); \
    uint2 B23 = rotl10(A12 ^ D1v); \
    uint2 B33 = rotl15(A23 ^ D2v); \
    uint2 B43 = rotl56(A34 ^ D3v); \
    uint2 B04 = rotl62(A20 ^ D2v); \
    uint2 B14 = rotl55(A31 ^ D3v); \
    uint2 B24 = rotl39(A42 ^ D4v); \
    uint2 B34 = rotl41(A03 ^ D0v); \
    uint2 B44 = rotl02(A14 ^ D1v); \
    A00 = B00 ^ ((~B10) & B20) ^ uint2((RCLO), (RCHI)); \
    A10 = B10 ^ ((~B20) & B30); \
    A20 = B20 ^ ((~B30) & B40); \
    A30 = B30 ^ ((~B40) & B00); \
    A40 = B40 ^ ((~B00) & B10); \
    A01 = B01 ^ ((~B11) & B21); \
    A11 = B11 ^ ((~B21) & B31); \
    A21 = B21 ^ ((~B31) & B41); \
    A31 = B31 ^ ((~B41) & B01); \
    A41 = B41 ^ ((~B01) & B11); \
    A02 = B02 ^ ((~B12) & B22); \
    A12 = B12 ^ ((~B22) & B32); \
    A22 = B22 ^ ((~B32) & B42); \
    A32 = B32 ^ ((~B42) & B02); \
    A42 = B42 ^ ((~B02) & B12); \
    A03 = B03 ^ ((~B13) & B23); \
    A13 = B13 ^ ((~B23) & B33); \
    A23 = B23 ^ ((~B33) & B43); \
    A33 = B33 ^ ((~B43) & B03); \
    A43 = B43 ^ ((~B03) & B13); \
    A04 = B04 ^ ((~B14) & B24); \
    A14 = B14 ^ ((~B24) & B34); \
    A24 = B24 ^ ((~B34) & B44); \
    A34 = B34 ^ ((~B44) & B04); \
    A44 = B44 ^ ((~B04) & B14); \
} while (0)

#define KECCAK_F1600() do { \
    KECCAK_ROUND(0x00000001u, 0x00000000u); \
    KECCAK_ROUND(0x00008082u, 0x00000000u); \
    KECCAK_ROUND(0x0000808Au, 0x80000000u); \
    KECCAK_ROUND(0x80008000u, 0x80000000u); \
    KECCAK_ROUND(0x0000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008009u, 0x80000000u); \
    KECCAK_ROUND(0x0000008Au, 0x00000000u); \
    KECCAK_ROUND(0x00000088u, 0x00000000u); \
    KECCAK_ROUND(0x80008009u, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x00000000u); \
    KECCAK_ROUND(0x8000808Bu, 0x00000000u); \
    KECCAK_ROUND(0x0000008Bu, 0x80000000u); \
    KECCAK_ROUND(0x00008089u, 0x80000000u); \
    KECCAK_ROUND(0x00008003u, 0x80000000u); \
    KECCAK_ROUND(0x00008002u, 0x80000000u); \
    KECCAK_ROUND(0x00000080u, 0x80000000u); \
    KECCAK_ROUND(0x0000800Au, 0x00000000u); \
    KECCAK_ROUND(0x8000000Au, 0x80000000u); \
    KECCAK_ROUND(0x80008081u, 0x80000000u); \
    KECCAK_ROUND(0x00008080u, 0x80000000u); \
    KECCAK_ROUND(0x80000001u, 0x00000000u); \
    KECCAK_ROUND(0x80008008u, 0x80000000u); \
} while (0)

#define RESET_N2() do { \
    A20 = D2; A30 = Z2; A40 = Z2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N3() do { \
    A30 = D2; A40 = Z2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N4() do { \
    A40 = D2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    const uint nc = n_chains;
    if (idx >= nc) return;

    const uint n_lanes = n_bytes >> 3;
    const uint ww = w;
    if (n_lanes == 0u) return;

    if (n_lanes == 4u) {
        uint base = idx << 2;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20 = load64le(seeds[base + 2u]);
        uint2 A30 = load64le(seeds[base + 3u]);
        uint2 A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = ww; step != 0u; --step) {
            RESET_N4();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        tips[base + 2u] = store64le(A20);
        tips[base + 3u] = store64le(A30);
        return;
    }

    if (n_lanes == 2u) {
        uint base = idx << 1;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20, A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = ww; step != 0u; --step) {
            RESET_N2();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        return;
    }

    if (n_lanes == 3u) {
        uint base = idx * 3u;

        uint2 A00 = load64le(seeds[base + 0u]);
        uint2 A10 = load64le(seeds[base + 1u]);
        uint2 A20 = load64le(seeds[base + 2u]);
        uint2 A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = ww; step != 0u; --step) {
            RESET_N3();
            KECCAK_F1600();
        }

        tips[base + 0u] = store64le(A00);
        tips[base + 1u] = store64le(A10);
        tips[base + 2u] = store64le(A20);
        return;
    }

    uint base = idx * n_lanes;

    uint2 A00 = Z2, A10 = Z2, A20 = Z2, A30 = Z2, A40 = Z2;
    uint2 A01 = Z2, A11 = Z2, A21 = Z2, A31 = Z2, A41 = Z2;
    uint2 A02 = Z2, A12 = Z2, A22 = Z2, A32 = Z2, A42 = Z2;
    uint2 A03 = Z2, A13 = Z2, A23 = Z2, A33 = Z2, A43 = Z2;
    uint2 A04 = Z2, A14 = Z2, A24 = Z2, A34 = Z2, A44 = Z2;

    if (n_lanes >  0u) A00 = load64le(seeds[base +  0u]);
    if (n_lanes >  1u) A10 = load64le(seeds[base +  1u]);
    if (n_lanes >  2u) A20 = load64le(seeds[base +  2u]);
    if (n_lanes >  3u) A30 = load64le(seeds[base +  3u]);
    if (n_lanes >  4u) A40 = load64le(seeds[base +  4u]);
    if (n_lanes >  5u) A01 = load64le(seeds[base +  5u]);
    if (n_lanes >  6u) A11 = load64le(seeds[base +  6u]);
    if (n_lanes >  7u) A21 = load64le(seeds[base +  7u]);
    if (n_lanes >  8u) A31 = load64le(seeds[base +  8u]);
    if (n_lanes >  9u) A41 = load64le(seeds[base +  9u]);
    if (n_lanes > 10u) A02 = load64le(seeds[base + 10u]);
    if (n_lanes > 11u) A12 = load64le(seeds[base + 11u]);
    if (n_lanes > 12u) A22 = load64le(seeds[base + 12u]);
    if (n_lanes > 13u) A32 = load64le(seeds[base + 13u]);
    if (n_lanes > 14u) A42 = load64le(seeds[base + 14u]);
    if (n_lanes > 15u) A03 = load64le(seeds[base + 15u]);

    for (uint step = ww; step != 0u; --step) {
        if (n_lanes <=  1u) A10 = Z2;
        if (n_lanes <=  2u) A20 = Z2;
        if (n_lanes <=  3u) A30 = Z2;
        if (n_lanes <=  4u) A40 = Z2;
        if (n_lanes <=  5u) A01 = Z2;
        if (n_lanes <=  6u) A11 = Z2;
        if (n_lanes <=  7u) A21 = Z2;
        if (n_lanes <=  8u) A31 = Z2;
        if (n_lanes <=  9u) A41 = Z2;
        if (n_lanes <= 10u) A02 = Z2;
        if (n_lanes <= 11u) A12 = Z2;
        if (n_lanes <= 12u) A22 = Z2;
        if (n_lanes <= 13u) A32 = Z2;
        if (n_lanes <= 14u) A42 = Z2;
        if (n_lanes <= 15u) A03 = Z2;

        A13 = Z2; A23 = Z2; A33 = Z2; A43 = Z2;
        A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2;

        switch (n_lanes) {
            case 1u:  A10 = D2; break;
            case 5u:  A01 = D2; break;
            case 6u:  A11 = D2; break;
            case 7u:  A21 = D2; break;
            case 8u:  A31 = D2; break;
            case 9u:  A41 = D2; break;
            case 10u: A02 = D2; break;
            case 11u: A12 = D2; break;
            case 12u: A22 = D2; break;
            case 13u: A32 = D2; break;
            case 14u: A42 = D2; break;
            case 15u: A03 = D2; break;
            case 16u: A13 = D2; break;
            default:  A00 = D2; break;
        }

        A13 ^= F2;

        KECCAK_F1600();
    }

    if (n_lanes >  0u) tips[base +  0u] = store64le(A00);
    if (n_lanes >  1u) tips[base +  1u] = store64le(A10);
    if (n_lanes >  2u) tips[base +  2u] = store64le(A20);
    if (n_lanes >  3u) tips[base +  3u] = store64le(A30);
    if (n_lanes >  4u) tips[base +  4u] = store64le(A40);
    if (n_lanes >  5u) tips[base +  5u] = store64le(A01);
    if (n_lanes >  6u) tips[base +  6u] = store64le(A11);
    if (n_lanes >  7u) tips[base +  7u] = store64le(A21);
    if (n_lanes >  8u) tips[base +  8u] = store64le(A31);
    if (n_lanes >  9u) tips[base +  9u] = store64le(A41);
    if (n_lanes > 10u) tips[base + 10u] = store64le(A02);
    if (n_lanes > 11u) tips[base + 11u] = store64le(A12);
    if (n_lanes > 12u) tips[base + 12u] = store64le(A22);
    if (n_lanes > 13u) tips[base + 13u] = store64le(A32);
    if (n_lanes > 14u) tips[base + 14u] = store64le(A42);
    if (n_lanes > 15u) tips[base + 15u] = store64le(A03);
}
```