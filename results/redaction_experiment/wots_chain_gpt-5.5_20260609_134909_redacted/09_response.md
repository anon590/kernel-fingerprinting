I’m switching the permutation core to native `ulong` lanes instead of manual `uint2` low/high halves. This is structurally different from the incumbent: it lets the Metal compiler lower 64-bit XOR/AND/shift patterns directly, reducing source-level swizzles, pair construction, and load/store packing overhead. The chain semantics and runtime `n_bytes`/`w` handling remain unchanged, with specialized reset paths for common lane counts.

```metal
#include <metal_stdlib>
using namespace metal;

#define U64(hi, lo) ((((ulong)(hi)) << 32) | ((ulong)(lo)))

#define Z64  ((ulong)0u)
#define D64  ((ulong)0x00000006u)
#define F64  U64(0x80000000u, 0x00000000u)
#define DF64 U64(0x80000000u, 0x00000006u)

#define ROL64(x, k) (((x) << (k)) | ((x) >> (64u - (k))))

#define KECCAK_ROUND64(RC) do { \
    ulong C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    ulong C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    ulong C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    ulong C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    ulong C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    ulong D0v = C4 ^ ROL64(C1, 1u); \
    ulong D1v = C0 ^ ROL64(C2, 1u); \
    ulong D2v = C1 ^ ROL64(C3, 1u); \
    ulong D3v = C2 ^ ROL64(C4, 1u); \
    ulong D4v = C3 ^ ROL64(C0, 1u); \
    A00 ^= D0v; A01 ^= D0v; A02 ^= D0v; A03 ^= D0v; A04 ^= D0v; \
    A10 ^= D1v; A11 ^= D1v; A12 ^= D1v; A13 ^= D1v; A14 ^= D1v; \
    A20 ^= D2v; A21 ^= D2v; A22 ^= D2v; A23 ^= D2v; A24 ^= D2v; \
    A30 ^= D3v; A31 ^= D3v; A32 ^= D3v; A33 ^= D3v; A34 ^= D3v; \
    A40 ^= D4v; A41 ^= D4v; A42 ^= D4v; A43 ^= D4v; A44 ^= D4v; \
    ulong T = A10; \
    A10 = ROL64(A11, 44u); \
    A11 = ROL64(A41, 20u); \
    A41 = ROL64(A24, 61u); \
    A24 = ROL64(A42, 39u); \
    A42 = ROL64(A04, 18u); \
    A04 = ROL64(A20, 62u); \
    A20 = ROL64(A22, 43u); \
    A22 = ROL64(A32, 25u); \
    A32 = ROL64(A43,  8u); \
    A43 = ROL64(A34, 56u); \
    A34 = ROL64(A03, 41u); \
    A03 = ROL64(A40, 27u); \
    A40 = ROL64(A44, 14u); \
    A44 = ROL64(A14,  2u); \
    A14 = ROL64(A31, 55u); \
    A31 = ROL64(A13, 45u); \
    A13 = ROL64(A01, 36u); \
    A01 = ROL64(A30, 28u); \
    A30 = ROL64(A33, 21u); \
    A33 = ROL64(A23, 15u); \
    A23 = ROL64(A12, 10u); \
    A12 = ROL64(A21,  6u); \
    A21 = ROL64(A02,  3u); \
    A02 = ROL64(T,    1u); \
    ulong T0, T1, T2, T3, T4; \
    T0 = A00; T1 = A10; T2 = A20; T3 = A30; T4 = A40; \
    A00 = T0 ^ ((~T1) & T2) ^ (RC); \
    A10 = T1 ^ ((~T2) & T3); \
    A20 = T2 ^ ((~T3) & T4); \
    A30 = T3 ^ ((~T4) & T0); \
    A40 = T4 ^ ((~T0) & T1); \
    T0 = A01; T1 = A11; T2 = A21; T3 = A31; T4 = A41; \
    A01 = T0 ^ ((~T1) & T2); \
    A11 = T1 ^ ((~T2) & T3); \
    A21 = T2 ^ ((~T3) & T4); \
    A31 = T3 ^ ((~T4) & T0); \
    A41 = T4 ^ ((~T0) & T1); \
    T0 = A02; T1 = A12; T2 = A22; T3 = A32; T4 = A42; \
    A02 = T0 ^ ((~T1) & T2); \
    A12 = T1 ^ ((~T2) & T3); \
    A22 = T2 ^ ((~T3) & T4); \
    A32 = T3 ^ ((~T4) & T0); \
    A42 = T4 ^ ((~T0) & T1); \
    T0 = A03; T1 = A13; T2 = A23; T3 = A33; T4 = A43; \
    A03 = T0 ^ ((~T1) & T2); \
    A13 = T1 ^ ((~T2) & T3); \
    A23 = T2 ^ ((~T3) & T4); \
    A33 = T3 ^ ((~T4) & T0); \
    A43 = T4 ^ ((~T0) & T1); \
    T0 = A04; T1 = A14; T2 = A24; T3 = A34; T4 = A44; \
    A04 = T0 ^ ((~T1) & T2); \
    A14 = T1 ^ ((~T2) & T3); \
    A24 = T2 ^ ((~T3) & T4); \
    A34 = T3 ^ ((~T4) & T0); \
    A44 = T4 ^ ((~T0) & T1); \
} while (0)

#define KECCAK_F1600_64() do { \
    KECCAK_ROUND64(U64(0x00000000u, 0x00000001u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x00008082u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x0000808Au)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008000u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000808Bu)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x80000001u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008081u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008009u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000008Au)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x00000088u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x80008009u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x8000000Au)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x8000808Bu)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x0000008Bu)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008089u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008003u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008002u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x00000080u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x0000800Au)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x8000000Au)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008081u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x00008080u)); \
    KECCAK_ROUND64(U64(0x00000000u, 0x80000001u)); \
    KECCAK_ROUND64(U64(0x80000000u, 0x80008008u)); \
} while (0)

#define RESET_N1_64() do { \
    A10 = D64; A20 = Z64; A30 = Z64; A40 = Z64; \
    A01 = Z64; A11 = Z64; A21 = Z64; A31 = Z64; A41 = Z64; \
    A02 = Z64; A12 = Z64; A22 = Z64; A32 = Z64; A42 = Z64; \
    A03 = Z64; A13 = F64; A23 = Z64; A33 = Z64; A43 = Z64; \
    A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64; \
} while (0)

#define RESET_N2_64() do { \
    A20 = D64; A30 = Z64; A40 = Z64; \
    A01 = Z64; A11 = Z64; A21 = Z64; A31 = Z64; A41 = Z64; \
    A02 = Z64; A12 = Z64; A22 = Z64; A32 = Z64; A42 = Z64; \
    A03 = Z64; A13 = F64; A23 = Z64; A33 = Z64; A43 = Z64; \
    A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64; \
} while (0)

#define RESET_N3_64() do { \
    A30 = D64; A40 = Z64; \
    A01 = Z64; A11 = Z64; A21 = Z64; A31 = Z64; A41 = Z64; \
    A02 = Z64; A12 = Z64; A22 = Z64; A32 = Z64; A42 = Z64; \
    A03 = Z64; A13 = F64; A23 = Z64; A33 = Z64; A43 = Z64; \
    A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64; \
} while (0)

#define RESET_N4_64() do { \
    A40 = D64; \
    A01 = Z64; A11 = Z64; A21 = Z64; A31 = Z64; A41 = Z64; \
    A02 = Z64; A12 = Z64; A22 = Z64; A32 = Z64; A42 = Z64; \
    A03 = Z64; A13 = F64; A23 = Z64; A33 = Z64; A43 = Z64; \
    A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64; \
} while (0)

#define RESET_N8_64() do { \
    A31 = D64; A41 = Z64; \
    A02 = Z64; A12 = Z64; A22 = Z64; A32 = Z64; A42 = Z64; \
    A03 = Z64; A13 = F64; A23 = Z64; A33 = Z64; A43 = Z64; \
    A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64; \
} while (0)

#define RESET_N16_64() do { \
    A13 = DF64; A23 = Z64; A33 = Z64; A43 = Z64; \
    A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64; \
} while (0)

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    const uint n_lanes = n_bytes >> 3;
    const uint ww = w;
    if (n_lanes == 0u) return;

    if (n_lanes == 4u) {
        const uint base = idx << 2;

        ulong A00 = seeds[base + 0u];
        ulong A10 = seeds[base + 1u];
        ulong A20 = seeds[base + 2u];
        ulong A30 = seeds[base + 3u];
        ulong A40, A01, A11, A21, A31, A41;
        ulong A02, A12, A22, A32, A42;
        ulong A03, A13, A23, A33, A43;
        ulong A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N4_64();
            KECCAK_F1600_64();
        }

        tips[base + 0u] = A00;
        tips[base + 1u] = A10;
        tips[base + 2u] = A20;
        tips[base + 3u] = A30;
        return;
    }

    if (n_lanes == 2u) {
        const uint base = idx << 1;

        ulong A00 = seeds[base + 0u];
        ulong A10 = seeds[base + 1u];
        ulong A20, A30, A40, A01, A11, A21, A31, A41;
        ulong A02, A12, A22, A32, A42;
        ulong A03, A13, A23, A33, A43;
        ulong A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N2_64();
            KECCAK_F1600_64();
        }

        tips[base + 0u] = A00;
        tips[base + 1u] = A10;
        return;
    }

    if (n_lanes == 3u) {
        const uint base = idx * 3u;

        ulong A00 = seeds[base + 0u];
        ulong A10 = seeds[base + 1u];
        ulong A20 = seeds[base + 2u];
        ulong A30, A40, A01, A11, A21, A31, A41;
        ulong A02, A12, A22, A32, A42;
        ulong A03, A13, A23, A33, A43;
        ulong A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N3_64();
            KECCAK_F1600_64();
        }

        tips[base + 0u] = A00;
        tips[base + 1u] = A10;
        tips[base + 2u] = A20;
        return;
    }

    if (n_lanes == 8u) {
        const uint base = idx << 3;

        ulong A00 = seeds[base + 0u];
        ulong A10 = seeds[base + 1u];
        ulong A20 = seeds[base + 2u];
        ulong A30 = seeds[base + 3u];
        ulong A40 = seeds[base + 4u];
        ulong A01 = seeds[base + 5u];
        ulong A11 = seeds[base + 6u];
        ulong A21 = seeds[base + 7u];
        ulong A31, A41;
        ulong A02, A12, A22, A32, A42;
        ulong A03, A13, A23, A33, A43;
        ulong A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N8_64();
            KECCAK_F1600_64();
        }

        tips[base + 0u] = A00;
        tips[base + 1u] = A10;
        tips[base + 2u] = A20;
        tips[base + 3u] = A30;
        tips[base + 4u] = A40;
        tips[base + 5u] = A01;
        tips[base + 6u] = A11;
        tips[base + 7u] = A21;
        return;
    }

    if (n_lanes == 16u) {
        const uint base = idx << 4;

        ulong A00 = seeds[base +  0u];
        ulong A10 = seeds[base +  1u];
        ulong A20 = seeds[base +  2u];
        ulong A30 = seeds[base +  3u];
        ulong A40 = seeds[base +  4u];
        ulong A01 = seeds[base +  5u];
        ulong A11 = seeds[base +  6u];
        ulong A21 = seeds[base +  7u];
        ulong A31 = seeds[base +  8u];
        ulong A41 = seeds[base +  9u];
        ulong A02 = seeds[base + 10u];
        ulong A12 = seeds[base + 11u];
        ulong A22 = seeds[base + 12u];
        ulong A32 = seeds[base + 13u];
        ulong A42 = seeds[base + 14u];
        ulong A03 = seeds[base + 15u];
        ulong A13, A23, A33, A43;
        ulong A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N16_64();
            KECCAK_F1600_64();
        }

        tips[base +  0u] = A00;
        tips[base +  1u] = A10;
        tips[base +  2u] = A20;
        tips[base +  3u] = A30;
        tips[base +  4u] = A40;
        tips[base +  5u] = A01;
        tips[base +  6u] = A11;
        tips[base +  7u] = A21;
        tips[base +  8u] = A31;
        tips[base +  9u] = A41;
        tips[base + 10u] = A02;
        tips[base + 11u] = A12;
        tips[base + 12u] = A22;
        tips[base + 13u] = A32;
        tips[base + 14u] = A42;
        tips[base + 15u] = A03;
        return;
    }

    if (n_lanes == 1u) {
        const uint base = idx;

        ulong A00 = seeds[base];
        ulong A10, A20, A30, A40, A01, A11, A21, A31, A41;
        ulong A02, A12, A22, A32, A42;
        ulong A03, A13, A23, A33, A43;
        ulong A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N1_64();
            KECCAK_F1600_64();
        }

        tips[base] = A00;
        return;
    }

    const uint base = idx * n_lanes;

    ulong A00 = Z64, A10 = Z64, A20 = Z64, A30 = Z64, A40 = Z64;
    ulong A01 = Z64, A11 = Z64, A21 = Z64, A31 = Z64, A41 = Z64;
    ulong A02 = Z64, A12 = Z64, A22 = Z64, A32 = Z64, A42 = Z64;
    ulong A03 = Z64, A13 = Z64, A23 = Z64, A33 = Z64, A43 = Z64;
    ulong A04 = Z64, A14 = Z64, A24 = Z64, A34 = Z64, A44 = Z64;

    if (n_lanes >  0u) A00 = seeds[base +  0u];
    if (n_lanes >  1u) A10 = seeds[base +  1u];
    if (n_lanes >  2u) A20 = seeds[base +  2u];
    if (n_lanes >  3u) A30 = seeds[base +  3u];
    if (n_lanes >  4u) A40 = seeds[base +  4u];
    if (n_lanes >  5u) A01 = seeds[base +  5u];
    if (n_lanes >  6u) A11 = seeds[base +  6u];
    if (n_lanes >  7u) A21 = seeds[base +  7u];
    if (n_lanes >  8u) A31 = seeds[base +  8u];
    if (n_lanes >  9u) A41 = seeds[base +  9u];
    if (n_lanes > 10u) A02 = seeds[base + 10u];
    if (n_lanes > 11u) A12 = seeds[base + 11u];
    if (n_lanes > 12u) A22 = seeds[base + 12u];
    if (n_lanes > 13u) A32 = seeds[base + 13u];
    if (n_lanes > 14u) A42 = seeds[base + 14u];
    if (n_lanes > 15u) A03 = seeds[base + 15u];

    for (uint step = 0u; step < ww; ++step) {
        if (n_lanes <=  1u) A10 = Z64;
        if (n_lanes <=  2u) A20 = Z64;
        if (n_lanes <=  3u) A30 = Z64;
        if (n_lanes <=  4u) A40 = Z64;
        if (n_lanes <=  5u) A01 = Z64;
        if (n_lanes <=  6u) A11 = Z64;
        if (n_lanes <=  7u) A21 = Z64;
        if (n_lanes <=  8u) A31 = Z64;
        if (n_lanes <=  9u) A41 = Z64;
        if (n_lanes <= 10u) A02 = Z64;
        if (n_lanes <= 11u) A12 = Z64;
        if (n_lanes <= 12u) A22 = Z64;
        if (n_lanes <= 13u) A32 = Z64;
        if (n_lanes <= 14u) A42 = Z64;
        if (n_lanes <= 15u) A03 = Z64;

        A13 = Z64; A23 = Z64; A33 = Z64; A43 = Z64;
        A04 = Z64; A14 = Z64; A24 = Z64; A34 = Z64; A44 = Z64;

        switch (n_lanes) {
            case 1u:  A10 = D64; break;
            case 2u:  A20 = D64; break;
            case 3u:  A30 = D64; break;
            case 4u:  A40 = D64; break;
            case 5u:  A01 = D64; break;
            case 6u:  A11 = D64; break;
            case 7u:  A21 = D64; break;
            case 8u:  A31 = D64; break;
            case 9u:  A41 = D64; break;
            case 10u: A02 = D64; break;
            case 11u: A12 = D64; break;
            case 12u: A22 = D64; break;
            case 13u: A32 = D64; break;
            case 14u: A42 = D64; break;
            case 15u: A03 = D64; break;
            case 16u: A13 = D64; break;
            default:  break;
        }

        A13 ^= F64;

        KECCAK_F1600_64();
    }

    if (n_lanes >  0u) tips[base +  0u] = A00;
    if (n_lanes >  1u) tips[base +  1u] = A10;
    if (n_lanes >  2u) tips[base +  2u] = A20;
    if (n_lanes >  3u) tips[base +  3u] = A30;
    if (n_lanes >  4u) tips[base +  4u] = A40;
    if (n_lanes >  5u) tips[base +  5u] = A01;
    if (n_lanes >  6u) tips[base +  6u] = A11;
    if (n_lanes >  7u) tips[base +  7u] = A21;
    if (n_lanes >  8u) tips[base +  8u] = A31;
    if (n_lanes >  9u) tips[base +  9u] = A41;
    if (n_lanes > 10u) tips[base + 10u] = A02;
    if (n_lanes > 11u) tips[base + 11u] = A12;
    if (n_lanes > 12u) tips[base + 12u] = A22;
    if (n_lanes > 13u) tips[base + 13u] = A32;
    if (n_lanes > 14u) tips[base + 14u] = A42;
    if (n_lanes > 15u) tips[base + 15u] = A03;
}
```