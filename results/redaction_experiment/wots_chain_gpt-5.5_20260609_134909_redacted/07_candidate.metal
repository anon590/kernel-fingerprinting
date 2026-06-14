#include <metal_stdlib>
using namespace metal;

#define Z2  uint2(0u, 0u)
#define D2  uint2(0x00000006u, 0u)
#define F2  uint2(0u, 0x80000000u)
#define DF2 uint2(0x00000006u, 0x80000000u)

#define ROL_LT(v,k) uint2((((v).x << (k)) | ((v).y >> (32u - (k)))), \
                          (((v).y << (k)) | ((v).x >> (32u - (k)))))

#define ROL_GT(v,k) uint2((((v).y << ((k) - 32u)) | ((v).x >> (64u - (k)))), \
                          (((v).x << ((k) - 32u)) | ((v).y >> (64u - (k)))))

#define KECCAK_ROUND(RCLO,RCHI) do { \
    uint2 C0 = A00 ^ A01 ^ A02 ^ A03 ^ A04; \
    uint2 C1 = A10 ^ A11 ^ A12 ^ A13 ^ A14; \
    uint2 C2 = A20 ^ A21 ^ A22 ^ A23 ^ A24; \
    uint2 C3 = A30 ^ A31 ^ A32 ^ A33 ^ A34; \
    uint2 C4 = A40 ^ A41 ^ A42 ^ A43 ^ A44; \
    uint2 D0v = C4 ^ ROL_LT(C1, 1u); \
    uint2 D1v = C0 ^ ROL_LT(C2, 1u); \
    uint2 D2v = C1 ^ ROL_LT(C3, 1u); \
    uint2 D3v = C2 ^ ROL_LT(C4, 1u); \
    uint2 D4v = C3 ^ ROL_LT(C0, 1u); \
    A00 ^= D0v; A01 ^= D0v; A02 ^= D0v; A03 ^= D0v; A04 ^= D0v; \
    A10 ^= D1v; A11 ^= D1v; A12 ^= D1v; A13 ^= D1v; A14 ^= D1v; \
    A20 ^= D2v; A21 ^= D2v; A22 ^= D2v; A23 ^= D2v; A24 ^= D2v; \
    A30 ^= D3v; A31 ^= D3v; A32 ^= D3v; A33 ^= D3v; A34 ^= D3v; \
    A40 ^= D4v; A41 ^= D4v; A42 ^= D4v; A43 ^= D4v; A44 ^= D4v; \
    uint2 T = A10; \
    A10 = ROL_GT(A11, 44u); \
    A11 = ROL_LT(A41, 20u); \
    A41 = ROL_GT(A24, 61u); \
    A24 = ROL_GT(A42, 39u); \
    A42 = ROL_LT(A04, 18u); \
    A04 = ROL_GT(A20, 62u); \
    A20 = ROL_GT(A22, 43u); \
    A22 = ROL_LT(A32, 25u); \
    A32 = ROL_LT(A43,  8u); \
    A43 = ROL_GT(A34, 56u); \
    A34 = ROL_GT(A03, 41u); \
    A03 = ROL_LT(A40, 27u); \
    A40 = ROL_LT(A44, 14u); \
    A44 = ROL_LT(A14,  2u); \
    A14 = ROL_GT(A31, 55u); \
    A31 = ROL_GT(A13, 45u); \
    A13 = ROL_GT(A01, 36u); \
    A01 = ROL_LT(A30, 28u); \
    A30 = ROL_LT(A33, 21u); \
    A33 = ROL_LT(A23, 15u); \
    A23 = ROL_LT(A12, 10u); \
    A12 = ROL_LT(A21,  6u); \
    A21 = ROL_LT(A02,  3u); \
    A02 = ROL_LT(T,    1u); \
    uint2 T0, T1, T2, T3, T4; \
    T0 = A00; T1 = A10; T2 = A20; T3 = A30; T4 = A40; \
    A00 = T0 ^ ((~T1) & T2) ^ uint2((RCLO), (RCHI)); \
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

#define KECCAK_FIRST_N4() do { \
    uint2 M1 = A10; \
    uint2 M2 = A20; \
    uint2 M3 = A30; \
    uint2 C1v = M1 ^ F2; \
    uint2 E0 = D2 ^ ROL_LT(C1v, 1u); \
    uint2 E1 = A00 ^ ROL_LT(M2, 1u); \
    uint2 E2v = C1v ^ ROL_LT(M3, 1u); \
    uint2 E3 = M2 ^ ROL_LT(D2, 1u); \
    uint2 E4 = M3 ^ ROL_LT(A00, 1u); \
    uint2 B0, B1, B2, B3, B4; \
    B0 = A00 ^ E0; B1 = ROL_GT(E1, 44u); B2 = ROL_GT(E2v, 43u); B3 = ROL_LT(E3, 21u); B4 = ROL_LT(E4, 14u); \
    A00 = B0 ^ ((~B1) & B2) ^ uint2(0x00000001u, 0x00000000u); \
    A10 = B1 ^ ((~B2) & B3); \
    A20 = B2 ^ ((~B3) & B4); \
    A30 = B3 ^ ((~B4) & B0); \
    A40 = B4 ^ ((~B0) & B1); \
    B0 = ROL_LT(M3 ^ E3, 28u); B1 = ROL_LT(E4, 20u); B2 = ROL_LT(E0, 3u); B3 = ROL_GT(F2 ^ E1, 45u); B4 = ROL_GT(E2v, 61u); \
    A01 = B0 ^ ((~B1) & B2); \
    A11 = B1 ^ ((~B2) & B3); \
    A21 = B2 ^ ((~B3) & B4); \
    A31 = B3 ^ ((~B4) & B0); \
    A41 = B4 ^ ((~B0) & B1); \
    B0 = ROL_LT(M1 ^ E1, 1u); B1 = ROL_LT(E2v, 6u); B2 = ROL_LT(E3, 25u); B3 = ROL_LT(E4, 8u); B4 = ROL_LT(E0, 18u); \
    A02 = B0 ^ ((~B1) & B2); \
    A12 = B1 ^ ((~B2) & B3); \
    A22 = B2 ^ ((~B3) & B4); \
    A32 = B3 ^ ((~B4) & B0); \
    A42 = B4 ^ ((~B0) & B1); \
    B0 = ROL_LT(D2 ^ E4, 27u); B1 = ROL_GT(E0, 36u); B2 = ROL_LT(E1, 10u); B3 = ROL_LT(E2v, 15u); B4 = ROL_GT(E3, 56u); \
    A03 = B0 ^ ((~B1) & B2); \
    A13 = B1 ^ ((~B2) & B3); \
    A23 = B2 ^ ((~B3) & B4); \
    A33 = B3 ^ ((~B4) & B0); \
    A43 = B4 ^ ((~B0) & B1); \
    B0 = ROL_GT(M2 ^ E2v, 62u); B1 = ROL_GT(E3, 55u); B2 = ROL_GT(E4, 39u); B3 = ROL_GT(E0, 41u); B4 = ROL_LT(E1, 2u); \
    A04 = B0 ^ ((~B1) & B2); \
    A14 = B1 ^ ((~B2) & B3); \
    A24 = B2 ^ ((~B3) & B4); \
    A34 = B3 ^ ((~B4) & B0); \
    A44 = B4 ^ ((~B0) & B1); \
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

#define KECCAK_F1600_N4() do { \
    KECCAK_FIRST_N4(); \
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

#define RESET_N1() do { \
    A10 = D2; A20 = Z2; A30 = Z2; A40 = Z2; \
    A01 = Z2; A11 = Z2; A21 = Z2; A31 = Z2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
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

#define RESET_N8() do { \
    A31 = D2; A41 = Z2; \
    A02 = Z2; A12 = Z2; A22 = Z2; A32 = Z2; A42 = Z2; \
    A03 = Z2; A13 = F2; A23 = Z2; A33 = Z2; A43 = Z2; \
    A04 = Z2; A14 = Z2; A24 = Z2; A34 = Z2; A44 = Z2; \
} while (0)

#define RESET_N16() do { \
    A13 = DF2; A23 = Z2; A33 = Z2; A43 = Z2; \
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
    if (idx >= n_chains) return;

    const uint n_lanes = n_bytes >> 3;
    const uint ww = w;
    if (n_lanes == 0u) return;

    device const uint2 *seeds2 = reinterpret_cast<device const uint2 *>(seeds);
    device       uint2 *tips2  = reinterpret_cast<device       uint2 *>(tips);

    if (n_lanes == 4u) {
        uint base = idx << 2;

        uint2 A00 = seeds2[base + 0u];
        uint2 A10 = seeds2[base + 1u];
        uint2 A20 = seeds2[base + 2u];
        uint2 A30 = seeds2[base + 3u];
        uint2 A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            KECCAK_F1600_N4();
        }

        tips2[base + 0u] = A00;
        tips2[base + 1u] = A10;
        tips2[base + 2u] = A20;
        tips2[base + 3u] = A30;
        return;
    }

    if (n_lanes == 2u) {
        uint base = idx << 1;

        uint2 A00 = seeds2[base + 0u];
        uint2 A10 = seeds2[base + 1u];
        uint2 A20, A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N2();
            KECCAK_F1600();
        }

        tips2[base + 0u] = A00;
        tips2[base + 1u] = A10;
        return;
    }

    if (n_lanes == 3u) {
        uint base = idx * 3u;

        uint2 A00 = seeds2[base + 0u];
        uint2 A10 = seeds2[base + 1u];
        uint2 A20 = seeds2[base + 2u];
        uint2 A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N3();
            KECCAK_F1600();
        }

        tips2[base + 0u] = A00;
        tips2[base + 1u] = A10;
        tips2[base + 2u] = A20;
        return;
    }

    if (n_lanes == 1u) {
        uint base = idx;

        uint2 A00 = seeds2[base];
        uint2 A10, A20, A30, A40, A01, A11, A21, A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N1();
            KECCAK_F1600();
        }

        tips2[base] = A00;
        return;
    }

    if (n_lanes == 8u) {
        uint base = idx << 3;

        uint2 A00 = seeds2[base + 0u];
        uint2 A10 = seeds2[base + 1u];
        uint2 A20 = seeds2[base + 2u];
        uint2 A30 = seeds2[base + 3u];
        uint2 A40 = seeds2[base + 4u];
        uint2 A01 = seeds2[base + 5u];
        uint2 A11 = seeds2[base + 6u];
        uint2 A21 = seeds2[base + 7u];
        uint2 A31, A41;
        uint2 A02, A12, A22, A32, A42;
        uint2 A03, A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N8();
            KECCAK_F1600();
        }

        tips2[base + 0u] = A00;
        tips2[base + 1u] = A10;
        tips2[base + 2u] = A20;
        tips2[base + 3u] = A30;
        tips2[base + 4u] = A40;
        tips2[base + 5u] = A01;
        tips2[base + 6u] = A11;
        tips2[base + 7u] = A21;
        return;
    }

    if (n_lanes == 16u) {
        uint base = idx << 4;

        uint2 A00 = seeds2[base +  0u];
        uint2 A10 = seeds2[base +  1u];
        uint2 A20 = seeds2[base +  2u];
        uint2 A30 = seeds2[base +  3u];
        uint2 A40 = seeds2[base +  4u];
        uint2 A01 = seeds2[base +  5u];
        uint2 A11 = seeds2[base +  6u];
        uint2 A21 = seeds2[base +  7u];
        uint2 A31 = seeds2[base +  8u];
        uint2 A41 = seeds2[base +  9u];
        uint2 A02 = seeds2[base + 10u];
        uint2 A12 = seeds2[base + 11u];
        uint2 A22 = seeds2[base + 12u];
        uint2 A32 = seeds2[base + 13u];
        uint2 A42 = seeds2[base + 14u];
        uint2 A03 = seeds2[base + 15u];
        uint2 A13, A23, A33, A43;
        uint2 A04, A14, A24, A34, A44;

        for (uint step = 0u; step < ww; ++step) {
            RESET_N16();
            KECCAK_F1600();
        }

        tips2[base +  0u] = A00;
        tips2[base +  1u] = A10;
        tips2[base +  2u] = A20;
        tips2[base +  3u] = A30;
        tips2[base +  4u] = A40;
        tips2[base +  5u] = A01;
        tips2[base +  6u] = A11;
        tips2[base +  7u] = A21;
        tips2[base +  8u] = A31;
        tips2[base +  9u] = A41;
        tips2[base + 10u] = A02;
        tips2[base + 11u] = A12;
        tips2[base + 12u] = A22;
        tips2[base + 13u] = A32;
        tips2[base + 14u] = A42;
        tips2[base + 15u] = A03;
        return;
    }

    uint base = idx * n_lanes;

    uint2 A00 = Z2, A10 = Z2, A20 = Z2, A30 = Z2, A40 = Z2;
    uint2 A01 = Z2, A11 = Z2, A21 = Z2, A31 = Z2, A41 = Z2;
    uint2 A02 = Z2, A12 = Z2, A22 = Z2, A32 = Z2, A42 = Z2;
    uint2 A03 = Z2, A13 = Z2, A23 = Z2, A33 = Z2, A43 = Z2;
    uint2 A04 = Z2, A14 = Z2, A24 = Z2, A34 = Z2, A44 = Z2;

    if (n_lanes >  0u) A00 = seeds2[base +  0u];
    if (n_lanes >  1u) A10 = seeds2[base +  1u];
    if (n_lanes >  2u) A20 = seeds2[base +  2u];
    if (n_lanes >  3u) A30 = seeds2[base +  3u];
    if (n_lanes >  4u) A40 = seeds2[base +  4u];
    if (n_lanes >  5u) A01 = seeds2[base +  5u];
    if (n_lanes >  6u) A11 = seeds2[base +  6u];
    if (n_lanes >  7u) A21 = seeds2[base +  7u];
    if (n_lanes >  8u) A31 = seeds2[base +  8u];
    if (n_lanes >  9u) A41 = seeds2[base +  9u];
    if (n_lanes > 10u) A02 = seeds2[base + 10u];
    if (n_lanes > 11u) A12 = seeds2[base + 11u];
    if (n_lanes > 12u) A22 = seeds2[base + 12u];
    if (n_lanes > 13u) A32 = seeds2[base + 13u];
    if (n_lanes > 14u) A42 = seeds2[base + 14u];
    if (n_lanes > 15u) A03 = seeds2[base + 15u];

    for (uint step = 0u; step < ww; ++step) {
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
            case 9u:  A41 = D2; break;
            case 10u: A02 = D2; break;
            case 11u: A12 = D2; break;
            case 12u: A22 = D2; break;
            case 13u: A32 = D2; break;
            case 14u: A42 = D2; break;
            case 15u: A03 = D2; break;
            default:  A00 = D2; break;
        }

        A13 ^= F2;

        KECCAK_F1600();
    }

    if (n_lanes >  0u) tips2[base +  0u] = A00;
    if (n_lanes >  1u) tips2[base +  1u] = A10;
    if (n_lanes >  2u) tips2[base +  2u] = A20;
    if (n_lanes >  3u) tips2[base +  3u] = A30;
    if (n_lanes >  4u) tips2[base +  4u] = A40;
    if (n_lanes >  5u) tips2[base +  5u] = A01;
    if (n_lanes >  6u) tips2[base +  6u] = A11;
    if (n_lanes >  7u) tips2[base +  7u] = A21;
    if (n_lanes >  8u) tips2[base +  8u] = A31;
    if (n_lanes >  9u) tips2[base +  9u] = A41;
    if (n_lanes > 10u) tips2[base + 10u] = A02;
    if (n_lanes > 11u) tips2[base + 11u] = A12;
    if (n_lanes > 12u) tips2[base + 12u] = A22;
    if (n_lanes > 13u) tips2[base + 13u] = A32;
    if (n_lanes > 14u) tips2[base + 14u] = A42;
    if (n_lanes > 15u) tips2[base + 15u] = A03;
}