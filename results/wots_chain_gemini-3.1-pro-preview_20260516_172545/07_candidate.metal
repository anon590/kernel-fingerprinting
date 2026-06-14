#include <metal_stdlib>
using namespace metal;

constexpr constant ulong SHA3_DOMAIN_WORD = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD   = 0x8000000000000000ul;
constexpr constant ulong KECCAK_RC_0      = 0x0000000000000001ul;
constexpr constant ulong KECCAK_RC_23     = 0x8000000080008008ul;

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul
};

inline ulong rotl64_opt(ulong x, uint k) {
    if (k == 0u) return x;
    return (x << k) | (x >> (64u - k));
}

struct State16 { ulong a0, a1; };
inline __attribute__((always_inline))
State16 keccak_step_16(ulong a0, ulong a1) {
    ulong A[25];

    // --- Hardcoded Round 0 ---
    ulong C0 = a0;
    ulong C1 = a1 ^ SHA3_FINAL_PAD;
    
    ulong D0 = rotl64_opt(C1, 1u);
    ulong D1 = C0 ^ 0x000000000000000Cul; // C2 = 0x06
    ulong D2 = C1;
    ulong D3 = 0x0000000000000006ul;
    ulong D4 = rotl64_opt(C0, 1u);

    ulong B0  = C0 ^ D0;
    ulong B10 = rotl64_opt(a1 ^ D1, 1u);
    ulong B20 = rotl64_opt(0x06ul ^ D2, 62u);
    ulong B5  = rotl64_opt(0x06ul, 28u);
    ulong B15 = rotl64_opt(D4, 27u);

    ulong B16 = rotl64_opt(D0, 36u);
    ulong B1  = rotl64_opt(D1, 44u);
    ulong B11 = rotl64_opt(D2, 6u);
    ulong B21 = rotl64_opt(0x06ul, 55u);
    ulong B6  = rotl64_opt(D4, 20u);

    ulong B7  = rotl64_opt(D0, 3u);
    ulong B17 = rotl64_opt(D1, 10u);
    ulong B2  = rotl64_opt(D2, 43u);
    ulong B12 = rotl64_opt(0x06ul, 25u);
    ulong B22 = rotl64_opt(D4, 39u);

    ulong B23 = rotl64_opt(D0, 41u);
    ulong B8  = rotl64_opt(SHA3_FINAL_PAD ^ D1, 45u);
    ulong B18 = rotl64_opt(D2, 15u);
    ulong B3  = rotl64_opt(0x06ul, 21u);
    ulong B13 = rotl64_opt(D4, 8u);

    ulong B14 = rotl64_opt(D0, 18u);
    ulong B24 = rotl64_opt(D1, 2u);
    ulong B9  = rotl64_opt(D2, 61u);
    ulong B19 = rotl64_opt(0x06ul, 56u);
    ulong B4  = rotl64_opt(D4, 14u);

    A[0]  = B0 ^ (~B1 & B2);
    A[1]  = B1 ^ (~B2 & B3);
    A[2]  = B2 ^ (~B3 & B4);
    A[3]  = B3 ^ (~B4 & B0);
    A[4]  = B4 ^ (~B0 & B1);
    
    A[5]  = B5 ^ (~B6 & B7);
    A[6]  = B6 ^ (~B7 & B8);
    A[7]  = B7 ^ (~B8 & B9);
    A[8]  = B8 ^ (~B9 & B5);
    A[9]  = B9 ^ (~B5 & B6);

    A[10] = B10 ^ (~B11 & B12);
    A[11] = B11 ^ (~B12 & B13);
    A[12] = B12 ^ (~B13 & B14);
    A[13] = B13 ^ (~B14 & B10);
    A[14] = B14 ^ (~B10 & B11);

    A[15] = B15 ^ (~B16 & B17);
    A[16] = B16 ^ (~B17 & B18);
    A[17] = B17 ^ (~B18 & B19);
    A[18] = B18 ^ (~B19 & B15);
    A[19] = B19 ^ (~B15 & B16);

    A[20] = B20 ^ (~B21 & B22);
    A[21] = B21 ^ (~B22 & B23);
    A[22] = B22 ^ (~B23 & B24);
    A[23] = B23 ^ (~B24 & B20);
    A[24] = B24 ^ (~B20 & B21);

    A[0] ^= KECCAK_RC_0;

    // --- Rounds 1 to 22 ---
    #pragma unroll
    for (uint r = 1; r < 23; ++r) {
        ulong C_r[5];
        C_r[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        C_r[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        C_r[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        C_r[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        C_r[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D_r[5];
        D_r[0] = C_r[4] ^ rotl64_opt(C_r[1], 1u);
        D_r[1] = C_r[0] ^ rotl64_opt(C_r[2], 1u);
        D_r[2] = C_r[1] ^ rotl64_opt(C_r[3], 1u);
        D_r[3] = C_r[2] ^ rotl64_opt(C_r[4], 1u);
        D_r[4] = C_r[3] ^ rotl64_opt(C_r[0], 1u);

        ulong B_r[25];
        B_r[ 0] = A[ 0] ^ D_r[0];
        B_r[10] = rotl64_opt(A[ 1] ^ D_r[1],  1u);
        B_r[20] = rotl64_opt(A[ 2] ^ D_r[2], 62u);
        B_r[ 5] = rotl64_opt(A[ 3] ^ D_r[3], 28u);
        B_r[15] = rotl64_opt(A[ 4] ^ D_r[4], 27u);

        B_r[16] = rotl64_opt(A[ 5] ^ D_r[0], 36u);
        B_r[ 1] = rotl64_opt(A[ 6] ^ D_r[1], 44u);
        B_r[11] = rotl64_opt(A[ 7] ^ D_r[2],  6u);
        B_r[21] = rotl64_opt(A[ 8] ^ D_r[3], 55u);
        B_r[ 6] = rotl64_opt(A[ 9] ^ D_r[4], 20u);

        B_r[ 7] = rotl64_opt(A[10] ^ D_r[0],  3u);
        B_r[17] = rotl64_opt(A[11] ^ D_r[1], 10u);
        B_r[ 2] = rotl64_opt(A[12] ^ D_r[2], 43u);
        B_r[12] = rotl64_opt(A[13] ^ D_r[3], 25u);
        B_r[22] = rotl64_opt(A[14] ^ D_r[4], 39u);

        B_r[23] = rotl64_opt(A[15] ^ D_r[0], 41u);
        B_r[ 8] = rotl64_opt(A[16] ^ D_r[1], 45u);
        B_r[18] = rotl64_opt(A[17] ^ D_r[2], 15u);
        B_r[ 3] = rotl64_opt(A[18] ^ D_r[3], 21u);
        B_r[13] = rotl64_opt(A[19] ^ D_r[4],  8u);

        B_r[14] = rotl64_opt(A[20] ^ D_r[0], 18u);
        B_r[24] = rotl64_opt(A[21] ^ D_r[1],  2u);
        B_r[ 9] = rotl64_opt(A[22] ^ D_r[2], 61u);
        B_r[19] = rotl64_opt(A[23] ^ D_r[3], 56u);
        B_r[ 4] = rotl64_opt(A[24] ^ D_r[4], 14u);

        #pragma unroll
        for (uint y = 0; y < 25; y += 5) {
            A[y + 0] = B_r[y + 0] ^ (~B_r[y + 1] & B_r[y + 2]);
            A[y + 1] = B_r[y + 1] ^ (~B_r[y + 2] & B_r[y + 3]);
            A[y + 2] = B_r[y + 2] ^ (~B_r[y + 3] & B_r[y + 4]);
            A[y + 3] = B_r[y + 3] ^ (~B_r[y + 4] & B_r[y + 0]);
            A[y + 4] = B_r[y + 4] ^ (~B_r[y + 0] & B_r[y + 1]);
        }

        A[0] ^= KECCAK_RC[r];
    }

    // --- Round 23 Truncated ---
    ulong C_23[5];
    C_23[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
    C_23[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
    C_23[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
    C_23[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
    C_23[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

    ulong D_23[4];
    D_23[0] = C_23[4] ^ rotl64_opt(C_23[1], 1u);
    D_23[1] = C_23[0] ^ rotl64_opt(C_23[2], 1u);
    D_23[2] = C_23[1] ^ rotl64_opt(C_23[3], 1u);
    D_23[3] = C_23[2] ^ rotl64_opt(C_23[4], 1u);

    ulong B0_23 = A[0] ^ D_23[0];
    ulong B1_23 = rotl64_opt(A[6] ^ D_23[1], 44u);
    ulong B2_23 = rotl64_opt(A[12] ^ D_23[2], 43u);
    ulong B3_23 = rotl64_opt(A[18] ^ D_23[3], 21u);

    ulong a0_out = (B0_23 ^ (~B1_23 & B2_23)) ^ KECCAK_RC_23;
    ulong a1_out = B1_23 ^ (~B2_23 & B3_23);
    return {a0_out, a1_out};
}

struct State32 { ulong a0, a1, a2, a3; };
inline __attribute__((always_inline))
State32 keccak_step_32(ulong a0, ulong a1, ulong a2, ulong a3) {
    ulong A[25];

    // --- Hardcoded Round 0 ---
    ulong C0 = a0;
    ulong C1 = a1 ^ SHA3_FINAL_PAD;
    ulong C2 = a2;
    ulong C3 = a3;
    
    ulong D0 = 0x06ul ^ rotl64_opt(C1, 1u);
    ulong D1 = C0 ^ rotl64_opt(C2, 1u);
    ulong D2 = C1 ^ rotl64_opt(C3, 1u);
    ulong D3 = C2 ^ 0x000000000000000Cul;
    ulong D4 = C3 ^ rotl64_opt(C0, 1u);

    ulong B0  = C0 ^ D0;
    ulong B10 = rotl64_opt(a1 ^ D1, 1u);
    ulong B20 = rotl64_opt(a2 ^ D2, 62u);
    ulong B5  = rotl64_opt(a3 ^ D3, 28u);
    ulong B15 = rotl64_opt(0x06ul ^ D4, 27u);

    ulong B16 = rotl64_opt(D0, 36u);
    ulong B1  = rotl64_opt(D1, 44u);
    ulong B11 = rotl64_opt(D2, 6u);
    ulong B21 = rotl64_opt(D3, 55u);
    ulong B6  = rotl64_opt(D4, 20u);

    ulong B7  = rotl64_opt(D0, 3u);
    ulong B17 = rotl64_opt(D1, 10u);
    ulong B2  = rotl64_opt(D2, 43u);
    ulong B12 = rotl64_opt(D3, 25u);
    ulong B22 = rotl64_opt(D4, 39u);

    ulong B23 = rotl64_opt(D0, 41u);
    ulong B8  = rotl64_opt(SHA3_FINAL_PAD ^ D1, 45u);
    ulong B18 = rotl64_opt(D2, 15u);
    ulong B3  = rotl64_opt(D3, 21u);
    ulong B13 = rotl64_opt(D4, 8u);

    ulong B14 = rotl64_opt(D0, 18u);
    ulong B24 = rotl64_opt(D1, 2u);
    ulong B9  = rotl64_opt(D2, 61u);
    ulong B19 = rotl64_opt(D3, 56u);
    ulong B4  = rotl64_opt(D4, 14u);

    A[0]  = B0 ^ (~B1 & B2);
    A[1]  = B1 ^ (~B2 & B3);
    A[2]  = B2 ^ (~B3 & B4);
    A[3]  = B3 ^ (~B4 & B0);
    A[4]  = B4 ^ (~B0 & B1);

    A[5]  = B5 ^ (~B6 & B7);
    A[6]  = B6 ^ (~B7 & B8);
    A[7]  = B7 ^ (~B8 & B9);
    A[8]  = B8 ^ (~B9 & B5);
    A[9]  = B9 ^ (~B5 & B6);

    A[10] = B10 ^ (~B11 & B12);
    A[11] = B11 ^ (~B12 & B13);
    A[12] = B12 ^ (~B13 & B14);
    A[13] = B13 ^ (~B14 & B10);
    A[14] = B14 ^ (~B10 & B11);

    A[15] = B15 ^ (~B16 & B17);
    A[16] = B16 ^ (~B17 & B18);
    A[17] = B17 ^ (~B18 & B19);
    A[18] = B18 ^ (~B19 & B15);
    A[19] = B19 ^ (~B15 & B16);

    A[20] = B20 ^ (~B21 & B22);
    A[21] = B21 ^ (~B22 & B23);
    A[22] = B22 ^ (~B23 & B24);
    A[23] = B23 ^ (~B24 & B20);
    A[24] = B24 ^ (~B20 & B21);

    A[0] ^= KECCAK_RC_0;

    // --- Rounds 1 to 22 ---
    #pragma unroll
    for (uint r = 1; r < 23; ++r) {
        ulong C_r[5];
        C_r[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        C_r[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        C_r[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        C_r[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        C_r[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D_r[5];
        D_r[0] = C_r[4] ^ rotl64_opt(C_r[1], 1u);
        D_r[1] = C_r[0] ^ rotl64_opt(C_r[2], 1u);
        D_r[2] = C_r[1] ^ rotl64_opt(C_r[3], 1u);
        D_r[3] = C_r[2] ^ rotl64_opt(C_r[4], 1u);
        D_r[4] = C_r[3] ^ rotl64_opt(C_r[0], 1u);

        ulong B_r[25];
        B_r[ 0] = A[ 0] ^ D_r[0];
        B_r[10] = rotl64_opt(A[ 1] ^ D_r[1],  1u);
        B_r[20] = rotl64_opt(A[ 2] ^ D_r[2], 62u);
        B_r[ 5] = rotl64_opt(A[ 3] ^ D_r[3], 28u);
        B_r[15] = rotl64_opt(A[ 4] ^ D_r[4], 27u);

        B_r[16] = rotl64_opt(A[ 5] ^ D_r[0], 36u);
        B_r[ 1] = rotl64_opt(A[ 6] ^ D_r[1], 44u);
        B_r[11] = rotl64_opt(A[ 7] ^ D_r[2],  6u);
        B_r[21] = rotl64_opt(A[ 8] ^ D_r[3], 55u);
        B_r[ 6] = rotl64_opt(A[ 9] ^ D_r[4], 20u);

        B_r[ 7] = rotl64_opt(A[10] ^ D_r[0],  3u);
        B_r[17] = rotl64_opt(A[11] ^ D_r[1], 10u);
        B_r[ 2] = rotl64_opt(A[12] ^ D_r[2], 43u);
        B_r[12] = rotl64_opt(A[13] ^ D_r[3], 25u);
        B_r[22] = rotl64_opt(A[14] ^ D_r[4], 39u);

        B_r[23] = rotl64_opt(A[15] ^ D_r[0], 41u);
        B_r[ 8] = rotl64_opt(A[16] ^ D_r[1], 45u);
        B_r[18] = rotl64_opt(A[17] ^ D_r[2], 15u);
        B_r[ 3] = rotl64_opt(A[18] ^ D_r[3], 21u);
        B_r[13] = rotl64_opt(A[19] ^ D_r[4],  8u);

        B_r[14] = rotl64_opt(A[20] ^ D_r[0], 18u);
        B_r[24] = rotl64_opt(A[21] ^ D_r[1],  2u);
        B_r[ 9] = rotl64_opt(A[22] ^ D_r[2], 61u);
        B_r[19] = rotl64_opt(A[23] ^ D_r[3], 56u);
        B_r[ 4] = rotl64_opt(A[24] ^ D_r[4], 14u);

        #pragma unroll
        for (uint y = 0; y < 25; y += 5) {
            A[y + 0] = B_r[y + 0] ^ (~B_r[y + 1] & B_r[y + 2]);
            A[y + 1] = B_r[y + 1] ^ (~B_r[y + 2] & B_r[y + 3]);
            A[y + 2] = B_r[y + 2] ^ (~B_r[y + 3] & B_r[y + 4]);
            A[y + 3] = B_r[y + 3] ^ (~B_r[y + 4] & B_r[y + 0]);
            A[y + 4] = B_r[y + 4] ^ (~B_r[y + 0] & B_r[y + 1]);
        }

        A[0] ^= KECCAK_RC[r];
    }

    // --- Round 23 Truncated ---
    ulong C_23[5];
    C_23[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
    C_23[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
    C_23[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
    C_23[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
    C_23[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

    ulong D_23[5];
    D_23[0] = C_23[4] ^ rotl64_opt(C_23[1], 1u);
    D_23[1] = C_23[0] ^ rotl64_opt(C_23[2], 1u);
    D_23[2] = C_23[1] ^ rotl64_opt(C_23[3], 1u);
    D_23[3] = C_23[2] ^ rotl64_opt(C_23[4], 1u);
    D_23[4] = C_23[3] ^ rotl64_opt(C_23[0], 1u);

    ulong B0_23 = A[0] ^ D_23[0];
    ulong B1_23 = rotl64_opt(A[6] ^ D_23[1], 44u);
    ulong B2_23 = rotl64_opt(A[12] ^ D_23[2], 43u);
    ulong B3_23 = rotl64_opt(A[18] ^ D_23[3], 21u);
    ulong B4_23 = rotl64_opt(A[24] ^ D_23[4], 14u);

    ulong a0_out = (B0_23 ^ (~B1_23 & B2_23)) ^ KECCAK_RC_23;
    ulong a1_out = B1_23 ^ (~B2_23 & B3_23);
    ulong a2_out = B2_23 ^ (~B3_23 & B4_23);
    ulong a3_out = B3_23 ^ (~B4_23 & B0_23);
    return {a0_out, a1_out, a2_out, a3_out};
}

inline __attribute__((always_inline))
void keccak_f1600_full(thread ulong (&A)[25]) {
    #pragma unroll
    for (uint r = 0; r < 24; ++r) {
        ulong C[5];
        C[0] = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        C[1] = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        C[2] = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        C[3] = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        C[4] = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D[5];
        D[0] = C[4] ^ rotl64_opt(C[1], 1u);
        D[1] = C[0] ^ rotl64_opt(C[2], 1u);
        D[2] = C[1] ^ rotl64_opt(C[3], 1u);
        D[3] = C[2] ^ rotl64_opt(C[4], 1u);
        D[4] = C[3] ^ rotl64_opt(C[0], 1u);

        ulong B[25];
        B[ 0] = A[ 0] ^ D[0];
        B[10] = rotl64_opt(A[ 1] ^ D[1],  1u);
        B[20] = rotl64_opt(A[ 2] ^ D[2], 62u);
        B[ 5] = rotl64_opt(A[ 3] ^ D[3], 28u);
        B[15] = rotl64_opt(A[ 4] ^ D[4], 27u);

        B[16] = rotl64_opt(A[ 5] ^ D[0], 36u);
        B[ 1] = rotl64_opt(A[ 6] ^ D[1], 44u);
        B[11] = rotl64_opt(A[ 7] ^ D[2],  6u);
        B[21] = rotl64_opt(A[ 8] ^ D[3], 55u);
        B[ 6] = rotl64_opt(A[ 9] ^ D[4], 20u);

        B[ 7] = rotl64_opt(A[10] ^ D[0],  3u);
        B[17] = rotl64_opt(A[11] ^ D[1], 10u);
        B[ 2] = rotl64_opt(A[12] ^ D[2], 43u);
        B[12] = rotl64_opt(A[13] ^ D[3], 25u);
        B[22] = rotl64_opt(A[14] ^ D[4], 39u);

        B[23] = rotl64_opt(A[15] ^ D[0], 41u);
        B[ 8] = rotl64_opt(A[16] ^ D[1], 45u);
        B[18] = rotl64_opt(A[17] ^ D[2], 15u);
        B[ 3] = rotl64_opt(A[18] ^ D[3], 21u);
        B[13] = rotl64_opt(A[19] ^ D[4],  8u);

        B[14] = rotl64_opt(A[20] ^ D[0], 18u);
        B[24] = rotl64_opt(A[21] ^ D[1],  2u);
        B[ 9] = rotl64_opt(A[22] ^ D[2], 61u);
        B[19] = rotl64_opt(A[23] ^ D[3], 56u);
        B[ 4] = rotl64_opt(A[24] ^ D[4], 14u);

        #pragma unroll
        for (uint y = 0; y < 25; y += 5) {
            A[y + 0] = B[y + 0] ^ (~B[y + 1] & B[y + 2]);
            A[y + 1] = B[y + 1] ^ (~B[y + 2] & B[y + 3]);
            A[y + 2] = B[y + 2] ^ (~B[y + 3] & B[y + 4]);
            A[y + 3] = B[y + 3] ^ (~B[y + 4] & B[y + 0]);
            A[y + 4] = B[y + 4] ^ (~B[y + 0] & B[y + 1]);
        }
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
    
    uint chain_len = w;

    if (n_bytes == 16) {
        uint base = idx * 2;
        State16 st = {seeds[base], seeds[base + 1]};
        
        for (uint step = 0; step < chain_len; ++step) {
            st = keccak_step_16(st.a0, st.a1);
        }
        
        tips[base]     = st.a0;
        tips[base + 1] = st.a1;
        
    } else if (n_bytes == 32) {
        uint base = idx * 4;
        State32 st = {seeds[base], seeds[base + 1], seeds[base + 2], seeds[base + 3]};
        
        for (uint step = 0; step < chain_len; ++step) {
            st = keccak_step_32(st.a0, st.a1, st.a2, st.a3);
        }
        
        tips[base]     = st.a0;
        tips[base + 1] = st.a1;
        tips[base + 2] = st.a2;
        tips[base + 3] = st.a3;
        
    } else {
        uint n_lanes = n_bytes >> 3;
        ulong A[25];
        uint base = idx * n_lanes;
        
        for (uint k = 0; k < n_lanes; ++k) {
            A[k] = seeds[base + k];
        }
        
        for (uint step = 0; step < chain_len; ++step) {
            for (uint i = n_lanes; i < 25; ++i) A[i] = 0ul;
            A[n_lanes] ^= SHA3_DOMAIN_WORD;
            A[16] ^= SHA3_FINAL_PAD;
            keccak_f1600_full(A);
        }
        
        for (uint k = 0; k < n_lanes; ++k) {
            tips[base + k] = A[k];
        }
    }
}