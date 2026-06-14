#include <metal_stdlib>
using namespace metal;

constexpr constant ulong SHA3_DOMAIN_WORD = 0x06ul;
constexpr constant ulong SHA3_FINAL_PAD   = 0x8000000000000000ul;

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

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64u - (k))))

#define KECCAK_ROUND(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, \
                     B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, RC) \
    do { \
        ulong C0 = A0 ^ A5 ^ A10 ^ A15 ^ A20; \
        ulong C1 = A1 ^ A6 ^ A11 ^ A16 ^ A21; \
        ulong C2 = A2 ^ A7 ^ A12 ^ A17 ^ A22; \
        ulong C3 = A3 ^ A8 ^ A13 ^ A18 ^ A23; \
        ulong C4 = A4 ^ A9 ^ A14 ^ A19 ^ A24; \
        ulong D0 = C4 ^ ROTL64(C1, 1u); \
        ulong D1 = C0 ^ ROTL64(C2, 1u); \
        ulong D2 = C1 ^ ROTL64(C3, 1u); \
        ulong D3 = C2 ^ ROTL64(C4, 1u); \
        ulong D4 = C3 ^ ROTL64(C0, 1u); \
        ulong X0  = A0 ^ D0; \
        ulong X10 = ROTL64(A1 ^ D1,  1u); \
        ulong X20 = ROTL64(A2 ^ D2, 62u); \
        ulong X5  = ROTL64(A3 ^ D3, 28u); \
        ulong X15 = ROTL64(A4 ^ D4, 27u); \
        ulong X16 = ROTL64(A5 ^ D0, 36u); \
        ulong X1  = ROTL64(A6 ^ D1, 44u); \
        ulong X11 = ROTL64(A7 ^ D2,  6u); \
        ulong X21 = ROTL64(A8 ^ D3, 55u); \
        ulong X6  = ROTL64(A9 ^ D4, 20u); \
        ulong X7  = ROTL64(A10 ^ D0,  3u); \
        ulong X17 = ROTL64(A11 ^ D1, 10u); \
        ulong X2  = ROTL64(A12 ^ D2, 43u); \
        ulong X12 = ROTL64(A13 ^ D3, 25u); \
        ulong X22 = ROTL64(A14 ^ D4, 39u); \
        ulong X23 = ROTL64(A15 ^ D0, 41u); \
        ulong X8  = ROTL64(A16 ^ D1, 45u); \
        ulong X18 = ROTL64(A17 ^ D2, 15u); \
        ulong X3  = ROTL64(A18 ^ D3, 21u); \
        ulong X13 = ROTL64(A19 ^ D4,  8u); \
        ulong X14 = ROTL64(A20 ^ D0, 18u); \
        ulong X24 = ROTL64(A21 ^ D1,  2u); \
        ulong X9  = ROTL64(A22 ^ D2, 61u); \
        ulong X19 = ROTL64(A23 ^ D3, 56u); \
        ulong X4  = ROTL64(A24 ^ D4, 14u); \
        B0 = X0 ^ (~X1 & X2) ^ RC; \
        B1 = X1 ^ (~X2 & X3); \
        B2 = X2 ^ (~X3 & X4); \
        B3 = X3 ^ (~X4 & X0); \
        B4 = X4 ^ (~X0 & X1); \
        B5 = X5 ^ (~X6 & X7); \
        B6 = X6 ^ (~X7 & X8); \
        B7 = X7 ^ (~X8 & X9); \
        B8 = X8 ^ (~X9 & X5); \
        B9 = X9 ^ (~X5 & X6); \
        B10 = X10 ^ (~X11 & X12); \
        B11 = X11 ^ (~X12 & X13); \
        B12 = X12 ^ (~X13 & X14); \
        B13 = X13 ^ (~X14 & X10); \
        B14 = X14 ^ (~X10 & X11); \
        B15 = X15 ^ (~X16 & X17); \
        B16 = X16 ^ (~X17 & X18); \
        B17 = X17 ^ (~X18 & X19); \
        B18 = X18 ^ (~X19 & X15); \
        B19 = X19 ^ (~X15 & X16); \
        B20 = X20 ^ (~X21 & X22); \
        B21 = X21 ^ (~X22 & X23); \
        B22 = X22 ^ (~X23 & X24); \
        B23 = X23 ^ (~X24 & X20); \
        B24 = X24 ^ (~X20 & X21); \
    } while(0)

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
        
        ulong A0 = seeds[base + 0];
        ulong A1 = seeds[base + 1];
        
        for (uint step = 0; step < chain_len; ++step) {
            ulong A2 = SHA3_DOMAIN_WORD;
            ulong A3 = 0, A4 = 0, A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
            ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0, A15 = 0;
            ulong A16 = SHA3_FINAL_PAD;
            ulong A17 = 0, A18 = 0, A19 = 0, A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;
            
            ulong E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24;

            #pragma unroll
            for (uint r = 0; r < 24; r += 2) {
                KECCAK_ROUND(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,
                             E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, KECCAK_RC[r]);
                KECCAK_ROUND(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24,
                             A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, KECCAK_RC[r+1]);
            }
        }
        
        tips[base + 0] = A0;
        tips[base + 1] = A1;
        
    } else if (n_bytes == 32) {
        uint base = idx * 4;
        
        ulong A0 = seeds[base + 0];
        ulong A1 = seeds[base + 1];
        ulong A2 = seeds[base + 2];
        ulong A3 = seeds[base + 3];
        
        for (uint step = 0; step < chain_len; ++step) {
            ulong A4 = SHA3_DOMAIN_WORD;
            ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
            ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0, A15 = 0;
            ulong A16 = SHA3_FINAL_PAD;
            ulong A17 = 0, A18 = 0, A19 = 0, A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;
            
            ulong E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24;

            #pragma unroll
            for (uint r = 0; r < 24; r += 2) {
                KECCAK_ROUND(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,
                             E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, KECCAK_RC[r]);
                KECCAK_ROUND(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24,
                             A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, KECCAK_RC[r+1]);
            }
        }
        
        tips[base + 0] = A0;
        tips[base + 1] = A1;
        tips[base + 2] = A2;
        tips[base + 3] = A3;
        
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
            
            ulong A0=A[0], A1=A[1], A2=A[2], A3=A[3], A4=A[4];
            ulong A5=A[5], A6=A[6], A7=A[7], A8=A[8], A9=A[9];
            ulong A10=A[10], A11=A[11], A12=A[12], A13=A[13], A14=A[14];
            ulong A15=A[15], A16=A[16], A17=A[17], A18=A[18], A19=A[19];
            ulong A20=A[20], A21=A[21], A22=A[22], A23=A[23], A24=A[24];
            
            ulong E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24;

            #pragma unroll
            for (uint r = 0; r < 24; r += 2) {
                KECCAK_ROUND(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24,
                             E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24, KECCAK_RC[r]);
                KECCAK_ROUND(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20, E21, E22, E23, E24,
                             A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, KECCAK_RC[r+1]);
            }
            
            A[0]=A0; A[1]=A1; A[2]=A2; A[3]=A3; A[4]=A4;
            A[5]=A5; A[6]=A6; A[7]=A7; A[8]=A8; A[9]=A9;
            A[10]=A10; A[11]=A11; A[12]=A12; A[13]=A13; A[14]=A14;
            A[15]=A15; A[16]=A16; A[17]=A17; A[18]=A18; A[19]=A19;
            A[20]=A20; A[21]=A21; A[22]=A22; A[23]=A23; A[24]=A24;
        }
        
        for (uint k = 0; k < n_lanes; ++k) {
            tips[base + k] = A[k];
        }
    }
}