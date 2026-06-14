#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline __attribute__((always_inline))
void keccak_f1600(thread ulong (&A)[25]) {
    #pragma unroll
    for (uint r = 0u; r < 24u; ++r) {
        ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D0 = C4 ^ rotl64(C1, 1u);
        ulong D1 = C0 ^ rotl64(C2, 1u);
        ulong D2 = C1 ^ rotl64(C3, 1u);
        ulong D3 = C2 ^ rotl64(C4, 1u);
        ulong D4 = C3 ^ rotl64(C0, 1u);

        ulong B0  = A[0] ^ D0;
        ulong B1  = rotl64(A[6] ^ D1, 44u);
        ulong B2  = rotl64(A[12] ^ D2, 43u);
        ulong B3  = rotl64(A[18] ^ D3, 21u);
        ulong B4  = rotl64(A[24] ^ D4, 14u);

        ulong B5  = rotl64(A[3] ^ D3, 28u);
        ulong B6  = rotl64(A[9] ^ D4, 20u);
        ulong B7  = rotl64(A[10] ^ D0, 3u);
        ulong B8  = rotl64(A[16] ^ D1, 45u);
        ulong B9  = rotl64(A[22] ^ D2, 61u);

        ulong B10 = rotl64(A[1] ^ D1, 1u);
        ulong B11 = rotl64(A[7] ^ D2, 6u);
        ulong B12 = rotl64(A[13] ^ D3, 25u);
        ulong B13 = rotl64(A[19] ^ D4, 8u);
        ulong B14 = rotl64(A[20] ^ D0, 18u);

        ulong B15 = rotl64(A[4] ^ D4, 27u);
        ulong B16 = rotl64(A[5] ^ D0, 36u);
        ulong B17 = rotl64(A[11] ^ D1, 10u);
        ulong B18 = rotl64(A[17] ^ D2, 15u);
        ulong B19 = rotl64(A[23] ^ D3, 56u);

        ulong B20 = rotl64(A[2] ^ D2, 62u);
        ulong B21 = rotl64(A[8] ^ D3, 55u);
        ulong B22 = rotl64(A[14] ^ D4, 39u);
        ulong B23 = rotl64(A[15] ^ D0, 41u);
        ulong B24 = rotl64(A[21] ^ D1, 2u);

        A[0]  = B0  ^ ((~B1)  & B2) ^ KECCAK_RC[r];
        A[1]  = B1  ^ ((~B2)  & B3);
        A[2]  = B2  ^ ((~B3)  & B4);
        A[3]  = B3  ^ ((~B4)  & B0);
        A[4]  = B4  ^ ((~B0)  & B1);

        A[5]  = B5  ^ ((~B6)  & B7);
        A[6]  = B6  ^ ((~B7)  & B8);
        A[7]  = B7  ^ ((~B8)  & B9);
        A[8]  = B8  ^ ((~B9)  & B5);
        A[9]  = B9  ^ ((~B5)  & B6);

        A[10] = B10 ^ ((~B11) & B12);
        A[11] = B11 ^ ((~B12) & B13);
        A[12] = B12 ^ ((~B13) & B14);
        A[13] = B13 ^ ((~B14) & B10);
        A[14] = B14 ^ ((~B10) & B11);

        A[15] = B15 ^ ((~B16) & B17);
        A[16] = B16 ^ ((~B17) & B18);
        A[17] = B17 ^ ((~B18) & B19);
        A[18] = B18 ^ ((~B19) & B15);
        A[19] = B19 ^ ((~B15) & B16);

        A[20] = B20 ^ ((~B21) & B22);
        A[21] = B21 ^ ((~B22) & B23);
        A[22] = B22 ^ ((~B23) & B24);
        A[23] = B23 ^ ((~B24) & B20);
        A[24] = B24 ^ ((~B20) & B21);
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

    uint n_lanes = n_bytes >> 3;
    uint base = idx * n_lanes;
    uint w_val = w;

    thread ulong A[25];

    if (n_lanes == 2u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
        
        for (uint step = 0u; step < w_val; ++step) {
            A[2]  = 0x06ul;
            A[3]  = 0ul; A[4]  = 0ul; A[5]  = 0ul; A[6]  = 0ul;
            A[7]  = 0ul; A[8]  = 0ul; A[9]  = 0ul; A[10] = 0ul;
            A[11] = 0ul; A[12] = 0ul; A[13] = 0ul; A[14] = 0ul;
            A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];

    } else if (n_lanes == 4u) {
        A[0] = seeds[base + 0u];
        A[1] = seeds[base + 1u];
        A[2] = seeds[base + 2u];
        A[3] = seeds[base + 3u];
        
        for (uint step = 0u; step < w_val; ++step) {
            A[4]  = 0x06ul;
            A[5]  = 0ul; A[6]  = 0ul; A[7]  = 0ul; A[8]  = 0ul; 
            A[9]  = 0ul; A[10] = 0ul; A[11] = 0ul; A[12] = 0ul; 
            A[13] = 0ul; A[14] = 0ul; A[15] = 0ul; 
            A[16] = 0x8000000000000000ul;
            A[17] = 0ul; A[18] = 0ul; A[19] = 0ul; A[20] = 0ul;
            A[21] = 0ul; A[22] = 0ul; A[23] = 0ul; A[24] = 0ul;
            
            keccak_f1600(A);
        }
        
        tips[base + 0u] = A[0];
        tips[base + 1u] = A[1];
        tips[base + 2u] = A[2];
        tips[base + 3u] = A[3];

    } else {
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) A[i] = 0ul;
                }
            }
            A[n_lanes] ^= 0x06ul;
            A[16u]     ^= 0x8000000000000000ul;
            
            keccak_f1600(A);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = A[i];
        }
    }
}