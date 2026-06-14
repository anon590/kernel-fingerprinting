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

inline ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= batch) return;

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong s[25] = {0};

    uint in_base = idx * msg_lanes;
    
    #pragma unroll(25)
    for (uint i = 0u; i < 25u; ++i) {
        if (i < msg_lanes) s[i] ^= in_data[in_base + i];
        if (i == msg_lanes) s[i] ^= (ulong)(domain & 0xFFu);
        if (i == rate_lanes - 1u) s[i] ^= 0x8000000000000000ul;
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        #pragma unroll(24)
        for (uint r = 0u; r < 24u; ++r) {
            // Theta
            ulong C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
            ulong C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
            ulong C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
            ulong C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
            ulong C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

            ulong D0 = C4 ^ rotl64(C1, 1u);
            ulong D1 = C0 ^ rotl64(C2, 1u);
            ulong D2 = C1 ^ rotl64(C3, 1u);
            ulong D3 = C2 ^ rotl64(C4, 1u);
            ulong D4 = C3 ^ rotl64(C0, 1u);

            // Rho & Pi combined
            ulong B0  = s[0]  ^ D0;
            ulong B10 = rotl64(s[1]  ^ D1, 1u);
            ulong B7  = rotl64(s[10] ^ D0, 3u);
            ulong B11 = rotl64(s[7]  ^ D2, 6u);
            ulong B17 = rotl64(s[11] ^ D1, 10u);
            ulong B18 = rotl64(s[17] ^ D2, 15u);
            ulong B3  = rotl64(s[18] ^ D3, 21u);
            ulong B5  = rotl64(s[3]  ^ D3, 28u);
            ulong B16 = rotl64(s[5]  ^ D0, 36u);
            ulong B8  = rotl64(s[16] ^ D1, 45u);
            ulong B21 = rotl64(s[8]  ^ D3, 55u);
            ulong B24 = rotl64(s[21] ^ D1, 2u);
            ulong B4  = rotl64(s[24] ^ D4, 14u);
            ulong B15 = rotl64(s[4]  ^ D4, 27u);
            ulong B23 = rotl64(s[15] ^ D0, 41u);
            ulong B19 = rotl64(s[23] ^ D3, 56u);
            ulong B13 = rotl64(s[19] ^ D4, 8u);
            ulong B12 = rotl64(s[13] ^ D3, 25u);
            ulong B2  = rotl64(s[12] ^ D2, 43u);
            ulong B20 = rotl64(s[2]  ^ D2, 62u);
            ulong B14 = rotl64(s[20] ^ D0, 18u);
            ulong B22 = rotl64(s[14] ^ D4, 39u);
            ulong B9  = rotl64(s[22] ^ D2, 61u);
            ulong B6  = rotl64(s[9]  ^ D4, 20u);
            ulong B1  = rotl64(s[6]  ^ D1, 44u);

            // Chi & Iota
            s[0]  = B0  ^ ((~B1)  & B2)  ^ KECCAK_RC[r];
            s[1]  = B1  ^ ((~B2)  & B3);
            s[2]  = B2  ^ ((~B3)  & B4);
            s[3]  = B3  ^ ((~B4)  & B0);
            s[4]  = B4  ^ ((~B0)  & B1);

            s[5]  = B5  ^ ((~B6)  & B7);
            s[6]  = B6  ^ ((~B7)  & B8);
            s[7]  = B7  ^ ((~B8)  & B9);
            s[8]  = B8  ^ ((~B9)  & B10);
            s[9]  = B9  ^ ((~B10) & B5);

            s[10] = B10 ^ ((~B11) & B12);
            s[11] = B11 ^ ((~B12) & B13);
            s[12] = B12 ^ ((~B13) & B14);
            s[13] = B13 ^ ((~B14) & B10);
            s[14] = B14 ^ ((~B10) & B11);

            s[15] = B15 ^ ((~B16) & B17);
            s[16] = B16 ^ ((~B17) & B18);
            s[17] = B17 ^ ((~B18) & B19);
            s[18] = B18 ^ ((~B19) & B15);
            s[19] = B19 ^ ((~B15) & B16);

            s[20] = B20 ^ ((~B21) & B22);
            s[21] = B21 ^ ((~B22) & B23);
            s[22] = B22 ^ ((~B23) & B24);
            s[23] = B23 ^ ((~B24) & B20);
            s[24] = B24 ^ ((~B20) & B21);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        #pragma unroll(25)
        for (uint j = 0u; j < 25u; ++j) {
            if (j < take) {
                out_data[out_base + written + j] = s[j];
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}