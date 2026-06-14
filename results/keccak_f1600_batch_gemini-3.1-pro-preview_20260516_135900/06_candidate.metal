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

inline __attribute__((always_inline)) ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

// In-place round transformation minimizing live ranges to eliminate register spilling.
#define ROUND(RC) do { \
    ulong C0 = A0 ^ A5 ^ A10 ^ A15 ^ A20; \
    ulong C1 = A1 ^ A6 ^ A11 ^ A16 ^ A21; \
    ulong C2 = A2 ^ A7 ^ A12 ^ A17 ^ A22; \
    ulong C3 = A3 ^ A8 ^ A13 ^ A18 ^ A23; \
    ulong C4 = A4 ^ A9 ^ A14 ^ A19 ^ A24; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    A0 ^= D0; A5 ^= D0; A10 ^= D0; A15 ^= D0; A20 ^= D0; \
    A1 ^= D1; A6 ^= D1; A11 ^= D1; A16 ^= D1; A21 ^= D1; \
    A2 ^= D2; A7 ^= D2; A12 ^= D2; A17 ^= D2; A22 ^= D2; \
    A3 ^= D3; A8 ^= D3; A13 ^= D3; A18 ^= D3; A23 ^= D3; \
    A4 ^= D4; A9 ^= D4; A14 ^= D4; A19 ^= D4; A24 ^= D4; \
    \
    ulong temp = A1; \
    ulong current; \
    current = A10; A10 = rotl64(temp, 1u);  temp = current; \
    current = A7;  A7  = rotl64(temp, 3u);  temp = current; \
    current = A11; A11 = rotl64(temp, 6u);  temp = current; \
    current = A17; A17 = rotl64(temp, 10u); temp = current; \
    current = A18; A18 = rotl64(temp, 15u); temp = current; \
    current = A3;  A3  = rotl64(temp, 21u); temp = current; \
    current = A5;  A5  = rotl64(temp, 28u); temp = current; \
    current = A16; A16 = rotl64(temp, 36u); temp = current; \
    current = A8;  A8  = rotl64(temp, 45u); temp = current; \
    current = A21; A21 = rotl64(temp, 55u); temp = current; \
    current = A24; A24 = rotl64(temp, 2u);  temp = current; \
    current = A4;  A4  = rotl64(temp, 14u); temp = current; \
    current = A15; A15 = rotl64(temp, 27u); temp = current; \
    current = A23; A23 = rotl64(temp, 41u); temp = current; \
    current = A19; A19 = rotl64(temp, 56u); temp = current; \
    current = A13; A13 = rotl64(temp, 8u);  temp = current; \
    current = A12; A12 = rotl64(temp, 25u); temp = current; \
    current = A2;  A2  = rotl64(temp, 43u); temp = current; \
    current = A20; A20 = rotl64(temp, 62u); temp = current; \
    current = A14; A14 = rotl64(temp, 18u); temp = current; \
    current = A22; A22 = rotl64(temp, 39u); temp = current; \
    current = A9;  A9  = rotl64(temp, 61u); temp = current; \
    current = A6;  A6  = rotl64(temp, 20u); temp = current; \
    A1 = rotl64(temp, 44u); \
    \
    ulong T0, T1, T2, T3, T4; \
    \
    T0 = A0; T1 = A1; T2 = A2; T3 = A3; T4 = A4; \
    A0 = T0 ^ ((~T1) & T2) ^ RC; \
    A1 = T1 ^ ((~T2) & T3); \
    A2 = T2 ^ ((~T3) & T4); \
    A3 = T3 ^ ((~T4) & T0); \
    A4 = T4 ^ ((~T0) & T1); \
    \
    T0 = A5; T1 = A6; T2 = A7; T3 = A8; T4 = A9; \
    A5 = T0 ^ ((~T1) & T2); \
    A6 = T1 ^ ((~T2) & T3); \
    A7 = T2 ^ ((~T3) & T4); \
    A8 = T3 ^ ((~T4) & T0); \
    A9 = T4 ^ ((~T0) & T1); \
    \
    T0 = A10; T1 = A11; T2 = A12; T3 = A13; T4 = A14; \
    A10 = T0 ^ ((~T1) & T2); \
    A11 = T1 ^ ((~T2) & T3); \
    A12 = T2 ^ ((~T3) & T4); \
    A13 = T3 ^ ((~T4) & T0); \
    A14 = T4 ^ ((~T0) & T1); \
    \
    T0 = A15; T1 = A16; T2 = A17; T3 = A18; T4 = A19; \
    A15 = T0 ^ ((~T1) & T2); \
    A16 = T1 ^ ((~T2) & T3); \
    A17 = T2 ^ ((~T3) & T4); \
    A18 = T3 ^ ((~T4) & T0); \
    A19 = T4 ^ ((~T0) & T1); \
    \
    T0 = A20; T1 = A21; T2 = A22; T3 = A23; T4 = A24; \
    A20 = T0 ^ ((~T1) & T2); \
    A21 = T1 ^ ((~T2) & T3); \
    A22 = T2 ^ ((~T3) & T4); \
    A23 = T3 ^ ((~T4) & T0); \
    A24 = T4 ^ ((~T0) & T1); \
} while(0)

#define APPLY_PAD(LANE, VAL) do { \
    A0  ^= (LANE == 0)  ? (VAL) : 0; \
    A1  ^= (LANE == 1)  ? (VAL) : 0; \
    A2  ^= (LANE == 2)  ? (VAL) : 0; \
    A3  ^= (LANE == 3)  ? (VAL) : 0; \
    A4  ^= (LANE == 4)  ? (VAL) : 0; \
    A5  ^= (LANE == 5)  ? (VAL) : 0; \
    A6  ^= (LANE == 6)  ? (VAL) : 0; \
    A7  ^= (LANE == 7)  ? (VAL) : 0; \
    A8  ^= (LANE == 8)  ? (VAL) : 0; \
    A9  ^= (LANE == 9)  ? (VAL) : 0; \
    A10 ^= (LANE == 10) ? (VAL) : 0; \
    A11 ^= (LANE == 11) ? (VAL) : 0; \
    A12 ^= (LANE == 12) ? (VAL) : 0; \
    A13 ^= (LANE == 13) ? (VAL) : 0; \
    A14 ^= (LANE == 14) ? (VAL) : 0; \
    A15 ^= (LANE == 15) ? (VAL) : 0; \
    A16 ^= (LANE == 16) ? (VAL) : 0; \
    A17 ^= (LANE == 17) ? (VAL) : 0; \
    A18 ^= (LANE == 18) ? (VAL) : 0; \
    A19 ^= (LANE == 19) ? (VAL) : 0; \
    A20 ^= (LANE == 20) ? (VAL) : 0; \
    A21 ^= (LANE == 21) ? (VAL) : 0; \
    A22 ^= (LANE == 22) ? (VAL) : 0; \
    A23 ^= (LANE == 23) ? (VAL) : 0; \
    A24 ^= (LANE == 24) ? (VAL) : 0; \
} while(0)

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

    ulong A0 = 0, A1 = 0, A2 = 0, A3 = 0, A4 = 0;
    ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
    ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
    ulong A15 = 0, A16 = 0, A17 = 0, A18 = 0, A19 = 0;
    ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;

    uint msg_lanes  = msg_bytes >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes >> 3;
    
    uint in_base = idx * msg_lanes;
    
    if (msg_lanes > 20) A20 = in_data[in_base + 20];
    if (msg_lanes > 19) A19 = in_data[in_base + 19];
    if (msg_lanes > 18) A18 = in_data[in_base + 18];
    if (msg_lanes > 17) A17 = in_data[in_base + 17];
    if (msg_lanes > 16) A16 = in_data[in_base + 16];
    if (msg_lanes > 15) A15 = in_data[in_base + 15];
    if (msg_lanes > 14) A14 = in_data[in_base + 14];
    if (msg_lanes > 13) A13 = in_data[in_base + 13];
    if (msg_lanes > 12) A12 = in_data[in_base + 12];
    if (msg_lanes > 11) A11 = in_data[in_base + 11];
    if (msg_lanes > 10) A10 = in_data[in_base + 10];
    if (msg_lanes > 9)  A9  = in_data[in_base + 9];
    if (msg_lanes > 8)  A8  = in_data[in_base + 8];
    if (msg_lanes > 7)  A7  = in_data[in_base + 7];
    if (msg_lanes > 6)  A6  = in_data[in_base + 6];
    if (msg_lanes > 5)  A5  = in_data[in_base + 5];
    if (msg_lanes > 4)  A4  = in_data[in_base + 4];
    if (msg_lanes > 3)  A3  = in_data[in_base + 3];
    if (msg_lanes > 2)  A2  = in_data[in_base + 2];
    if (msg_lanes > 1)  A1  = in_data[in_base + 1];
    if (msg_lanes > 0)  A0  = in_data[in_base + 0];

    APPLY_PAD(msg_lanes, (ulong)(domain & 0xFFu));
    APPLY_PAD(rate_lanes - 1u, 0x8000000000000000ul);

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        #pragma unroll(24)
        for (uint r = 0u; r < 24u; ++r) {
            ROUND(KECCAK_RC[r]);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;

        if (take > 0)  out_data[out_base + written + 0]  = A0;
        if (take > 1)  out_data[out_base + written + 1]  = A1;
        if (take > 2)  out_data[out_base + written + 2]  = A2;
        if (take > 3)  out_data[out_base + written + 3]  = A3;
        if (take > 4)  out_data[out_base + written + 4]  = A4;
        if (take > 5)  out_data[out_base + written + 5]  = A5;
        if (take > 6)  out_data[out_base + written + 6]  = A6;
        if (take > 7)  out_data[out_base + written + 7]  = A7;
        if (take > 8)  out_data[out_base + written + 8]  = A8;
        if (take > 9)  out_data[out_base + written + 9]  = A9;
        if (take > 10) out_data[out_base + written + 10] = A10;
        if (take > 11) out_data[out_base + written + 11] = A11;
        if (take > 12) out_data[out_base + written + 12] = A12;
        if (take > 13) out_data[out_base + written + 13] = A13;
        if (take > 14) out_data[out_base + written + 14] = A14;
        if (take > 15) out_data[out_base + written + 15] = A15;
        if (take > 16) out_data[out_base + written + 16] = A16;
        if (take > 17) out_data[out_base + written + 17] = A17;
        if (take > 18) out_data[out_base + written + 18] = A18;
        if (take > 19) out_data[out_base + written + 19] = A19;
        if (take > 20) out_data[out_base + written + 20] = A20;
        if (take > 21) out_data[out_base + written + 21] = A21;
        if (take > 22) out_data[out_base + written + 22] = A22;
        if (take > 23) out_data[out_base + written + 23] = A23;
        if (take > 24) out_data[out_base + written + 24] = A24;

        written += take;
        if (written >= out_lanes) break;
    }
}