#include <metal_stdlib>
using namespace metal;

inline ulong ROTL64(ulong x, uint k) {
    return (x << k) | (x >> (64 - k));
}

#define KECCAK_ROUND(RCV)                                            \
{                                                                    \
    ulong C0 = a0 ^ a5 ^ a10 ^ a15 ^ a20;                            \
    ulong C1 = a1 ^ a6 ^ a11 ^ a16 ^ a21;                            \
    ulong C2 = a2 ^ a7 ^ a12 ^ a17 ^ a22;                            \
    ulong C3 = a3 ^ a8 ^ a13 ^ a18 ^ a23;                            \
    ulong C4 = a4 ^ a9 ^ a14 ^ a19 ^ a24;                            \
    ulong D0 = C4 ^ ROTL64(C1, 1);                                   \
    ulong D1 = C0 ^ ROTL64(C2, 1);                                   \
    ulong D2 = C1 ^ ROTL64(C3, 1);                                   \
    ulong D3 = C2 ^ ROTL64(C4, 1);                                   \
    ulong D4 = C3 ^ ROTL64(C0, 1);                                   \
    ulong b00 =        (a0  ^ D0)      ;                             \
    ulong b02 = ROTL64(a1  ^ D1,  1);                                \
    ulong b04 = ROTL64(a2  ^ D2, 62);                                \
    ulong b01 = ROTL64(a3  ^ D3, 28);                                \
    ulong b03 = ROTL64(a4  ^ D4, 27);                                \
    ulong b13 = ROTL64(a5  ^ D0, 36);                                \
    ulong b10 = ROTL64(a6  ^ D1, 44);                                \
    ulong b12 = ROTL64(a7  ^ D2,  6);                                \
    ulong b14 = ROTL64(a8  ^ D3, 55);                                \
    ulong b11 = ROTL64(a9  ^ D4, 20);                                \
    ulong b21 = ROTL64(a10 ^ D0,  3);                                \
    ulong b23 = ROTL64(a11 ^ D1, 10);                                \
    ulong b20 = ROTL64(a12 ^ D2, 43);                                \
    ulong b22 = ROTL64(a13 ^ D3, 25);                                \
    ulong b24 = ROTL64(a14 ^ D4, 39);                                \
    ulong b34 = ROTL64(a15 ^ D0, 41);                                \
    ulong b31 = ROTL64(a16 ^ D1, 45);                                \
    ulong b33 = ROTL64(a17 ^ D2, 15);                                \
    ulong b30 = ROTL64(a18 ^ D3, 21);                                \
    ulong b32 = ROTL64(a19 ^ D4,  8);                                \
    ulong b42 = ROTL64(a20 ^ D0, 18);                                \
    ulong b44 = ROTL64(a21 ^ D1,  2);                                \
    ulong b41 = ROTL64(a22 ^ D2, 61);                                \
    ulong b43 = ROTL64(a23 ^ D3, 56);                                \
    ulong b40 = ROTL64(a24 ^ D4, 14);                                \
    a0  = b00 ^ ((~b10) & b20) ^ (RCV);                              \
    a1  = b10 ^ ((~b20) & b30);                                      \
    a2  = b20 ^ ((~b30) & b40);                                      \
    a3  = b30 ^ ((~b40) & b00);                                      \
    a4  = b40 ^ ((~b00) & b10);                                      \
    a5  = b01 ^ ((~b11) & b21);                                      \
    a6  = b11 ^ ((~b21) & b31);                                      \
    a7  = b21 ^ ((~b31) & b41);                                      \
    a8  = b31 ^ ((~b41) & b01);                                      \
    a9  = b41 ^ ((~b01) & b11);                                      \
    a10 = b02 ^ ((~b12) & b22);                                      \
    a11 = b12 ^ ((~b22) & b32);                                      \
    a12 = b22 ^ ((~b32) & b42);                                      \
    a13 = b32 ^ ((~b42) & b02);                                      \
    a14 = b42 ^ ((~b02) & b12);                                      \
    a15 = b03 ^ ((~b13) & b23);                                      \
    a16 = b13 ^ ((~b23) & b33);                                      \
    a17 = b23 ^ ((~b33) & b43);                                      \
    a18 = b33 ^ ((~b43) & b03);                                      \
    a19 = b43 ^ ((~b03) & b13);                                      \
    a20 = b04 ^ ((~b14) & b24);                                      \
    a21 = b14 ^ ((~b24) & b34);                                      \
    a22 = b24 ^ ((~b34) & b44);                                      \
    a23 = b34 ^ ((~b44) & b04);                                      \
    a24 = b44 ^ ((~b04) & b14);                                      \
}

#define KECCAK_PERMUTE()                                             \
    KECCAK_ROUND(0x0000000000000001ul);                              \
    KECCAK_ROUND(0x0000000000008082ul);                              \
    KECCAK_ROUND(0x800000000000808Aul);                              \
    KECCAK_ROUND(0x8000000080008000ul);                              \
    KECCAK_ROUND(0x000000000000808Bul);                              \
    KECCAK_ROUND(0x0000000080000001ul);                              \
    KECCAK_ROUND(0x8000000080008081ul);                              \
    KECCAK_ROUND(0x8000000000008009ul);                              \
    KECCAK_ROUND(0x000000000000008Aul);                              \
    KECCAK_ROUND(0x0000000000000088ul);                              \
    KECCAK_ROUND(0x0000000080008009ul);                              \
    KECCAK_ROUND(0x000000008000000Aul);                              \
    KECCAK_ROUND(0x000000008000808Bul);                              \
    KECCAK_ROUND(0x800000000000008Bul);                              \
    KECCAK_ROUND(0x8000000000008089ul);                              \
    KECCAK_ROUND(0x8000000000008003ul);                              \
    KECCAK_ROUND(0x8000000000008002ul);                              \
    KECCAK_ROUND(0x8000000000000080ul);                              \
    KECCAK_ROUND(0x000000000000800Aul);                              \
    KECCAK_ROUND(0x800000008000000Aul);                              \
    KECCAK_ROUND(0x8000000080008081ul);                              \
    KECCAK_ROUND(0x8000000000008080ul);                              \
    KECCAK_ROUND(0x0000000080000001ul);                              \
    KECCAK_ROUND(0x8000000080008008ul);

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

    // Initialise full state to zero in scalar registers.
    ulong a0=0, a1=0, a2=0, a3=0, a4=0;
    ulong a5=0, a6=0, a7=0, a8=0, a9=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // Absorb msg_lanes lanes from in_data into a0..a(msg_lanes-1).
    // All test cases have msg_bytes = 32 => msg_lanes = 4.
    uint in_base = idx * msg_lanes;
    device const ulong *src = in_data + in_base;

    if (msg_lanes == 4u) {
        a0 = src[0];
        a1 = src[1];
        a2 = src[2];
        a3 = src[3];
        // Domain byte goes into lane 4, byte 0.
        a4 = (ulong)(domain & 0xFFu);
    } else {
        // General path.
        ulong tmp[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) tmp[i] = 0ul;
        for (uint i = 0; i < msg_lanes; ++i) tmp[i] = src[i];
        tmp[msg_lanes] ^= (ulong)(domain & 0xFFu);
        a0=tmp[0]; a1=tmp[1]; a2=tmp[2]; a3=tmp[3]; a4=tmp[4];
        a5=tmp[5]; a6=tmp[6]; a7=tmp[7]; a8=tmp[8]; a9=tmp[9];
        a10=tmp[10]; a11=tmp[11]; a12=tmp[12]; a13=tmp[13]; a14=tmp[14];
        a15=tmp[15]; a16=tmp[16]; a17=tmp[17]; a18=tmp[18]; a19=tmp[19];
        a20=tmp[20]; a21=tmp[21]; a22=tmp[22]; a23=tmp[23]; a24=tmp[24];
    }

    // XOR 0x80 into byte 7 of lane (rate_lanes - 1).
    ulong pad_hi = 0x8000000000000000ul;
    uint pad_lane = rate_lanes - 1u;
    switch (pad_lane) {
        case 16u: a16 ^= pad_hi; break; // SHA3-256: rate=136, lane 16
        case 20u: a20 ^= pad_hi; break; // SHAKE128: rate=168, lane 20
        case  8u: a8  ^= pad_hi; break;
        case  9u: a9  ^= pad_hi; break;
        case 10u: a10 ^= pad_hi; break;
        case 11u: a11 ^= pad_hi; break;
        case 12u: a12 ^= pad_hi; break;
        case 13u: a13 ^= pad_hi; break;
        case 14u: a14 ^= pad_hi; break;
        case 15u: a15 ^= pad_hi; break;
        case 17u: a17 ^= pad_hi; break;
        case 18u: a18 ^= pad_hi; break;
        case 19u: a19 ^= pad_hi; break;
        case 21u: a21 ^= pad_hi; break;
        default:  break;
    }

    // First permutation.
    KECCAK_PERMUTE();

    // Squeeze. Common case: out_lanes <= rate_lanes (SHA3-256, out=4 lanes).
    uint out_base = idx * out_lanes;
    device ulong *dst = out_data + out_base;

    if (out_lanes <= rate_lanes) {
        // Fast path: single squeeze block, no further permutation needed.
        // SHA3-256: out_lanes = 4.
        if (out_lanes == 4u) {
            dst[0] = a0;
            dst[1] = a1;
            dst[2] = a2;
            dst[3] = a3;
        } else {
            // Generic short squeeze.
            ulong st[25] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,
                            a15,a16,a17,a18,a19,a20,a21,a22,a23,a24};
            for (uint j = 0; j < out_lanes; ++j) dst[j] = st[j];
        }
        return;
    }

    // Multi-squeeze path (SHAKE128 etc.).
    uint written = 0u;
    for (;;) {
        ulong st[25] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,
                        a15,a16,a17,a18,a19,a20,a21,a22,a23,a24};
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) dst[written + j] = st[j];
        written += take;
        if (written >= out_lanes) break;
        KECCAK_PERMUTE();
    }
}