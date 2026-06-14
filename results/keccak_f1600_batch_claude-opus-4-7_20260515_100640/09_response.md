**Optimization:** The incumbent (iter 5, 0.9638) is the best so far. My recent attempts to deviate haven't improved things. Looking at the bug in attempt 8 (likely a pi-permutation indexing error), I'll go back to the proven incumbent structure but try one targeted improvement: hoist the loads/stores by working entirely in registers without the round-trip through `thread ulong *a`. Also, since `msg_bytes=32` always (4 lanes input) and SHA3-256 is the dominant case (out_lanes=4 ≤ rate_lanes), I'll specialize that path to skip the squeeze loop entirely. The permutation runs inline on local vars (no array indirection), and the SHAKE path is kept correct but separate.

```metal
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

#define KECCAK_24_ROUNDS()                                           \
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

    // Registers for state
    ulong a0=0,a1=0,a2=0,a3=0,a4=0;
    ulong a5=0,a6=0,a7=0,a8=0,a9=0;
    ulong a10=0,a11=0,a12=0,a13=0,a14=0;
    ulong a15=0,a16=0,a17=0,a18=0,a19=0;
    ulong a20=0,a21=0,a22=0,a23=0,a24=0;

    uint in_base = idx * msg_lanes;
    // Absorb message lanes (msg_lanes < rate_lanes, typically 4 for SHA3-256)
    if (msg_lanes > 0) a0 = in_data[in_base + 0];
    if (msg_lanes > 1) a1 = in_data[in_base + 1];
    if (msg_lanes > 2) a2 = in_data[in_base + 2];
    if (msg_lanes > 3) a3 = in_data[in_base + 3];
    for (uint i = 4; i < msg_lanes; ++i) {
        ulong v = in_data[in_base + i];
        switch (i) {
            case 4:  a4  ^= v; break;
            case 5:  a5  ^= v; break;
            case 6:  a6  ^= v; break;
            case 7:  a7  ^= v; break;
            case 8:  a8  ^= v; break;
            case 9:  a9  ^= v; break;
            case 10: a10 ^= v; break;
            case 11: a11 ^= v; break;
            case 12: a12 ^= v; break;
            case 13: a13 ^= v; break;
            case 14: a14 ^= v; break;
            case 15: a15 ^= v; break;
            case 16: a16 ^= v; break;
            case 17: a17 ^= v; break;
            case 18: a18 ^= v; break;
            case 19: a19 ^= v; break;
            case 20: a20 ^= v; break;
        }
    }

    // Domain separator at byte position msg_bytes (lane msg_lanes, byte 0)
    ulong dom = (ulong)(domain & 0xFFu);
    switch (msg_lanes) {
        case 0:  a0  ^= dom; break;
        case 1:  a1  ^= dom; break;
        case 2:  a2  ^= dom; break;
        case 3:  a3  ^= dom; break;
        case 4:  a4  ^= dom; break;
        case 5:  a5  ^= dom; break;
        case 6:  a6  ^= dom; break;
        case 7:  a7  ^= dom; break;
        case 8:  a8  ^= dom; break;
        case 9:  a9  ^= dom; break;
        case 10: a10 ^= dom; break;
        case 11: a11 ^= dom; break;
        case 12: a12 ^= dom; break;
        case 13: a13 ^= dom; break;
        case 14: a14 ^= dom; break;
        case 15: a15 ^= dom; break;
        case 16: a16 ^= dom; break;
        case 17: a17 ^= dom; break;
        case 18: a18 ^= dom; break;
        case 19: a19 ^= dom; break;
        case 20: a20 ^= dom; break;
    }

    // 0x80 at byte position rate_bytes - 1 (lane rate_lanes-1, byte 7)
    ulong pad = 0x8000000000000000ul;
    uint last = rate_lanes - 1u;
    switch (last) {
        case 0:  a0  ^= pad; break;
        case 1:  a1  ^= pad; break;
        case 2:  a2  ^= pad; break;
        case 3:  a3  ^= pad; break;
        case 4:  a4  ^= pad; break;
        case 5:  a5  ^= pad; break;
        case 6:  a6  ^= pad; break;
        case 7:  a7  ^= pad; break;
        case 8:  a8  ^= pad; break;
        case 9:  a9  ^= pad; break;
        case 10: a10 ^= pad; break;
        case 11: a11 ^= pad; break;
        case 12: a12 ^= pad; break;
        case 13: a13 ^= pad; break;
        case 14: a14 ^= pad; break;
        case 15: a15 ^= pad; break;
        case 16: a16 ^= pad; break;
        case 17: a17 ^= pad; break;
        case 18: a18 ^= pad; break;
        case 19: a19 ^= pad; break;
        case 20: a20 ^= pad; break;
    }

    uint out_base = idx * out_lanes;

    KECCAK_24_ROUNDS();

    // Fast path: out_lanes <= rate_lanes (SHA3-256: out=4, rate=17)
    if (out_lanes <= rate_lanes) {
        if (out_lanes > 0) out_data[out_base + 0] = a0;
        if (out_lanes > 1) out_data[out_base + 1] = a1;
        if (out_lanes > 2) out_data[out_base + 2] = a2;
        if (out_lanes > 3) out_data[out_base + 3] = a3;
        for (uint j = 4; j < out_lanes; ++j) {
            ulong v;
            switch (j) {
                case 4:  v = a4;  break;
                case 5:  v = a5;  break;
                case 6:  v = a6;  break;
                case 7:  v = a7;  break;
                case 8:  v = a8;  break;
                case 9:  v = a9;  break;
                case 10: v = a10; break;
                case 11: v = a11; break;
                case 12: v = a12; break;
                case 13: v = a13; break;
                case 14: v = a14; break;
                case 15: v = a15; break;
                case 16: v = a16; break;
                case 17: v = a17; break;
                case 18: v = a18; break;
                case 19: v = a19; break;
                case 20: v = a20; break;
                default: v = 0; break;
            }
            out_data[out_base + j] = v;
        }
        return;
    }

    // Slow path: multiple squeezes (SHAKE128). Use array form.
    ulong a[25];
    a[0]=a0;a[1]=a1;a[2]=a2;a[3]=a3;a[4]=a4;
    a[5]=a5;a[6]=a6;a[7]=a7;a[8]=a8;a[9]=a9;
    a[10]=a10;a[11]=a11;a[12]=a12;a[13]=a13;a[14]=a14;
    a[15]=a15;a[16]=a16;a[17]=a17;a[18]=a18;a[19]=a19;
    a[20]=a20;a[21]=a21;a[22]=a22;a[23]=a23;a[24]=a24;

    uint written = 0u;
    for (;;) {
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = a[j];
        }
        written += take;
        if (written >= out_lanes) break;

        // Run permutation on array
        a0=a[0];a1=a[1];a2=a[2];a3=a[3];a4=a[4];
        a5=a[5];a6=a[6];a7=a[7];a8=a[8];a9=a[9];
        a10=a[10];a11=a[11];a12=a[12];a13=a[13];a14=a[14];
        a15=a[15];a16=a[16];a17=a[17];a18=a[18];a19=a[19];
        a20=a[20];a21=a[21];a22=a[22];a23=a[23];a24=a[24];
        KECCAK_24_ROUNDS();
        a[0]=a0;a[1]=a1;a[2]=a2;a[3]=a3;a[4]=a4;
        a[5]=a5;a[6]=a6;a[7]=a7;a[8]=a8;a[9]=a9;
        a[10]=a10;a[11]=a11;a[12]=a12;a[13]=a13;a[14]=a14;
        a[15]=a15;a[16]=a16;a[17]=a17;a[18]=a18;a[19]=a19;
        a[20]=a20;a[21]=a21;a[22]=a22;a[23]=a23;a[24]=a24;
    }
}
```