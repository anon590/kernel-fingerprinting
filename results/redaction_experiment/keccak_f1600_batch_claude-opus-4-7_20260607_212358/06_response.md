**Optimization**: The previous version uses a generic loop with runtime-checked output writes, even for the dominant SHA3-256 case where `out_lanes=4 < rate_lanes`. I'll specialize the common path: detect when `out_lanes <= rate_lanes` (single permutation, no loop, no staging array) and directly write the 4-8 output lanes from the named registers without going through an intermediate array. I also eliminate the absorb staging array entirely — XOR input lanes directly into the named registers using a small switch on `msg_lanes` (host says `msg_bytes=32` always, so 4 lanes), and place the padding bytes directly. This removes two 25-lane local arrays and the variable-bound store loop, reducing register pressure and address arithmetic. The SHAKE128 multi-squeeze path remains correct via a fallback loop.

```metal
#include <metal_stdlib>
using namespace metal;

constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

#define ROTL(x,k) (((x) << (k)) | ((x) >> (64 - (k))))

#define KECCAK_ROUND(RC) {                                            \
    ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                           \
    ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                           \
    ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                           \
    ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                           \
    ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                           \
    ulong D0 = C4 ^ ROTL(C1, 1);                                      \
    ulong D1 = C0 ^ ROTL(C2, 1);                                      \
    ulong D2 = C1 ^ ROTL(C3, 1);                                      \
    ulong D3 = C2 ^ ROTL(C4, 1);                                      \
    ulong D4 = C3 ^ ROTL(C0, 1);                                      \
    a00 ^= D0; a10 ^= D1; a20 ^= D2; a30 ^= D3; a40 ^= D4;            \
    a01 ^= D0; a11 ^= D1; a21 ^= D2; a31 ^= D3; a41 ^= D4;            \
    a02 ^= D0; a12 ^= D1; a22 ^= D2; a32 ^= D3; a42 ^= D4;            \
    a03 ^= D0; a13 ^= D1; a23 ^= D2; a33 ^= D3; a43 ^= D4;            \
    a04 ^= D0; a14 ^= D1; a24 ^= D2; a34 ^= D3; a44 ^= D4;            \
    ulong B00 = a00;                                                  \
    ulong B02 = ROTL(a10, 1);                                         \
    ulong B04 = ROTL(a20, 62);                                        \
    ulong B01 = ROTL(a30, 28);                                        \
    ulong B03 = ROTL(a40, 27);                                        \
    ulong B13 = ROTL(a01, 36);                                        \
    ulong B10 = ROTL(a11, 44);                                        \
    ulong B12 = ROTL(a21, 6);                                         \
    ulong B14 = ROTL(a31, 55);                                        \
    ulong B11 = ROTL(a41, 20);                                        \
    ulong B21 = ROTL(a02, 3);                                         \
    ulong B23 = ROTL(a12, 10);                                        \
    ulong B20 = ROTL(a22, 43);                                        \
    ulong B22 = ROTL(a32, 25);                                        \
    ulong B24 = ROTL(a42, 39);                                        \
    ulong B34 = ROTL(a03, 41);                                        \
    ulong B31 = ROTL(a13, 45);                                        \
    ulong B33 = ROTL(a23, 15);                                        \
    ulong B30 = ROTL(a33, 21);                                        \
    ulong B32 = ROTL(a43, 8);                                         \
    ulong B42 = ROTL(a04, 18);                                        \
    ulong B44 = ROTL(a14, 2);                                         \
    ulong B41 = ROTL(a24, 61);                                        \
    ulong B43 = ROTL(a34, 56);                                        \
    ulong B40 = ROTL(a44, 14);                                        \
    a00 = B00 ^ ((~B10) & B20) ^ (RC);                                \
    a10 = B10 ^ ((~B20) & B30);                                       \
    a20 = B20 ^ ((~B30) & B40);                                       \
    a30 = B30 ^ ((~B40) & B00);                                       \
    a40 = B40 ^ ((~B00) & B10);                                       \
    a01 = B01 ^ ((~B11) & B21);                                       \
    a11 = B11 ^ ((~B21) & B31);                                       \
    a21 = B21 ^ ((~B31) & B41);                                       \
    a31 = B31 ^ ((~B41) & B01);                                       \
    a41 = B41 ^ ((~B01) & B11);                                       \
    a02 = B02 ^ ((~B12) & B22);                                       \
    a12 = B12 ^ ((~B22) & B32);                                       \
    a22 = B22 ^ ((~B32) & B42);                                       \
    a32 = B32 ^ ((~B42) & B02);                                       \
    a42 = B42 ^ ((~B02) & B12);                                       \
    a03 = B03 ^ ((~B13) & B23);                                       \
    a13 = B13 ^ ((~B23) & B33);                                       \
    a23 = B23 ^ ((~B33) & B43);                                       \
    a33 = B33 ^ ((~B43) & B03);                                       \
    a43 = B43 ^ ((~B03) & B13);                                       \
    a04 = B04 ^ ((~B14) & B24);                                       \
    a14 = B14 ^ ((~B24) & B34);                                       \
    a24 = B24 ^ ((~B34) & B44);                                       \
    a34 = B34 ^ ((~B44) & B04);                                       \
    a44 = B44 ^ ((~B04) & B14);                                       \
}

#define KECCAK_PERMUTE() {                          \
    KECCAK_ROUND(KECCAK_RC[ 0])                     \
    KECCAK_ROUND(KECCAK_RC[ 1])                     \
    KECCAK_ROUND(KECCAK_RC[ 2])                     \
    KECCAK_ROUND(KECCAK_RC[ 3])                     \
    KECCAK_ROUND(KECCAK_RC[ 4])                     \
    KECCAK_ROUND(KECCAK_RC[ 5])                     \
    KECCAK_ROUND(KECCAK_RC[ 6])                     \
    KECCAK_ROUND(KECCAK_RC[ 7])                     \
    KECCAK_ROUND(KECCAK_RC[ 8])                     \
    KECCAK_ROUND(KECCAK_RC[ 9])                     \
    KECCAK_ROUND(KECCAK_RC[10])                     \
    KECCAK_ROUND(KECCAK_RC[11])                     \
    KECCAK_ROUND(KECCAK_RC[12])                     \
    KECCAK_ROUND(KECCAK_RC[13])                     \
    KECCAK_ROUND(KECCAK_RC[14])                     \
    KECCAK_ROUND(KECCAK_RC[15])                     \
    KECCAK_ROUND(KECCAK_RC[16])                     \
    KECCAK_ROUND(KECCAK_RC[17])                     \
    KECCAK_ROUND(KECCAK_RC[18])                     \
    KECCAK_ROUND(KECCAK_RC[19])                     \
    KECCAK_ROUND(KECCAK_RC[20])                     \
    KECCAK_ROUND(KECCAK_RC[21])                     \
    KECCAK_ROUND(KECCAK_RC[22])                     \
    KECCAK_ROUND(KECCAK_RC[23])                     \
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

    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    uint in_base = idx * msg_lanes;
    device const ulong *in_ptr = in_data + in_base;

    // Absorb input lanes. msg_bytes < rate_bytes, so msg_lanes < rate_lanes.
    // All test sizes have msg_bytes = 32 (msg_lanes = 4), but support general case.
    // We unroll based on lane index since msg_lanes is small (<= ~21 for SHAKE128).
    {
        ulong dom = (ulong)(domain & 0xFFu);
        ulong pad_hi = 0x8000000000000000ul;

        // Read input lanes one at a time into the appropriate register.
        // Use a switch-like cascade; the compiler can fold msg_lanes=4 specially.
        for (uint i = 0u; i < msg_lanes; ++i) {
            ulong v = in_ptr[i];
            switch (i) {
                case  0: a00 ^= v; break;
                case  1: a10 ^= v; break;
                case  2: a20 ^= v; break;
                case  3: a30 ^= v; break;
                case  4: a40 ^= v; break;
                case  5: a01 ^= v; break;
                case  6: a11 ^= v; break;
                case  7: a21 ^= v; break;
                case  8: a31 ^= v; break;
                case  9: a41 ^= v; break;
                case 10: a02 ^= v; break;
                case 11: a12 ^= v; break;
                case 12: a22 ^= v; break;
                case 13: a32 ^= v; break;
                case 14: a42 ^= v; break;
                case 15: a03 ^= v; break;
                case 16: a13 ^= v; break;
                case 17: a23 ^= v; break;
                case 18: a33 ^= v; break;
                case 19: a43 ^= v; break;
                case 20: a04 ^= v; break;
                case 21: a14 ^= v; break;
                case 22: a24 ^= v; break;
                case 23: a34 ^= v; break;
                default: break;
            }
        }

        // Domain byte at lane msg_lanes, byte 0.
        switch (msg_lanes) {
            case  0: a00 ^= dom; break;
            case  1: a10 ^= dom; break;
            case  2: a20 ^= dom; break;
            case  3: a30 ^= dom; break;
            case  4: a40 ^= dom; break;
            case  5: a01 ^= dom; break;
            case  6: a11 ^= dom; break;
            case  7: a21 ^= dom; break;
            case  8: a31 ^= dom; break;
            case  9: a41 ^= dom; break;
            case 10: a02 ^= dom; break;
            case 11: a12 ^= dom; break;
            case 12: a22 ^= dom; break;
            case 13: a32 ^= dom; break;
            case 14: a42 ^= dom; break;
            case 15: a03 ^= dom; break;
            case 16: a13 ^= dom; break;
            case 17: a23 ^= dom; break;
            case 18: a33 ^= dom; break;
            case 19: a43 ^= dom; break;
            case 20: a04 ^= dom; break;
            case 21: a14 ^= dom; break;
            case 22: a24 ^= dom; break;
            case 23: a34 ^= dom; break;
            default: break;
        }

        // 0x80 at byte rate_bytes - 1 = lane rate_lanes - 1, byte 7.
        uint plast = rate_lanes - 1u;
        switch (plast) {
            case  0: a00 ^= pad_hi; break;
            case  1: a10 ^= pad_hi; break;
            case  2: a20 ^= pad_hi; break;
            case  3: a30 ^= pad_hi; break;
            case  4: a40 ^= pad_hi; break;
            case  5: a01 ^= pad_hi; break;
            case  6: a11 ^= pad_hi; break;
            case  7: a21 ^= pad_hi; break;
            case  8: a31 ^= pad_hi; break;
            case  9: a41 ^= pad_hi; break;
            case 10: a02 ^= pad_hi; break;
            case 11: a12 ^= pad_hi; break;
            case 12: a22 ^= pad_hi; break;
            case 13: a32 ^= pad_hi; break;
            case 14: a42 ^= pad_hi; break;
            case 15: a03 ^= pad_hi; break;
            case 16: a13 ^= pad_hi; break;
            case 17: a23 ^= pad_hi; break;
            case 18: a33 ^= pad_hi; break;
            case 19: a43 ^= pad_hi; break;
            case 20: a04 ^= pad_hi; break;
            case 21: a14 ^= pad_hi; break;
            case 22: a24 ^= pad_hi; break;
            case 23: a34 ^= pad_hi; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    device ulong *op = out_data + out_base;

    // Fast path: single permutation, output fits in first squeeze.
    if (out_lanes <= rate_lanes) {
        KECCAK_PERMUTE();

        // Direct write of out_lanes lanes (out_lanes is a small constant per dispatch).
        // Common: out_lanes = 4 (SHA3-256).
        if (out_lanes >= 1u) op[0] = a00;
        if (out_lanes >= 2u) op[1] = a10;
        if (out_lanes >= 3u) op[2] = a20;
        if (out_lanes >= 4u) op[3] = a30;
        if (out_lanes >= 5u) op[4] = a40;
        if (out_lanes >= 6u) op[5] = a01;
        if (out_lanes >= 7u) op[6] = a11;
        if (out_lanes >= 8u) op[7] = a21;
        if (out_lanes > 8u) {
            // Fallback for larger out_lanes (still <= rate_lanes).
            ulong OUT[25];
            OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
            OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
            OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
            OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
            OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;
            for (uint j = 8u; j < out_lanes; ++j) op[j] = OUT[j];
        }
        return;
    }

    // Multi-squeeze path (SHAKE128 etc.).
    uint written = 0u;
    for (;;) {
        KECCAK_PERMUTE();

        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
        OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            op[written + j] = OUT[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```