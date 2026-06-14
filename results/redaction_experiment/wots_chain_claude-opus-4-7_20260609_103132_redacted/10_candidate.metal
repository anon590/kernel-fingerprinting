#include <metal_stdlib>
using namespace metal;

constant ulong KRC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

#define ROL(x,k) rotate((ulong)(x), (ulong)(k))

// One Keccak-f round operating on 25 named lane registers.
#define KECCAK_ROUND(RC) {                                                 \
    ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;                                \
    ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;                                \
    ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;                                \
    ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;                                \
    ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;                                \
    ulong D0 = C4 ^ ROL(C1,1);                                             \
    ulong D1 = C0 ^ ROL(C2,1);                                             \
    ulong D2 = C1 ^ ROL(C3,1);                                             \
    ulong D3 = C2 ^ ROL(C4,1);                                             \
    ulong D4 = C3 ^ ROL(C0,1);                                             \
    ulong B00 =     (a00 ^ D0);                                            \
    ulong B10 = ROL(a01 ^ D1,  1);                                         \
    ulong B20 = ROL(a02 ^ D2, 62);                                         \
    ulong B05 = ROL(a03 ^ D3, 28);                                         \
    ulong B15 = ROL(a04 ^ D4, 27);                                         \
    ulong B16 = ROL(a05 ^ D0, 36);                                         \
    ulong B01 = ROL(a06 ^ D1, 44);                                         \
    ulong B11 = ROL(a07 ^ D2,  6);                                         \
    ulong B21 = ROL(a08 ^ D3, 55);                                         \
    ulong B06 = ROL(a09 ^ D4, 20);                                         \
    ulong B07 = ROL(a10 ^ D0,  3);                                         \
    ulong B17 = ROL(a11 ^ D1, 10);                                         \
    ulong B02 = ROL(a12 ^ D2, 43);                                         \
    ulong B12 = ROL(a13 ^ D3, 25);                                         \
    ulong B22 = ROL(a14 ^ D4, 39);                                         \
    ulong B23 = ROL(a15 ^ D0, 41);                                         \
    ulong B08 = ROL(a16 ^ D1, 45);                                         \
    ulong B18 = ROL(a17 ^ D2, 15);                                         \
    ulong B03 = ROL(a18 ^ D3, 21);                                         \
    ulong B13 = ROL(a19 ^ D4,  8);                                         \
    ulong B14 = ROL(a20 ^ D0, 18);                                         \
    ulong B24 = ROL(a21 ^ D1,  2);                                         \
    ulong B09 = ROL(a22 ^ D2, 61);                                         \
    ulong B19 = ROL(a23 ^ D3, 56);                                         \
    ulong B04 = ROL(a24 ^ D4, 14);                                         \
    a00 = B00 ^ ((~B01) & B02) ^ (RC);                                     \
    a01 = B01 ^ ((~B02) & B03);                                            \
    a02 = B02 ^ ((~B03) & B04);                                            \
    a03 = B03 ^ ((~B04) & B00);                                            \
    a04 = B04 ^ ((~B00) & B01);                                            \
    a05 = B05 ^ ((~B06) & B07);                                            \
    a06 = B06 ^ ((~B07) & B08);                                            \
    a07 = B07 ^ ((~B08) & B09);                                            \
    a08 = B08 ^ ((~B09) & B05);                                            \
    a09 = B09 ^ ((~B05) & B06);                                            \
    a10 = B10 ^ ((~B11) & B12);                                            \
    a11 = B11 ^ ((~B12) & B13);                                            \
    a12 = B12 ^ ((~B13) & B14);                                            \
    a13 = B13 ^ ((~B14) & B10);                                            \
    a14 = B14 ^ ((~B10) & B11);                                            \
    a15 = B15 ^ ((~B16) & B17);                                            \
    a16 = B16 ^ ((~B17) & B18);                                            \
    a17 = B17 ^ ((~B18) & B19);                                            \
    a18 = B18 ^ ((~B19) & B15);                                            \
    a19 = B19 ^ ((~B15) & B16);                                            \
    a20 = B20 ^ ((~B21) & B22);                                            \
    a21 = B21 ^ ((~B22) & B23);                                            \
    a22 = B22 ^ ((~B23) & B24);                                            \
    a23 = B23 ^ ((~B24) & B20);                                            \
    a24 = B24 ^ ((~B20) & B21);                                            \
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

    const uint n_lanes = n_bytes >> 3;
    const uint base = idx * n_lanes;
    const uint W = w;

    // Precompute per-lane padding masks (constant over the chain).
    // pad[k] gives bits to XOR into lane k of a zero state to encode
    // SHA3-256 padding when message occupies lanes 0..n_lanes-1.
    ulong pad00=0, pad01=0, pad02=0, pad03=0, pad04=0;
    ulong pad05=0, pad06=0, pad07=0, pad08=0, pad09=0;
    ulong pad10=0, pad11=0, pad12=0, pad13=0, pad14=0;
    ulong pad15=0, pad16=0x8000000000000000ul;
    switch (n_lanes) {
        case 0:  pad00 = 0x06ul; break;
        case 1:  pad01 = 0x06ul; break;
        case 2:  pad02 = 0x06ul; break;
        case 3:  pad03 = 0x06ul; break;
        case 4:  pad04 = 0x06ul; break;
        case 5:  pad05 = 0x06ul; break;
        case 6:  pad06 = 0x06ul; break;
        case 7:  pad07 = 0x06ul; break;
        case 8:  pad08 = 0x06ul; break;
        case 9:  pad09 = 0x06ul; break;
        case 10: pad10 = 0x06ul; break;
        case 11: pad11 = 0x06ul; break;
        case 12: pad12 = 0x06ul; break;
        case 13: pad13 = 0x06ul; break;
        case 14: pad14 = 0x06ul; break;
        case 15: pad15 = 0x06ul; break;
        case 16: pad16 ^= 0x06ul; break;
        default: break;
    }

    // Load message lanes into m00..m15.
    ulong m00=0,m01=0,m02=0,m03=0,m04=0,m05=0,m06=0,m07=0;
    ulong m08=0,m09=0,m10=0,m11=0,m12=0,m13=0,m14=0,m15=0;
    if (n_lanes > 0u)  m00 = seeds[base + 0];
    if (n_lanes > 1u)  m01 = seeds[base + 1];
    if (n_lanes > 2u)  m02 = seeds[base + 2];
    if (n_lanes > 3u)  m03 = seeds[base + 3];
    if (n_lanes > 4u)  m04 = seeds[base + 4];
    if (n_lanes > 5u)  m05 = seeds[base + 5];
    if (n_lanes > 6u)  m06 = seeds[base + 6];
    if (n_lanes > 7u)  m07 = seeds[base + 7];
    if (n_lanes > 8u)  m08 = seeds[base + 8];
    if (n_lanes > 9u)  m09 = seeds[base + 9];
    if (n_lanes > 10u) m10 = seeds[base + 10];
    if (n_lanes > 11u) m11 = seeds[base + 11];
    if (n_lanes > 12u) m12 = seeds[base + 12];
    if (n_lanes > 13u) m13 = seeds[base + 13];
    if (n_lanes > 14u) m14 = seeds[base + 14];
    if (n_lanes > 15u) m15 = seeds[base + 15];

    for (uint step = 0u; step < W; ++step) {
        // Build state with message in lanes 0..n_lanes-1 (others zero)
        // plus padding XOR.
        ulong a00 = m00 ^ pad00;
        ulong a01 = m01 ^ pad01;
        ulong a02 = m02 ^ pad02;
        ulong a03 = m03 ^ pad03;
        ulong a04 = m04 ^ pad04;
        ulong a05 = m05 ^ pad05;
        ulong a06 = m06 ^ pad06;
        ulong a07 = m07 ^ pad07;
        ulong a08 = m08 ^ pad08;
        ulong a09 = m09 ^ pad09;
        ulong a10 = m10 ^ pad10;
        ulong a11 = m11 ^ pad11;
        ulong a12 = m12 ^ pad12;
        ulong a13 = m13 ^ pad13;
        ulong a14 = m14 ^ pad14;
        ulong a15 = m15 ^ pad15;
        ulong a16 = pad16;
        ulong a17 = 0, a18 = 0, a19 = 0;
        ulong a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

        // 24 rounds, fully unrolled to give the compiler maximum freedom.
        KECCAK_ROUND(KRC[0]);
        KECCAK_ROUND(KRC[1]);
        KECCAK_ROUND(KRC[2]);
        KECCAK_ROUND(KRC[3]);
        KECCAK_ROUND(KRC[4]);
        KECCAK_ROUND(KRC[5]);
        KECCAK_ROUND(KRC[6]);
        KECCAK_ROUND(KRC[7]);
        KECCAK_ROUND(KRC[8]);
        KECCAK_ROUND(KRC[9]);
        KECCAK_ROUND(KRC[10]);
        KECCAK_ROUND(KRC[11]);
        KECCAK_ROUND(KRC[12]);
        KECCAK_ROUND(KRC[13]);
        KECCAK_ROUND(KRC[14]);
        KECCAK_ROUND(KRC[15]);
        KECCAK_ROUND(KRC[16]);
        KECCAK_ROUND(KRC[17]);
        KECCAK_ROUND(KRC[18]);
        KECCAK_ROUND(KRC[19]);
        KECCAK_ROUND(KRC[20]);
        KECCAK_ROUND(KRC[21]);
        KECCAK_ROUND(KRC[22]);
        KECCAK_ROUND(KRC[23]);

        // Truncate to first n_lanes lanes for next iteration.
        m00 = a00; m01 = a01; m02 = a02; m03 = a03;
        m04 = a04; m05 = a05; m06 = a06; m07 = a07;
        m08 = a08; m09 = a09; m10 = a10; m11 = a11;
        m12 = a12; m13 = a13; m14 = a14; m15 = a15;
    }

    if (n_lanes > 0u)  tips[base + 0]  = m00;
    if (n_lanes > 1u)  tips[base + 1]  = m01;
    if (n_lanes > 2u)  tips[base + 2]  = m02;
    if (n_lanes > 3u)  tips[base + 3]  = m03;
    if (n_lanes > 4u)  tips[base + 4]  = m04;
    if (n_lanes > 5u)  tips[base + 5]  = m05;
    if (n_lanes > 6u)  tips[base + 6]  = m06;
    if (n_lanes > 7u)  tips[base + 7]  = m07;
    if (n_lanes > 8u)  tips[base + 8]  = m08;
    if (n_lanes > 9u)  tips[base + 9]  = m09;
    if (n_lanes > 10u) tips[base + 10] = m10;
    if (n_lanes > 11u) tips[base + 11] = m11;
    if (n_lanes > 12u) tips[base + 12] = m12;
    if (n_lanes > 13u) tips[base + 13] = m13;
    if (n_lanes > 14u) tips[base + 14] = m14;
    if (n_lanes > 15u) tips[base + 15] = m15;
}