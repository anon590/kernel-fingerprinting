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

// Lane indexing: a[x][y] -> state lane x + 5*y.
// rho[x][y] table (FIPS 202):
//   x=0: y=0..4: 0,36, 3,41,18
//   x=1:         1,44,10,45, 2
//   x=2:        62, 6,43,15,61
//   x=3:        28,55,25,21,56
//   x=4:        27,20,39, 8,14
//
// pi: source (x,y) -> destination (y, (2x+3y)%5).
// We enumerate every (x,y) source and emit B[dst_x][dst_y] = ROTL(a[x][y], rho[x][y]).
// Source list with (dst_x, dst_y) = (y, (2x+3y)%5):
//   (0,0)->(0,0)  rho 0
//   (1,0)->(0,2)  rho 1
//   (2,0)->(0,4)  rho 62
//   (3,0)->(0,1)  rho 28
//   (4,0)->(0,3)  rho 27
//   (0,1)->(1,3)  rho 36
//   (1,1)->(1,0)  rho 44
//   (2,1)->(1,2)  rho 6
//   (3,1)->(1,4)  rho 55
//   (4,1)->(1,1)  rho 20
//   (0,2)->(2,1)  rho 3
//   (1,2)->(2,3)  rho 10
//   (2,2)->(2,0)  rho 43
//   (3,2)->(2,2)  rho 25
//   (4,2)->(2,4)  rho 39
//   (0,3)->(3,4)  rho 41
//   (1,3)->(3,1)  rho 45
//   (2,3)->(3,3)  rho 15
//   (3,3)->(3,0)  rho 21
//   (4,3)->(3,2)  rho 8
//   (0,4)->(4,2)  rho 18
//   (1,4)->(4,4)  rho 2
//   (2,4)->(4,1)  rho 61
//   (3,4)->(4,3)  rho 56
//   (4,4)->(4,0)  rho 14

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
    /* rho + pi: B[dst_x][dst_y] = ROTL(a[src_x][src_y], rho) */      \
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
    /* chi + iota: a[x][y] = B[x][y] ^ ((~B[(x+1)%5][y]) & B[(x+2)%5][y]) */ \
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

    // 25-lane state in named registers (lane x + 5*y).
    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    // Stage absorb through small array, then move into registers.
    ulong S[25];
    for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;
    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) S[i] = in_data[in_base + i];
    S[msg_lanes]       ^= (ulong)(domain & 0xFFu);
    S[rate_lanes - 1u] ^= 0x8000000000000000ul;

    a00 = S[ 0]; a10 = S[ 1]; a20 = S[ 2]; a30 = S[ 3]; a40 = S[ 4];
    a01 = S[ 5]; a11 = S[ 6]; a21 = S[ 7]; a31 = S[ 8]; a41 = S[ 9];
    a02 = S[10]; a12 = S[11]; a22 = S[12]; a32 = S[13]; a42 = S[14];
    a03 = S[15]; a13 = S[16]; a23 = S[17]; a33 = S[18]; a43 = S[19];
    a04 = S[20]; a14 = S[21]; a24 = S[22]; a34 = S[23]; a44 = S[24];

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        KECCAK_ROUND(KECCAK_RC[ 0])
        KECCAK_ROUND(KECCAK_RC[ 1])
        KECCAK_ROUND(KECCAK_RC[ 2])
        KECCAK_ROUND(KECCAK_RC[ 3])
        KECCAK_ROUND(KECCAK_RC[ 4])
        KECCAK_ROUND(KECCAK_RC[ 5])
        KECCAK_ROUND(KECCAK_RC[ 6])
        KECCAK_ROUND(KECCAK_RC[ 7])
        KECCAK_ROUND(KECCAK_RC[ 8])
        KECCAK_ROUND(KECCAK_RC[ 9])
        KECCAK_ROUND(KECCAK_RC[10])
        KECCAK_ROUND(KECCAK_RC[11])
        KECCAK_ROUND(KECCAK_RC[12])
        KECCAK_ROUND(KECCAK_RC[13])
        KECCAK_ROUND(KECCAK_RC[14])
        KECCAK_ROUND(KECCAK_RC[15])
        KECCAK_ROUND(KECCAK_RC[16])
        KECCAK_ROUND(KECCAK_RC[17])
        KECCAK_ROUND(KECCAK_RC[18])
        KECCAK_ROUND(KECCAK_RC[19])
        KECCAK_ROUND(KECCAK_RC[20])
        KECCAK_ROUND(KECCAK_RC[21])
        KECCAK_ROUND(KECCAK_RC[22])
        KECCAK_ROUND(KECCAK_RC[23])

        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
        OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = OUT[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}