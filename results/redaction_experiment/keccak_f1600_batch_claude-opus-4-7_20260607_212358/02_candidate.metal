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

inline ulong ROTL64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

// One Keccak round, fully unrolled with named lanes a[x][y] -> aXY.
// Pi: destination (X,Y) gets source (x=(X+3Y)%5, y=X), with rho offset rho[x][y].
#define KECCAK_ROUND(rc)                                                     \
    {                                                                        \
        ulong c0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                              \
        ulong c1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                              \
        ulong c2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                              \
        ulong c3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                              \
        ulong c4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                              \
        ulong d0 = c4 ^ ROTL64(c1, 1);                                       \
        ulong d1 = c0 ^ ROTL64(c2, 1);                                       \
        ulong d2 = c1 ^ ROTL64(c3, 1);                                       \
        ulong d3 = c2 ^ ROTL64(c4, 1);                                       \
        ulong d4 = c3 ^ ROTL64(c0, 1);                                       \
        /* theta-mixed lanes */                                              \
        ulong t00 = a00 ^ d0; ulong t10 = a10 ^ d1; ulong t20 = a20 ^ d2;    \
        ulong t30 = a30 ^ d3; ulong t40 = a40 ^ d4;                          \
        ulong t01 = a01 ^ d0; ulong t11 = a11 ^ d1; ulong t21 = a21 ^ d2;    \
        ulong t31 = a31 ^ d3; ulong t41 = a41 ^ d4;                          \
        ulong t02 = a02 ^ d0; ulong t12 = a12 ^ d1; ulong t22 = a22 ^ d2;    \
        ulong t32 = a32 ^ d3; ulong t42 = a42 ^ d4;                          \
        ulong t03 = a03 ^ d0; ulong t13 = a13 ^ d1; ulong t23 = a23 ^ d2;    \
        ulong t33 = a33 ^ d3; ulong t43 = a43 ^ d4;                          \
        ulong t04 = a04 ^ d0; ulong t14 = a14 ^ d1; ulong t24 = a24 ^ d2;    \
        ulong t34 = a34 ^ d3; ulong t44 = a44 ^ d4;                          \
        /* rho+pi: B[X][Y] = rotl(t[src_x][src_y], rho[src_x][src_y])        \
           where src_x = (X + 3*Y) % 5, src_y = X.                           \
           rho table indexed [x][y]:                                          \
             rho[0]={ 0,36, 3,41,18}                                          \
             rho[1]={ 1,44,10,45, 2}                                          \
             rho[2]={62, 6,43,15,61}                                          \
             rho[3]={28,55,25,21,56}                                          \
             rho[4]={27,20,39, 8,14}                              */         \
        /* Y=0 */                                                            \
        ulong b00 = t00;                /* src (0,0) rho 0  */               \
        ulong b10 = ROTL64(t31,  1);    /* src (3,1) rho 1? wait */          \
        /* Let me recompute using formula src=(X+3Y)%5,X */                  \
        /* For Y=0: X=0..4, src_x=X, src_y=0. rho[X][0]. */                  \
        /* b00=rot(t00,0); b10=rot(t10,1); b20=rot(t20,62); b30=rot(t30,28); b40=rot(t40,27) */ \
        b10 = ROTL64(t10,  1);                                               \
        ulong b20 = ROTL64(t20, 62);                                         \
        ulong b30 = ROTL64(t30, 28);                                         \
        ulong b40 = ROTL64(t40, 27);                                         \
        /* Y=1: src_x=(X+3)%5, src_y=X. rho[src_x][src_y]=rho[(X+3)%5][X] */ \
        /* X=0: src=(3,0) rho[3][0]=28 */                                    \
        /* X=1: src=(4,1) rho[4][1]=20 */                                    \
        /* X=2: src=(0,2) rho[0][2]=3  */                                    \
        /* X=3: src=(1,3) rho[1][3]=45 */                                    \
        /* X=4: src=(2,4) rho[2][4]=61 */                                    \
        ulong b01 = ROTL64(t30, 28);                                         \
        ulong b11 = ROTL64(t41, 20);                                         \
        ulong b21 = ROTL64(t02,  3);                                         \
        ulong b31 = ROTL64(t13, 45);                                         \
        ulong b41 = ROTL64(t24, 61);                                         \
        /* Y=2: src_x=(X+6)%5=(X+1)%5, src_y=X. rho[(X+1)%5][X] */           \
        /* X=0: src=(1,0) rho[1][0]=1  */                                    \
        /* X=1: src=(2,1) rho[2][1]=6  */                                    \
        /* X=2: src=(3,2) rho[3][2]=25 */                                    \
        /* X=3: src=(4,3) rho[4][3]=8  */                                    \
        /* X=4: src=(0,4) rho[0][4]=18 */                                    \
        ulong b02 = ROTL64(t10,  1);                                         \
        ulong b12 = ROTL64(t21,  6);                                         \
        ulong b22 = ROTL64(t32, 25);                                         \
        ulong b32 = ROTL64(t43,  8);                                         \
        ulong b42 = ROTL64(t04, 18);                                         \
        /* Y=3: src_x=(X+9)%5=(X+4)%5, src_y=X. rho[(X+4)%5][X] */           \
        /* X=0: src=(4,0) rho[4][0]=27 */                                    \
        /* X=1: src=(0,1) rho[0][1]=36 */                                    \
        /* X=2: src=(1,2) rho[1][2]=10 */                                    \
        /* X=3: src=(2,3) rho[2][3]=15 */                                    \
        /* X=4: src=(3,4) rho[3][4]=56 */                                    \
        ulong b03 = ROTL64(t40, 27);                                         \
        ulong b13 = ROTL64(t01, 36);                                         \
        ulong b23 = ROTL64(t12, 10);                                         \
        ulong b33 = ROTL64(t23, 15);                                         \
        ulong b43 = ROTL64(t34, 56);                                         \
        /* Y=4: src_x=(X+12)%5=(X+2)%5, src_y=X. rho[(X+2)%5][X] */          \
        /* X=0: src=(2,0) rho[2][0]=62 */                                    \
        /* X=1: src=(3,1) rho[3][1]=55 */                                    \
        /* X=2: src=(4,2) rho[4][2]=39 */                                    \
        /* X=3: src=(0,3) rho[0][3]=41 */                                    \
        /* X=4: src=(1,4) rho[1][4]=2  */                                    \
        ulong b04 = ROTL64(t20, 62);                                         \
        ulong b14 = ROTL64(t31, 55);                                         \
        ulong b24 = ROTL64(t42, 39);                                         \
        ulong b34 = ROTL64(t03, 41);                                         \
        ulong b44 = ROTL64(t14,  2);                                         \
        /* chi + iota */                                                     \
        a00 = b00 ^ ((~b10) & b20) ^ (rc);                                   \
        a10 = b10 ^ ((~b20) & b30);                                          \
        a20 = b20 ^ ((~b30) & b40);                                          \
        a30 = b30 ^ ((~b40) & b00);                                          \
        a40 = b40 ^ ((~b00) & b10);                                          \
        a01 = b01 ^ ((~b11) & b21);                                          \
        a11 = b11 ^ ((~b21) & b31);                                          \
        a21 = b21 ^ ((~b31) & b41);                                          \
        a31 = b31 ^ ((~b41) & b01);                                          \
        a41 = b41 ^ ((~b01) & b11);                                          \
        a02 = b02 ^ ((~b12) & b22);                                          \
        a12 = b12 ^ ((~b22) & b32);                                          \
        a22 = b22 ^ ((~b32) & b42);                                          \
        a32 = b32 ^ ((~b42) & b02);                                          \
        a42 = b42 ^ ((~b02) & b12);                                          \
        a03 = b03 ^ ((~b13) & b23);                                          \
        a13 = b13 ^ ((~b23) & b33);                                          \
        a23 = b23 ^ ((~b33) & b43);                                          \
        a33 = b33 ^ ((~b43) & b03);                                          \
        a43 = b43 ^ ((~b03) & b13);                                          \
        a04 = b04 ^ ((~b14) & b24);                                          \
        a14 = b14 ^ ((~b24) & b34);                                          \
        a24 = b24 ^ ((~b34) & b44);                                          \
        a34 = b34 ^ ((~b44) & b04);                                          \
        a44 = b44 ^ ((~b04) & b14);                                          \
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

    // Absorb message into lanes 0..msg_lanes-1 via a small array (loop-friendly).
    uint in_base = idx * msg_lanes;
    {
        ulong tmp[25];
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
        tmp[msg_lanes]       ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00 = tmp[0];  a10 = tmp[1];  a20 = tmp[2];  a30 = tmp[3];  a40 = tmp[4];
        a01 = tmp[5];  a11 = tmp[6];  a21 = tmp[7];  a31 = tmp[8];  a41 = tmp[9];
        a02 = tmp[10]; a12 = tmp[11]; a22 = tmp[12]; a32 = tmp[13]; a42 = tmp[14];
        a03 = tmp[15]; a13 = tmp[16]; a23 = tmp[17]; a33 = tmp[18]; a43 = tmp[19];
        a04 = tmp[20]; a14 = tmp[21]; a24 = tmp[22]; a34 = tmp[23]; a44 = tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;
    for (;;) {
        KECCAK_ROUND(KECCAK_RC[0])
        KECCAK_ROUND(KECCAK_RC[1])
        KECCAK_ROUND(KECCAK_RC[2])
        KECCAK_ROUND(KECCAK_RC[3])
        KECCAK_ROUND(KECCAK_RC[4])
        KECCAK_ROUND(KECCAK_RC[5])
        KECCAK_ROUND(KECCAK_RC[6])
        KECCAK_ROUND(KECCAK_RC[7])
        KECCAK_ROUND(KECCAK_RC[8])
        KECCAK_ROUND(KECCAK_RC[9])
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

        // Squeeze lanes 0..min(rate_lanes, remaining)-1.
        // State lane k is stored in aXY where x=k%5, y=k/5.
        ulong squeezed[25];
        squeezed[0]=a00;  squeezed[1]=a10;  squeezed[2]=a20;  squeezed[3]=a30;  squeezed[4]=a40;
        squeezed[5]=a01;  squeezed[6]=a11;  squeezed[7]=a21;  squeezed[8]=a31;  squeezed[9]=a41;
        squeezed[10]=a02; squeezed[11]=a12; squeezed[12]=a22; squeezed[13]=a32; squeezed[14]=a42;
        squeezed[15]=a03; squeezed[16]=a13; squeezed[17]=a23; squeezed[18]=a33; squeezed[19]=a43;
        squeezed[20]=a04; squeezed[21]=a14; squeezed[22]=a24; squeezed[23]=a34; squeezed[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = squeezed[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}