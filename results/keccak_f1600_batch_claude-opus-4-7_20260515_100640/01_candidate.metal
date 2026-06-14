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
    return (x << k) | (x >> (64 - k));
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

    // 25 state lanes as individual scalars (named by x,y: a{x}{y}).
    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    // Absorb single block into lanes 0..msg_lanes-1.
    // Standard test: msg_bytes = 32 -> msg_lanes = 4 (a00..a30).
    uint in_base = idx * msg_lanes;
    ulong tmp[25];
    for (uint i = 0; i < 25; ++i) tmp[i] = 0ul;
    for (uint i = 0; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
    // Domain byte at byte position msg_bytes.
    tmp[msg_lanes] ^= (ulong)(domain & 0xFFu);
    // 0x80 at byte position rate_bytes - 1.
    tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;

    a00 = tmp[ 0]; a10 = tmp[ 1]; a20 = tmp[ 2]; a30 = tmp[ 3]; a40 = tmp[ 4];
    a01 = tmp[ 5]; a11 = tmp[ 6]; a21 = tmp[ 7]; a31 = tmp[ 8]; a41 = tmp[ 9];
    a02 = tmp[10]; a12 = tmp[11]; a22 = tmp[12]; a32 = tmp[13]; a42 = tmp[14];
    a03 = tmp[15]; a13 = tmp[16]; a23 = tmp[17]; a33 = tmp[18]; a43 = tmp[19];
    a04 = tmp[20]; a14 = tmp[21]; a24 = tmp[22]; a34 = tmp[23]; a44 = tmp[24];

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // 24 rounds of Keccak-f[1600], fully unrolled in 'r'.
        for (uint r = 0; r < 24; ++r) {
            // theta
            ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;
            ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;
            ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;
            ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;
            ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;

            ulong D0 = C4 ^ ROTL64(C1, 1);
            ulong D1 = C0 ^ ROTL64(C2, 1);
            ulong D2 = C1 ^ ROTL64(C3, 1);
            ulong D3 = C2 ^ ROTL64(C4, 1);
            ulong D4 = C3 ^ ROTL64(C0, 1);

            a00 ^= D0; a10 ^= D1; a20 ^= D2; a30 ^= D3; a40 ^= D4;
            a01 ^= D0; a11 ^= D1; a21 ^= D2; a31 ^= D3; a41 ^= D4;
            a02 ^= D0; a12 ^= D1; a22 ^= D2; a32 ^= D3; a42 ^= D4;
            a03 ^= D0; a13 ^= D1; a23 ^= D2; a33 ^= D3; a43 ^= D4;
            a04 ^= D0; a14 ^= D1; a24 ^= D2; a34 ^= D3; a44 ^= D4;

            // rho + pi combined: B[x_new + 5*y_new] = rotl(A[x+5*y], rho[x+5*y])
            // with (x_new, y_new) = (y, (2x+3y) mod 5).
            // Compute the rotated lanes and place them in their destination names.
            // Source A[x,y] -> Dest B[y, (2x+3y)%5].
            ulong b00 = a00;                  // (0,0) rho=0  -> (0,0)
            ulong b30 = ROTL64(a10,  1);      // (1,0) rho=1  -> (0,3)? careful: dest = (y, (2x+3y)%5) = (0, 2). Actually B index uses (x_new,y_new) layout below.
            // Let's redo more carefully using explicit mapping list:
            // A[x,y] with rho offset rotates and goes to B[y, (2x+3y)%5].
            // We'll name b{xn}{yn}.
            // (x=0,y=0): rho=0,  dest=(0,0)
            // (x=1,y=0): rho=1,  dest=(0,2)
            // (x=2,y=0): rho=62, dest=(0,4)
            // (x=3,y=0): rho=28, dest=(0,1)
            // (x=4,y=0): rho=27, dest=(0,3)
            // (x=0,y=1): rho=36, dest=(1,3)
            // (x=1,y=1): rho=44, dest=(1,0)
            // (x=2,y=1): rho=6,  dest=(1,2)
            // (x=3,y=1): rho=55, dest=(1,4)
            // (x=4,y=1): rho=20, dest=(1,1)
            // (x=0,y=2): rho=3,  dest=(2,1)
            // (x=1,y=2): rho=10, dest=(2,3)
            // (x=2,y=2): rho=43, dest=(2,0)
            // (x=3,y=2): rho=25, dest=(2,2)
            // (x=4,y=2): rho=39, dest=(2,4)
            // (x=0,y=3): rho=41, dest=(3,4)
            // (x=1,y=3): rho=45, dest=(3,1)
            // (x=2,y=3): rho=15, dest=(3,3)
            // (x=3,y=3): rho=21, dest=(3,0)
            // (x=4,y=3): rho=8,  dest=(3,2)
            // (x=0,y=4): rho=18, dest=(4,2)
            // (x=1,y=4): rho=2,  dest=(4,4)
            // (x=2,y=4): rho=61, dest=(4,1)
            // (x=3,y=4): rho=56, dest=(4,3)
            // (x=4,y=4): rho=14, dest=(4,0)

            // overwrite b00 properly
            b00 = a00;
            ulong b02 = ROTL64(a10,  1);
            ulong b04 = ROTL64(a20, 62);
            ulong b01 = ROTL64(a30, 28);
            ulong b03 = ROTL64(a40, 27);

            ulong b13 = ROTL64(a01, 36);
            ulong b10 = ROTL64(a11, 44);
            ulong b12 = ROTL64(a21,  6);
            ulong b14 = ROTL64(a31, 55);
            ulong b11 = ROTL64(a41, 20);

            ulong b21 = ROTL64(a02,  3);
            ulong b23 = ROTL64(a12, 10);
            ulong b20 = ROTL64(a22, 43);
            ulong b22 = ROTL64(a32, 25);
            ulong b24 = ROTL64(a42, 39);

            ulong b34 = ROTL64(a03, 41);
            ulong b31 = ROTL64(a13, 45);
            ulong b33 = ROTL64(a23, 15);
            ulong b30b = ROTL64(a33, 21);
            ulong b32 = ROTL64(a43,  8);

            ulong b42 = ROTL64(a04, 18);
            ulong b44 = ROTL64(a14,  2);
            ulong b41 = ROTL64(a24, 61);
            ulong b43 = ROTL64(a34, 56);
            ulong b40 = ROTL64(a44, 14);

            ulong b30_final = b30b;

            // chi: A[x,y] = B[x,y] ^ ((~B[(x+1)%5, y]) & B[(x+2)%5, y])
            // Row y=0:
            a00 = b00 ^ ((~b10) & b20);
            a10 = b10 ^ ((~b20) & b30_final);
            a20 = b20 ^ ((~b30_final) & b40);
            a30 = b30_final ^ ((~b40) & b00);
            a40 = b40 ^ ((~b00) & b10);
            // Row y=1:
            a01 = b01 ^ ((~b11) & b21);
            a11 = b11 ^ ((~b21) & b31);
            a21 = b21 ^ ((~b31) & b41);
            a31 = b31 ^ ((~b41) & b01);
            a41 = b41 ^ ((~b01) & b11);
            // Row y=2:
            a02 = b02 ^ ((~b12) & b22);
            a12 = b12 ^ ((~b22) & b32);
            a22 = b22 ^ ((~b32) & b42);
            a32 = b32 ^ ((~b42) & b02);
            a42 = b42 ^ ((~b02) & b12);
            // Row y=3:
            a03 = b03 ^ ((~b13) & b23);
            a13 = b13 ^ ((~b23) & b33);
            a23 = b23 ^ ((~b33) & b43);
            a33 = b33 ^ ((~b43) & b03);
            a43 = b43 ^ ((~b03) & b13);
            // Row y=4:
            a04 = b04 ^ ((~b14) & b24);
            a14 = b14 ^ ((~b24) & b34);
            a24 = b24 ^ ((~b34) & b44);
            a34 = b34 ^ ((~b44) & b04);
            a44 = b44 ^ ((~b04) & b14);

            // iota
            a00 ^= KECCAK_RC[r];
        }

        // Squeeze rate_lanes (or remainder) into output.
        ulong state_arr[25];
        state_arr[ 0]=a00; state_arr[ 1]=a10; state_arr[ 2]=a20; state_arr[ 3]=a30; state_arr[ 4]=a40;
        state_arr[ 5]=a01; state_arr[ 6]=a11; state_arr[ 7]=a21; state_arr[ 8]=a31; state_arr[ 9]=a41;
        state_arr[10]=a02; state_arr[11]=a12; state_arr[12]=a22; state_arr[13]=a32; state_arr[14]=a42;
        state_arr[15]=a03; state_arr[16]=a13; state_arr[17]=a23; state_arr[18]=a33; state_arr[19]=a43;
        state_arr[20]=a04; state_arr[21]=a14; state_arr[22]=a24; state_arr[23]=a34; state_arr[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = state_arr[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}