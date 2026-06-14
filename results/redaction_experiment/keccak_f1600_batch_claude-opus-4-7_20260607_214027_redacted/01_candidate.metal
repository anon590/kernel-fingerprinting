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

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

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

    // 25 state lanes laid out as (x + 5*y).
    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // Absorb (single block).
    uint in_base = idx * msg_lanes;
    ulong *Aarr[25] = {
        &a00,&a01,&a02,&a03,&a04,
        &a05,&a06,&a07,&a08,&a09,
        &a10,&a11,&a12,&a13,&a14,
        &a15,&a16,&a17,&a18,&a19,
        &a20,&a21,&a22,&a23,&a24
    };
    for (uint i = 0u; i < msg_lanes; ++i) {
        *Aarr[i] ^= in_data[in_base + i];
    }
    *Aarr[msg_lanes] ^= (ulong)(domain & 0xFFu);
    *Aarr[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // ---- 24 rounds of Keccak-f[1600], fully unrolled per round ----
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROTL64(C1, 1);
            ulong D1 = C0 ^ ROTL64(C2, 1);
            ulong D2 = C1 ^ ROTL64(C3, 1);
            ulong D3 = C2 ^ ROTL64(C4, 1);
            ulong D4 = C3 ^ ROTL64(C0, 1);

            ulong t00 = a00 ^ D0;
            ulong t01 = a01 ^ D1;
            ulong t02 = a02 ^ D2;
            ulong t03 = a03 ^ D3;
            ulong t04 = a04 ^ D4;
            ulong t05 = a05 ^ D0;
            ulong t06 = a06 ^ D1;
            ulong t07 = a07 ^ D2;
            ulong t08 = a08 ^ D3;
            ulong t09 = a09 ^ D4;
            ulong t10 = a10 ^ D0;
            ulong t11 = a11 ^ D1;
            ulong t12 = a12 ^ D2;
            ulong t13 = a13 ^ D3;
            ulong t14 = a14 ^ D4;
            ulong t15 = a15 ^ D0;
            ulong t16 = a16 ^ D1;
            ulong t17 = a17 ^ D2;
            ulong t18 = a18 ^ D3;
            ulong t19 = a19 ^ D4;
            ulong t20 = a20 ^ D0;
            ulong t21 = a21 ^ D1;
            ulong t22 = a22 ^ D2;
            ulong t23 = a23 ^ D3;
            ulong t24 = a24 ^ D4;

            // rho + pi: B[y, (2x+3y)%5] = rotl(t[x,y], r[x][y])
            // i.e. B[x_new + 5*y_new] with x_new=y, y_new=(2x+3y)%5
            // Rho offsets r[x][y] indexed by (x+5y):
            //  0  1 62 28 27
            // 36 44  6 55 20
            //  3 10 43 25 39
            // 41 45 15 21  8
            // 18  2 61 56 14
            ulong b00 = t00;                 // (0,0)->(0,0), rot 0
            ulong b10 = ROTL64(t01,  1);     // (1,0)->(0,2)
            ulong b20 = ROTL64(t02, 62);     // (2,0)->(0,4)
            ulong b05 = ROTL64(t03, 28);     // (3,0)->(0,1)
            ulong b15 = ROTL64(t04, 27);     // (4,0)->(0,3)

            ulong b16 = ROTL64(t05, 36);     // (0,1)->(1,3)
            ulong b01 = ROTL64(t06, 44);     // (1,1)->(1,0)
            ulong b11 = ROTL64(t07,  6);     // (2,1)->(1,2)
            ulong b21 = ROTL64(t08, 55);     // (3,1)->(1,4)
            ulong b06 = ROTL64(t09, 20);     // (4,1)->(1,1)

            ulong b07 = ROTL64(t10,  3);     // (0,2)->(2,1)
            ulong b17 = ROTL64(t11, 10);     // (1,2)->(2,3)
            ulong b02 = ROTL64(t12, 43);     // (2,2)->(2,0)
            ulong b12 = ROTL64(t13, 25);     // (3,2)->(2,2)
            ulong b22 = ROTL64(t14, 39);     // (4,2)->(2,4)

            ulong b23 = ROTL64(t15, 41);     // (0,3)->(3,4)
            ulong b08 = ROTL64(t16, 45);     // (1,3)->(3,1)
            ulong b18 = ROTL64(t17, 15);     // (2,3)->(3,3)
            ulong b03 = ROTL64(t18, 21);     // (3,3)->(3,0)
            ulong b13 = ROTL64(t19,  8);     // (4,3)->(3,2)

            ulong b14 = ROTL64(t20, 18);     // (0,4)->(4,2)
            ulong b24 = ROTL64(t21,  2);     // (1,4)->(4,4)
            ulong b09 = ROTL64(t22, 61);     // (2,4)->(4,1)
            ulong b19 = ROTL64(t23, 56);     // (3,4)->(4,3)
            ulong b04 = ROTL64(t24, 14);     // (4,4)->(4,0)

            // chi: A[x,y] = B[x,y] ^ ((~B[x+1,y]) & B[x+2,y])
            a00 = b00 ^ ((~b01) & b02);
            a01 = b01 ^ ((~b02) & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = b08 ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            a10 = b10 ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = b12 ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & b10);
            a14 = b14 ^ ((~b10) & b11);

            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & b18);
            a17 = b17 ^ ((~b18) & b19);
            a18 = b18 ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            a20 = b20 ^ ((~b21) & b22);
            a21 = b21 ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & b20);
            a24 = b24 ^ ((~b20) & b21);

            // iota
            a00 ^= KECCAK_RC[r];
        }

        // Squeeze rate_lanes lanes (or fewer for the final chunk).
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            ulong v;
            switch (j) {
                case  0: v = a00; break; case  1: v = a01; break;
                case  2: v = a02; break; case  3: v = a03; break;
                case  4: v = a04; break; case  5: v = a05; break;
                case  6: v = a06; break; case  7: v = a07; break;
                case  8: v = a08; break; case  9: v = a09; break;
                case 10: v = a10; break; case 11: v = a11; break;
                case 12: v = a12; break; case 13: v = a13; break;
                case 14: v = a14; break; case 15: v = a15; break;
                case 16: v = a16; break; case 17: v = a17; break;
                case 18: v = a18; break; case 19: v = a19; break;
                case 20: v = a20; break; case 21: v = a21; break;
                case 22: v = a22; break; case 23: v = a23; break;
                default: v = a24; break;
            }
            out_data[out_base + written + j] = v;
        }
        written += take;
        if (written >= out_lanes) break;
    }
}