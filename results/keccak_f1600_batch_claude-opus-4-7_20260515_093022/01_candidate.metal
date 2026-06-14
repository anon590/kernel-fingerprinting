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

inline ulong ROTL(ulong x, uint k) {
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

    // 25 state lanes as named registers.
    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;
    ulong a30=0, a31=0, a32=0, a33=0, a34=0;
    ulong a40=0, a41=0, a42=0, a43=0, a44=0;

    // Absorb. Use an array view trick: write into a small thread-array then
    // hoist back; but to keep registers, do it via switch on msg_lanes==4
    // (covers the common test case) plus a generic fallback.
    uint in_base = idx * msg_lanes;
    // Generic absorb: load up to 25 lanes (msg_lanes < rate_lanes <= 21).
    // We'll use a small thread array only for the absorb, then copy out.
    {
        ulong tmp[25];
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
        // Padding
        tmp[msg_lanes]      ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;

        a00 = tmp[ 0]; a10 = tmp[ 1]; a20 = tmp[ 2]; a30 = tmp[ 3]; a40 = tmp[ 4];
        a01 = tmp[ 5]; a11 = tmp[ 6]; a21 = tmp[ 7]; a31 = tmp[ 8]; a41 = tmp[ 9];
        a02 = tmp[10]; a12 = tmp[11]; a22 = tmp[12]; a32 = tmp[13]; a42 = tmp[14];
        a03 = tmp[15]; a13 = tmp[16]; a23 = tmp[17]; a33 = tmp[18]; a43 = tmp[19];
        a04 = tmp[20]; a14 = tmp[21]; a24 = tmp[22]; a34 = tmp[23]; a44 = tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // 24 rounds, fully unrolled.
        #pragma unroll
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;
            ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;
            ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;
            ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;
            ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;

            ulong D0 = C4 ^ ROTL(C1, 1);
            ulong D1 = C0 ^ ROTL(C2, 1);
            ulong D2 = C1 ^ ROTL(C3, 1);
            ulong D3 = C2 ^ ROTL(C4, 1);
            ulong D4 = C3 ^ ROTL(C0, 1);

            // theta XOR + rho rotate + pi permute, fused.
            // dest[y, (2x+3y)%5] = rotl(A[x,y] ^ D[x], rho[x,y])
            // Use lane naming a{x}{y}.
            ulong b00 = ROTL(a00 ^ D0,  0);
            ulong b10 = ROTL(a30 ^ D3, 28);
            ulong b20 = ROTL(a10 ^ D1,  1);
            ulong b30 = ROTL(a40 ^ D4, 27);
            ulong b40 = ROTL(a20 ^ D2, 62);

            ulong b01 = ROTL(a11 ^ D1, 44);
            ulong b11 = ROTL(a41 ^ D4, 20);
            ulong b21 = ROTL(a21 ^ D2,  6);
            ulong b31 = ROTL(a01 ^ D0, 36);
            ulong b41 = ROTL(a31 ^ D3, 55);

            ulong b02 = ROTL(a22 ^ D2, 43);
            ulong b12 = ROTL(a02 ^ D0,  3);
            ulong b22 = ROTL(a32 ^ D3, 25);
            ulong b32 = ROTL(a12 ^ D1, 10);
            ulong b42 = ROTL(a42 ^ D4, 39);

            ulong b03 = ROTL(a33 ^ D3, 21);
            ulong b13 = ROTL(a13 ^ D1, 45);
            ulong b23 = ROTL(a43 ^ D4,  8);
            ulong b33 = ROTL(a23 ^ D2, 15);
            ulong b43 = ROTL(a03 ^ D0, 41);

            ulong b04 = ROTL(a44 ^ D4, 14);
            ulong b14 = ROTL(a24 ^ D2, 61);
            ulong b24 = ROTL(a04 ^ D0, 18);
            ulong b34 = ROTL(a34 ^ D3, 56);
            ulong b44 = ROTL(a14 ^ D1,  2);

            // chi
            a00 = b00 ^ ((~b10) & b20);
            a10 = b10 ^ ((~b20) & b30);
            a20 = b20 ^ ((~b30) & b40);
            a30 = b30 ^ ((~b40) & b00);
            a40 = b40 ^ ((~b00) & b10);

            a01 = b01 ^ ((~b11) & b21);
            a11 = b11 ^ ((~b21) & b31);
            a21 = b21 ^ ((~b31) & b41);
            a31 = b31 ^ ((~b41) & b01);
            a41 = b41 ^ ((~b01) & b11);

            a02 = b02 ^ ((~b12) & b22);
            a12 = b12 ^ ((~b22) & b32);
            a22 = b22 ^ ((~b32) & b42);
            a32 = b32 ^ ((~b42) & b02);
            a42 = b42 ^ ((~b02) & b12);

            a03 = b03 ^ ((~b13) & b23);
            a13 = b13 ^ ((~b23) & b33);
            a23 = b23 ^ ((~b33) & b43);
            a33 = b33 ^ ((~b43) & b03);
            a43 = b43 ^ ((~b03) & b13);

            a04 = b04 ^ ((~b14) & b24);
            a14 = b14 ^ ((~b24) & b34);
            a24 = b24 ^ ((~b34) & b44);
            a34 = b34 ^ ((~b44) & b04);
            a44 = b44 ^ ((~b04) & b14);

            // iota
            a00 ^= KECCAK_RC[r];
        }

        // Squeeze up to rate_lanes lanes.
        ulong lanes[25] = {
            a00, a10, a20, a30, a40,
            a01, a11, a21, a31, a41,
            a02, a12, a22, a32, a42,
            a03, a13, a23, a33, a43,
            a04, a14, a24, a34, a44
        };

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = lanes[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}