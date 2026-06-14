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

// One Keccak round on 25 named lanes. Lane naming: a{x}{y}.
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
        /* theta + rho + pi: compute rotated lanes b{x}{y} (destination layout). */ \
        /* B[x_new + 5*y_new] = rotl(A[x + 5*y] ^ D[x], r[x][y])             */ \
        /* with (x_new, y_new) = (y, (2x+3y)%5)                              */ \
        ulong b00 = (a00 ^ d0);                                              \
        ulong b10 = ROTL64(a30 ^ d3, 28);                                    \
        ulong b20 = ROTL64(a10 ^ d1, 1);                                     \
        ulong b30 = ROTL64(a40 ^ d4, 27);                                    \
        ulong b40 = ROTL64(a20 ^ d2, 62);                                    \
        ulong b01 = ROTL64(a11 ^ d1, 44);                                    \
        ulong b11 = ROTL64(a41 ^ d4, 20);                                    \
        ulong b21 = ROTL64(a21 ^ d2, 6);                                     \
        ulong b31 = ROTL64(a01 ^ d0, 36);                                    \
        ulong b41 = ROTL64(a31 ^ d3, 55);                                    \
        ulong b02 = ROTL64(a22 ^ d2, 43);                                    \
        ulong b12 = ROTL64(a02 ^ d0, 3);                                     \
        ulong b22 = ROTL64(a32 ^ d3, 25);                                    \
        ulong b32 = ROTL64(a12 ^ d1, 10);                                    \
        ulong b42 = ROTL64(a42 ^ d4, 39);                                    \
        ulong b03 = ROTL64(a33 ^ d3, 21);                                    \
        ulong b13 = ROTL64(a13 ^ d1, 45);                                    \
        ulong b23 = ROTL64(a43 ^ d4, 8);                                     \
        ulong b33 = ROTL64(a23 ^ d2, 15);                                    \
        ulong b43 = ROTL64(a03 ^ d0, 41);                                    \
        ulong b04 = ROTL64(a44 ^ d4, 14);                                    \
        ulong b14 = ROTL64(a24 ^ d2, 61);                                    \
        ulong b24 = ROTL64(a04 ^ d0, 18);                                    \
        ulong b34 = ROTL64(a34 ^ d3, 56);                                    \
        ulong b44 = ROTL64(a14 ^ d1, 2);                                     \
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

    // Initialise 25 lanes in registers.
    ulong a00 = 0, a10 = 0, a20 = 0, a30 = 0, a40 = 0;
    ulong a01 = 0, a11 = 0, a21 = 0, a31 = 0, a41 = 0;
    ulong a02 = 0, a12 = 0, a22 = 0, a32 = 0, a42 = 0;
    ulong a03 = 0, a13 = 0, a23 = 0, a33 = 0, a43 = 0;
    ulong a04 = 0, a14 = 0, a24 = 0, a34 = 0, a44 = 0;

    // Absorb msg_lanes from in_data into lanes 0..msg_lanes-1.
    // Common test path: msg_bytes = 32 (4 lanes).
    uint in_base = idx * msg_lanes;
    device const ulong *msg = in_data + in_base;

    // Generic absorb via a small dispatch (msg_lanes is at most rate_lanes-1).
    // We use a temp array small enough to live in registers/threadlocal.
    {
        ulong tmp[25];
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = msg[i];
        tmp[msg_lanes]     ^= (ulong)(domain & 0xFFu);
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
        // 24 unrolled rounds.
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

        // Squeeze up to rate_lanes lanes (or fewer for last chunk).
        ulong squeezed[25];
        squeezed[0]  = a00; squeezed[1]  = a10; squeezed[2]  = a20; squeezed[3]  = a30; squeezed[4]  = a40;
        squeezed[5]  = a01; squeezed[6]  = a11; squeezed[7]  = a21; squeezed[8]  = a31; squeezed[9]  = a41;
        squeezed[10] = a02; squeezed[11] = a12; squeezed[12] = a22; squeezed[13] = a32; squeezed[14] = a42;
        squeezed[15] = a03; squeezed[16] = a13; squeezed[17] = a23; squeezed[18] = a33; squeezed[19] = a43;
        squeezed[20] = a04; squeezed[21] = a14; squeezed[22] = a24; squeezed[23] = a34; squeezed[24] = a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = squeezed[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}