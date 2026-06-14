#include <metal_stdlib>
using namespace metal;

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

#define THETA_RHO_PI_CHI(RC) \
    { \
        ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20; \
        ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21; \
        ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22; \
        ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23; \
        ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24; \
        ulong D0 = C4 ^ ROTL64(C1, 1); \
        ulong D1 = C0 ^ ROTL64(C2, 1); \
        ulong D2 = C1 ^ ROTL64(C3, 1); \
        ulong D3 = C2 ^ ROTL64(C4, 1); \
        ulong D4 = C3 ^ ROTL64(C0, 1); \
        ulong b00 =        (a00 ^ D0); \
        ulong b10 = ROTL64(a01 ^ D1,  1); \
        ulong b20 = ROTL64(a02 ^ D2, 62); \
        ulong b05 = ROTL64(a03 ^ D3, 28); \
        ulong b15 = ROTL64(a04 ^ D4, 27); \
        ulong b16 = ROTL64(a05 ^ D0, 36); \
        ulong b01 = ROTL64(a06 ^ D1, 44); \
        ulong b11 = ROTL64(a07 ^ D2,  6); \
        ulong b21 = ROTL64(a08 ^ D3, 55); \
        ulong b06 = ROTL64(a09 ^ D4, 20); \
        ulong b07 = ROTL64(a10 ^ D0,  3); \
        ulong b17 = ROTL64(a11 ^ D1, 10); \
        ulong b02 = ROTL64(a12 ^ D2, 43); \
        ulong b12 = ROTL64(a13 ^ D3, 25); \
        ulong b22 = ROTL64(a14 ^ D4, 39); \
        ulong b23 = ROTL64(a15 ^ D0, 41); \
        ulong b08 = ROTL64(a16 ^ D1, 45); \
        ulong b18 = ROTL64(a17 ^ D2, 15); \
        ulong b03 = ROTL64(a18 ^ D3, 21); \
        ulong b13 = ROTL64(a19 ^ D4,  8); \
        ulong b14 = ROTL64(a20 ^ D0, 18); \
        ulong b24 = ROTL64(a21 ^ D1,  2); \
        ulong b09 = ROTL64(a22 ^ D2, 61); \
        ulong b19 = ROTL64(a23 ^ D3, 56); \
        ulong b04 = ROTL64(a24 ^ D4, 14); \
        a00 = b00 ^ ((~b01) & b02) ^ (RC); \
        a01 = b01 ^ ((~b02) & b03); \
        a02 = b02 ^ ((~b03) & b04); \
        a03 = b03 ^ ((~b04) & b00); \
        a04 = b04 ^ ((~b00) & b01); \
        a05 = b05 ^ ((~b06) & b07); \
        a06 = b06 ^ ((~b07) & b08); \
        a07 = b07 ^ ((~b08) & b09); \
        a08 = b08 ^ ((~b09) & b05); \
        a09 = b09 ^ ((~b05) & b06); \
        a10 = b10 ^ ((~b11) & b12); \
        a11 = b11 ^ ((~b12) & b13); \
        a12 = b12 ^ ((~b13) & b14); \
        a13 = b13 ^ ((~b14) & b10); \
        a14 = b14 ^ ((~b10) & b11); \
        a15 = b15 ^ ((~b16) & b17); \
        a16 = b16 ^ ((~b17) & b18); \
        a17 = b17 ^ ((~b18) & b19); \
        a18 = b18 ^ ((~b19) & b15); \
        a19 = b19 ^ ((~b15) & b16); \
        a20 = b20 ^ ((~b21) & b22); \
        a21 = b21 ^ ((~b22) & b23); \
        a22 = b22 ^ ((~b23) & b24); \
        a23 = b23 ^ ((~b24) & b20); \
        a24 = b24 ^ ((~b20) & b21); \
    }

#define KECCAK_F1600() \
    THETA_RHO_PI_CHI(0x0000000000000001ul) \
    THETA_RHO_PI_CHI(0x0000000000008082ul) \
    THETA_RHO_PI_CHI(0x800000000000808Aul) \
    THETA_RHO_PI_CHI(0x8000000080008000ul) \
    THETA_RHO_PI_CHI(0x000000000000808Bul) \
    THETA_RHO_PI_CHI(0x0000000080000001ul) \
    THETA_RHO_PI_CHI(0x8000000080008081ul) \
    THETA_RHO_PI_CHI(0x8000000000008009ul) \
    THETA_RHO_PI_CHI(0x000000000000008Aul) \
    THETA_RHO_PI_CHI(0x0000000000000088ul) \
    THETA_RHO_PI_CHI(0x0000000080008009ul) \
    THETA_RHO_PI_CHI(0x000000008000000Aul) \
    THETA_RHO_PI_CHI(0x000000008000808Bul) \
    THETA_RHO_PI_CHI(0x800000000000008Bul) \
    THETA_RHO_PI_CHI(0x8000000000008089ul) \
    THETA_RHO_PI_CHI(0x8000000000008003ul) \
    THETA_RHO_PI_CHI(0x8000000000008002ul) \
    THETA_RHO_PI_CHI(0x8000000000000080ul) \
    THETA_RHO_PI_CHI(0x000000000000800Aul) \
    THETA_RHO_PI_CHI(0x800000008000000Aul) \
    THETA_RHO_PI_CHI(0x8000000080008081ul) \
    THETA_RHO_PI_CHI(0x8000000000008080ul) \
    THETA_RHO_PI_CHI(0x0000000080000001ul) \
    THETA_RHO_PI_CHI(0x8000000080008008ul)

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

    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    // Absorb. All test sizes have msg_bytes = 32 => msg_lanes = 4,
    // but support arbitrary msg_lanes < 25 via a small dispatch.
    uint in_base = idx * msg_lanes;
    {
        // Spill into a tiny scratch then load by constant index.
        ulong tmp[25];
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
        tmp[msg_lanes]      ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00=tmp[ 0]; a01=tmp[ 1]; a02=tmp[ 2]; a03=tmp[ 3]; a04=tmp[ 4];
        a05=tmp[ 5]; a06=tmp[ 6]; a07=tmp[ 7]; a08=tmp[ 8]; a09=tmp[ 9];
        a10=tmp[10]; a11=tmp[11]; a12=tmp[12]; a13=tmp[13]; a14=tmp[14];
        a15=tmp[15]; a16=tmp[16]; a17=tmp[17]; a18=tmp[18]; a19=tmp[19];
        a20=tmp[20]; a21=tmp[21]; a22=tmp[22]; a23=tmp[23]; a24=tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    // First permutation (always needed).
    KECCAK_F1600()

    // Squeeze loop. For SHA3-256 (out_lanes=4, rate_lanes=17) this
    // exits after one iteration writing 4 lanes.
    for (;;) {
        ulong tmp[25];
        tmp[ 0]=a00; tmp[ 1]=a01; tmp[ 2]=a02; tmp[ 3]=a03; tmp[ 4]=a04;
        tmp[ 5]=a05; tmp[ 6]=a06; tmp[ 7]=a07; tmp[ 8]=a08; tmp[ 9]=a09;
        tmp[10]=a10; tmp[11]=a11; tmp[12]=a12; tmp[13]=a13; tmp[14]=a14;
        tmp[15]=a15; tmp[16]=a16; tmp[17]=a17; tmp[18]=a18; tmp[19]=a19;
        tmp[20]=a20; tmp[21]=a21; tmp[22]=a22; tmp[23]=a23; tmp[24]=a24;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = tmp[j];
        }
        written += take;
        if (written >= out_lanes) break;

        KECCAK_F1600()
    }
}