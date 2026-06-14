**Optimization**: The incumbent uses a local `ulong a[25]` array which the compiler may spill to threadgroup/stack memory because of the indexed accesses in the absorb/squeeze loops. The previous attempt with scalar registers used switch statements that hurt performance. The key insight: since all test sizes have `msg_bytes = 32` (so `msg_lanes = 4`), and `out_lanes <= rate_lanes` for SHA3-256 (4 ≤ 17), I'll fast-path the common case with pure scalar registers (no array, no switches), keeping the Keccak permutation fully inlined in registers. The SHAKE128 path uses the array form. This avoids both the register-spill risk of arrays and the switch overhead, giving us best of both worlds. I also remove the dead pragma unroll and write the message lanes directly to named registers.

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

static inline void keccak_f1600_state(thread ulong *a) {
    ulong a0=a[0],a1=a[1],a2=a[2],a3=a[3],a4=a[4];
    ulong a5=a[5],a6=a[6],a7=a[7],a8=a[8],a9=a[9];
    ulong a10=a[10],a11=a[11],a12=a[12],a13=a[13],a14=a[14];
    ulong a15=a[15],a16=a[16],a17=a[17],a18=a[18],a19=a[19];
    ulong a20=a[20],a21=a[21],a22=a[22],a23=a[23],a24=a[24];

    KECCAK_24_ROUNDS();

    a[0]=a0;a[1]=a1;a[2]=a2;a[3]=a3;a[4]=a4;
    a[5]=a5;a[6]=a6;a[7]=a7;a[8]=a8;a[9]=a9;
    a[10]=a10;a[11]=a11;a[12]=a12;a[13]=a13;a[14]=a14;
    a[15]=a15;a[16]=a16;a[17]=a17;a[18]=a18;a[19]=a19;
    a[20]=a20;a[21]=a21;a[22]=a22;a[23]=a23;a[24]=a24;
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

    // Fast path: SHA3-256 style — msg_lanes == 4, single squeeze (out_lanes <= rate_lanes).
    // All in registers, no array indexing.
    if (msg_lanes == 4u && out_lanes <= rate_lanes) {
        uint in_base = idx * 4u;
        ulong a0 = in_data[in_base + 0];
        ulong a1 = in_data[in_base + 1];
        ulong a2 = in_data[in_base + 2];
        ulong a3 = in_data[in_base + 3];
        ulong a4 = (ulong)(domain & 0xFFu);
        ulong a5=0,a6=0,a7=0,a8=0,a9=0;
        ulong a10=0,a11=0,a12=0,a13=0,a14=0;
        ulong a15=0,a16=0,a17=0,a18=0,a19=0;
        ulong a20=0,a21=0,a22=0,a23=0,a24=0;

        // 0x80 at byte (rate_bytes - 1): lane (rate_lanes-1), byte 7.
        ulong pad = 0x8000000000000000ul;
        uint last = rate_lanes - 1u;
        // For SHA3-256, last = 16; for SHAKE128 (not this path), last = 20.
        // Use conditional assignment without branches.
        if (last == 16u) a16 ^= pad;
        else if (last == 20u) a20 ^= pad;
        else {
            // Generic fallback (shouldn't hit on tested paths)
            switch (last) {
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
                case 17: a17 ^= pad; break;
                case 18: a18 ^= pad; break;
                case 19: a19 ^= pad; break;
                case 21: a21 ^= pad; break;
                case 22: a22 ^= pad; break;
                case 23: a23 ^= pad; break;
                case 24: a24 ^= pad; break;
                default: break;
            }
        }

        KECCAK_24_ROUNDS();

        uint out_base = idx * out_lanes;
        // out_lanes <= rate_lanes; typical SHA3-256 has out_lanes = 4.
        if (out_lanes >= 1u) out_data[out_base + 0] = a0;
        if (out_lanes >= 2u) out_data[out_base + 1] = a1;
        if (out_lanes >= 3u) out_data[out_base + 2] = a2;
        if (out_lanes >= 4u) out_data[out_base + 3] = a3;
        // Longer outputs (still single squeeze): handle generically.
        if (out_lanes > 4u) {
            ulong rest[21] = {a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,
                              a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24};
            for (uint j = 4; j < out_lanes; ++j) {
                out_data[out_base + j] = rest[j - 4u];
            }
        }
        return;
    }

    // General path: array-based state for SHAKE / non-4-lane messages.
    ulong a[25];
    for (uint i = 0; i < 25; ++i) a[i] = 0ul;

    uint in_base = idx * msg_lanes;
    for (uint i = 0; i < msg_lanes; ++i) a[i] = in_data[in_base + i];
    a[msg_lanes]       ^= (ulong)(domain & 0xFFu);
    a[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;

    keccak_f1600_state(a);

    if (out_lanes <= rate_lanes) {
        for (uint j = 0; j < out_lanes; ++j) {
            out_data[out_base + j] = a[j];
        }
        return;
    }

    uint written = 0u;
    for (;;) {
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = a[j];
        }
        written += take;
        if (written >= out_lanes) break;
        keccak_f1600_state(a);
    }
}
```