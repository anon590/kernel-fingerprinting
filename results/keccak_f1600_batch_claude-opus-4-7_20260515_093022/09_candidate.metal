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

// AND-NOT: a & ~b  -> single bic op on most archs.
#define ANDN(a,b) ((a) & ~(b))

#define KECCAK_ROUND(RCV)                                              \
    {                                                                  \
        ulong C0 = s00 ^ s01 ^ s02 ^ s03 ^ s04;                        \
        ulong C1 = s10 ^ s11 ^ s12 ^ s13 ^ s14;                        \
        ulong C2 = s20 ^ s21 ^ s22 ^ s23 ^ s24;                        \
        ulong C3 = s30 ^ s31 ^ s32 ^ s33 ^ s34;                        \
        ulong C4 = s40 ^ s41 ^ s42 ^ s43 ^ s44;                        \
        ulong D0 = C4 ^ ROTL64(C1, 1);                                 \
        ulong D1 = C0 ^ ROTL64(C2, 1);                                 \
        ulong D2 = C1 ^ ROTL64(C3, 1);                                 \
        ulong D3 = C2 ^ ROTL64(C4, 1);                                 \
        ulong D4 = C3 ^ ROTL64(C0, 1);                                 \
        ulong b00 = (s00 ^ D0);                                        \
        ulong b02 = ROTL64(s10 ^ D1,  1);                              \
        ulong b04 = ROTL64(s20 ^ D2, 62);                              \
        ulong b01 = ROTL64(s30 ^ D3, 28);                              \
        ulong b03 = ROTL64(s40 ^ D4, 27);                              \
        ulong b13 = ROTL64(s01 ^ D0, 36);                              \
        ulong b10 = ROTL64(s11 ^ D1, 44);                              \
        ulong b12 = ROTL64(s21 ^ D2,  6);                              \
        ulong b14 = ROTL64(s31 ^ D3, 55);                              \
        ulong b11 = ROTL64(s41 ^ D4, 20);                              \
        ulong b21 = ROTL64(s02 ^ D0,  3);                              \
        ulong b23 = ROTL64(s12 ^ D1, 10);                              \
        ulong b20 = ROTL64(s22 ^ D2, 43);                              \
        ulong b22 = ROTL64(s32 ^ D3, 25);                              \
        ulong b24 = ROTL64(s42 ^ D4, 39);                              \
        ulong b34 = ROTL64(s03 ^ D0, 41);                              \
        ulong b31 = ROTL64(s13 ^ D1, 45);                              \
        ulong b33 = ROTL64(s23 ^ D2, 15);                              \
        ulong b30 = ROTL64(s33 ^ D3, 21);                              \
        ulong b32 = ROTL64(s43 ^ D4,  8);                              \
        ulong b42 = ROTL64(s04 ^ D0, 18);                              \
        ulong b44 = ROTL64(s14 ^ D1,  2);                              \
        ulong b41 = ROTL64(s24 ^ D2, 61);                              \
        ulong b43 = ROTL64(s34 ^ D3, 56);                              \
        ulong b40 = ROTL64(s44 ^ D4, 14);                              \
        s00 = b00 ^ ANDN(b20, b10) ^ (RCV);                            \
        s10 = b10 ^ ANDN(b30, b20);                                    \
        s20 = b20 ^ ANDN(b40, b30);                                    \
        s30 = b30 ^ ANDN(b00, b40);                                    \
        s40 = b40 ^ ANDN(b10, b00);                                    \
        s01 = b01 ^ ANDN(b21, b11);                                    \
        s11 = b11 ^ ANDN(b31, b21);                                    \
        s21 = b21 ^ ANDN(b41, b31);                                    \
        s31 = b31 ^ ANDN(b01, b41);                                    \
        s41 = b41 ^ ANDN(b11, b01);                                    \
        s02 = b02 ^ ANDN(b22, b12);                                    \
        s12 = b12 ^ ANDN(b32, b22);                                    \
        s22 = b22 ^ ANDN(b42, b32);                                    \
        s32 = b32 ^ ANDN(b02, b42);                                    \
        s42 = b42 ^ ANDN(b12, b02);                                    \
        s03 = b03 ^ ANDN(b23, b13);                                    \
        s13 = b13 ^ ANDN(b33, b23);                                    \
        s23 = b23 ^ ANDN(b43, b33);                                    \
        s33 = b33 ^ ANDN(b03, b43);                                    \
        s43 = b43 ^ ANDN(b13, b03);                                    \
        s04 = b04 ^ ANDN(b24, b14);                                    \
        s14 = b14 ^ ANDN(b34, b24);                                    \
        s24 = b24 ^ ANDN(b44, b34);                                    \
        s34 = b34 ^ ANDN(b04, b44);                                    \
        s44 = b44 ^ ANDN(b14, b04);                                    \
    }

#define KECCAK_F1600()                  \
    KECCAK_ROUND(KECCAK_RC[ 0]);        \
    KECCAK_ROUND(KECCAK_RC[ 1]);        \
    KECCAK_ROUND(KECCAK_RC[ 2]);        \
    KECCAK_ROUND(KECCAK_RC[ 3]);        \
    KECCAK_ROUND(KECCAK_RC[ 4]);        \
    KECCAK_ROUND(KECCAK_RC[ 5]);        \
    KECCAK_ROUND(KECCAK_RC[ 6]);        \
    KECCAK_ROUND(KECCAK_RC[ 7]);        \
    KECCAK_ROUND(KECCAK_RC[ 8]);        \
    KECCAK_ROUND(KECCAK_RC[ 9]);        \
    KECCAK_ROUND(KECCAK_RC[10]);        \
    KECCAK_ROUND(KECCAK_RC[11]);        \
    KECCAK_ROUND(KECCAK_RC[12]);        \
    KECCAK_ROUND(KECCAK_RC[13]);        \
    KECCAK_ROUND(KECCAK_RC[14]);        \
    KECCAK_ROUND(KECCAK_RC[15]);        \
    KECCAK_ROUND(KECCAK_RC[16]);        \
    KECCAK_ROUND(KECCAK_RC[17]);        \
    KECCAK_ROUND(KECCAK_RC[18]);        \
    KECCAK_ROUND(KECCAK_RC[19]);        \
    KECCAK_ROUND(KECCAK_RC[20]);        \
    KECCAK_ROUND(KECCAK_RC[21]);        \
    KECCAK_ROUND(KECCAK_RC[22]);        \
    KECCAK_ROUND(KECCAK_RC[23]);

kernel void keccak_f1600_batch(
    device const ulong *in_data    [[buffer(0)]],
    device       ulong *out_data   [[buffer(1)]],
    constant uint      &batch      [[buffer(2)]],
    constant uint      &msg_bytes  [[buffer(3)]],
    constant uint      &rate_bytes [[buffer(4)]],
    constant uint      &out_bytes  [[buffer(5)]],
    constant uint      &domain     [[buffer(6)]],
    uint idx     [[thread_position_in_grid]],
    uint tid_tg  [[thread_position_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong s00=0, s10=0, s20=0, s30=0, s40=0;
    ulong s01=0, s11=0, s21=0, s31=0, s41=0;
    ulong s02=0, s12=0, s22=0, s32=0, s42=0;
    ulong s03=0, s13=0, s23=0, s33=0, s43=0;
    ulong s04=0, s14=0, s24=0, s34=0, s44=0;

    // Fast path: SHA3-256 with 32-byte input.
    if (msg_lanes == 4u && rate_lanes == 17u && out_lanes == 4u) {
        threadgroup ulong tg_buf[64 * 4];

        uint tg_base_in = tg_id * tg_size * 4u;
        uint valid_threads = min(tg_size, batch - tg_id * tg_size);
        uint valid_in = valid_threads * 4u;

        for (uint k = 0u; k < 4u; ++k) {
            uint off = k * tg_size + tid_tg;
            if (off < valid_in) {
                tg_buf[off] = in_data[tg_base_in + off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx < batch) {
            uint lb = tid_tg * 4u;
            s00 = tg_buf[lb + 0u];
            s10 = tg_buf[lb + 1u];
            s20 = tg_buf[lb + 2u];
            s30 = tg_buf[lb + 3u];
            s40 = (ulong)(domain & 0xFFu);
            s13 = 0x8000000000000000ul;

            KECCAK_F1600();
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (idx < batch) {
            uint lb = tid_tg * 4u;
            tg_buf[lb + 0u] = s00;
            tg_buf[lb + 1u] = s10;
            tg_buf[lb + 2u] = s20;
            tg_buf[lb + 3u] = s30;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint tg_base_out = tg_id * tg_size * 4u;
        uint valid_out = valid_threads * 4u;
        for (uint k = 0u; k < 4u; ++k) {
            uint off = k * tg_size + tid_tg;
            if (off < valid_out) {
                out_data[tg_base_out + off] = tg_buf[off];
            }
        }
        return;
    }

    if (idx >= batch) return;

    // General path
    {
        ulong tmp[25];
        for (uint i = 0u; i < 25u; ++i) tmp[i] = 0ul;
        uint in_base = idx * msg_lanes;
        for (uint i = 0u; i < msg_lanes; ++i) tmp[i] = in_data[in_base + i];
        tmp[msg_lanes]       ^= (ulong)(domain & 0xFFu);
        tmp[rate_lanes - 1u] ^= 0x8000000000000000ul;

        s00 = tmp[ 0]; s10 = tmp[ 1]; s20 = tmp[ 2]; s30 = tmp[ 3]; s40 = tmp[ 4];
        s01 = tmp[ 5]; s11 = tmp[ 6]; s21 = tmp[ 7]; s31 = tmp[ 8]; s41 = tmp[ 9];
        s02 = tmp[10]; s12 = tmp[11]; s22 = tmp[12]; s32 = tmp[13]; s42 = tmp[14];
        s03 = tmp[15]; s13 = tmp[16]; s23 = tmp[17]; s33 = tmp[18]; s43 = tmp[19];
        s04 = tmp[20]; s14 = tmp[21]; s24 = tmp[22]; s34 = tmp[23]; s44 = tmp[24];
    }

    uint out_base = idx * out_lanes;
    uint written  = 0u;
    for (;;) {
        KECCAK_F1600();

        ulong lanes[25] = {
            s00, s10, s20, s30, s40,
            s01, s11, s21, s31, s41,
            s02, s12, s22, s32, s42,
            s03, s13, s23, s33, s43,
            s04, s14, s24, s34, s44
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