This optimization improves the kernel by **merging the SHA3-256 and SHAKE128 fast paths**, which reduces the number of unrolled `ROUND` macro expansions from 72 down to 48. Because the `ROUND` macro expands to over 100 operations, reducing the total loop count significantly shrinks the compiled kernel size. This decreases instruction cache pressure, helps the compiler's register allocator avoid spilling, and reduces kernel compilation/launch latency, which is especially beneficial for the smaller `B16K` batches. The uniform conditionals evaluating `rate_bytes` map to single scalar instructions, avoiding any branch divergence.

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64u - (k))))

#define ROUND(RC) do { \
    ulong C0 = A0 ^ A5 ^ A10 ^ A15 ^ A20; \
    ulong C1 = A1 ^ A6 ^ A11 ^ A16 ^ A21; \
    ulong C2 = A2 ^ A7 ^ A12 ^ A17 ^ A22; \
    ulong C3 = A3 ^ A8 ^ A13 ^ A18 ^ A23; \
    ulong C4 = A4 ^ A9 ^ A14 ^ A19 ^ A24; \
    \
    ulong D0 = C4 ^ ROTL64(C1, 1u); \
    ulong D1 = C0 ^ ROTL64(C2, 1u); \
    ulong D2 = C1 ^ ROTL64(C3, 1u); \
    ulong D3 = C2 ^ ROTL64(C4, 1u); \
    ulong D4 = C3 ^ ROTL64(C0, 1u); \
    \
    ulong B0  = A0 ^ D0; \
    ulong B1  = ROTL64(A1 ^ D1, 1u); \
    ulong B2  = ROTL64(A2 ^ D2, 62u); \
    ulong B3  = ROTL64(A3 ^ D3, 28u); \
    ulong B4  = ROTL64(A4 ^ D4, 27u); \
    ulong B5  = ROTL64(A5 ^ D0, 36u); \
    ulong B6  = ROTL64(A6 ^ D1, 44u); \
    ulong B7  = ROTL64(A7 ^ D2, 6u); \
    ulong B8  = ROTL64(A8 ^ D3, 55u); \
    ulong B9  = ROTL64(A9 ^ D4, 20u); \
    ulong B10 = ROTL64(A10 ^ D0, 3u); \
    ulong B11 = ROTL64(A11 ^ D1, 10u); \
    ulong B12 = ROTL64(A12 ^ D2, 43u); \
    ulong B13 = ROTL64(A13 ^ D3, 25u); \
    ulong B14 = ROTL64(A14 ^ D4, 39u); \
    ulong B15 = ROTL64(A15 ^ D0, 41u); \
    ulong B16 = ROTL64(A16 ^ D1, 45u); \
    ulong B17 = ROTL64(A17 ^ D2, 15u); \
    ulong B18 = ROTL64(A18 ^ D3, 21u); \
    ulong B19 = ROTL64(A19 ^ D4, 8u); \
    ulong B20 = ROTL64(A20 ^ D0, 18u); \
    ulong B21 = ROTL64(A21 ^ D1, 2u); \
    ulong B22 = ROTL64(A22 ^ D2, 61u); \
    ulong B23 = ROTL64(A23 ^ D3, 56u); \
    ulong B24 = ROTL64(A24 ^ D4, 14u); \
    \
    A0 = B0 ^ (B12 & ~B6) ^ RC; \
    A1 = B6 ^ (B18 & ~B12); \
    A2 = B12 ^ (B24 & ~B18); \
    A3 = B18 ^ (B0 & ~B24); \
    A4 = B24 ^ (B6 & ~B0); \
    \
    A5 = B3 ^ (B10 & ~B9); \
    A6 = B9 ^ (B16 & ~B10); \
    A7 = B10 ^ (B22 & ~B16); \
    A8 = B16 ^ (B3 & ~B22); \
    A9 = B22 ^ (B9 & ~B3); \
    \
    A10 = B1 ^ (B13 & ~B7); \
    A11 = B7 ^ (B19 & ~B13); \
    A12 = B13 ^ (B20 & ~B19); \
    A13 = B19 ^ (B1 & ~B20); \
    A14 = B20 ^ (B7 & ~B1); \
    \
    A15 = B4 ^ (B11 & ~B5); \
    A16 = B5 ^ (B17 & ~B11); \
    A17 = B11 ^ (B23 & ~B17); \
    A18 = B17 ^ (B4 & ~B23); \
    A19 = B23 ^ (B5 & ~B4); \
    \
    A20 = B2 ^ (B14 & ~B8); \
    A21 = B8 ^ (B15 & ~B14); \
    A22 = B14 ^ (B21 & ~B15); \
    A23 = B15 ^ (B2 & ~B21); \
    A24 = B21 ^ (B8 & ~B2); \
} while(0)

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

    // Combined fast path for SHA3-256 and SHAKE128
    if (msg_bytes == 32 && ((rate_bytes == 136 && out_bytes == 32) || (rate_bytes == 168 && out_bytes == 256))) {
        ulong4 in_val = ((device const ulong4 *)in_data)[idx];
        ulong A0 = in_val[0];
        ulong A1 = in_val[1];
        ulong A2 = in_val[2];
        ulong A3 = in_val[3];
        ulong A4 = (ulong)(domain & 0xFFu);
        
        ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
        ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
        ulong A15 = 0;
        ulong A16 = (rate_bytes == 136) ? 0x8000000000000000ul : 0ul;
        ulong A17 = 0, A18 = 0, A19 = 0;
        ulong A20 = (rate_bytes == 168) ? 0x8000000000000000ul : 0ul;
        ulong A21 = 0, A22 = 0, A23 = 0, A24 = 0;

        #pragma unroll(24)
        for (uint r = 0u; r < 24u; ++r) {
            ROUND(KECCAK_RC[r]);
        }

        if (rate_bytes == 136) { // SHA3-256
            ((device ulong4 *)out_data)[idx] = ulong4(A0, A1, A2, A3);
            return;
        } else { // SHAKE128
            uint out_base = idx * 32;
            device ulong4 *ptr = (device ulong4 *)(out_data + out_base);
            ptr[0] = ulong4(A0, A1, A2, A3);
            ptr[1] = ulong4(A4, A5, A6, A7);
            ptr[2] = ulong4(A8, A9, A10, A11);
            ptr[3] = ulong4(A12, A13, A14, A15);
            ptr[4] = ulong4(A16, A17, A18, A19);
            ulong saved_A20 = A20;

            #pragma unroll(24)
            for (uint r = 0u; r < 24u; ++r) {
                ROUND(KECCAK_RC[r]);
            }

            device ulong4 *ptr2 = (device ulong4 *)(out_data + out_base + 20);
            ptr2[0] = ulong4(saved_A20, A0, A1, A2);
            ptr2[1] = ulong4(A3, A4, A5, A6);
            ptr2[2] = ulong4(A7, A8, A9, A10);
            return;
        }
    }

    // Fallback path
    ulong A0 = 0, A1 = 0, A2 = 0, A3 = 0, A4 = 0;
    ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
    ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
    ulong A15 = 0, A16 = 0, A17 = 0, A18 = 0, A19 = 0;
    ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;

    uint msg_lanes  = msg_bytes >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes >> 3;
    
    uint in_base = idx * msg_lanes;
    switch (msg_lanes) {
        case 24: A23 = in_data[in_base + 23]; [[fallthrough]];
        case 23: A22 = in_data[in_base + 22]; [[fallthrough]];
        case 22: A21 = in_data[in_base + 21]; [[fallthrough]];
        case 21: A20 = in_data[in_base + 20]; [[fallthrough]];
        case 20: A19 = in_data[in_base + 19]; [[fallthrough]];
        case 19: A18 = in_data[in_base + 18]; [[fallthrough]];
        case 18: A17 = in_data[in_base + 17]; [[fallthrough]];
        case 17: A16 = in_data[in_base + 16]; [[fallthrough]];
        case 16: A15 = in_data[in_base + 15]; [[fallthrough]];
        case 15: A14 = in_data[in_base + 14]; [[fallthrough]];
        case 14: A13 = in_data[in_base + 13]; [[fallthrough]];
        case 13: A12 = in_data[in_base + 12]; [[fallthrough]];
        case 12: A11 = in_data[in_base + 11]; [[fallthrough]];
        case 11: A10 = in_data[in_base + 10]; [[fallthrough]];
        case 10: A9  = in_data[in_base + 9];  [[fallthrough]];
        case 9:  A8  = in_data[in_base + 8];  [[fallthrough]];
        case 8:  A7  = in_data[in_base + 7];  [[fallthrough]];
        case 7:  A6  = in_data[in_base + 6];  [[fallthrough]];
        case 6:  A5  = in_data[in_base + 5];  [[fallthrough]];
        case 5:  A4  = in_data[in_base + 4];  [[fallthrough]];
        case 4:  A3  = in_data[in_base + 3];  [[fallthrough]];
        case 3:  A2  = in_data[in_base + 2];  [[fallthrough]];
        case 2:  A1  = in_data[in_base + 1];  [[fallthrough]];
        case 1:  A0  = in_data[in_base + 0];  [[fallthrough]];
        case 0: break;
    }

    switch (msg_lanes) {
        case 0:  A0  ^= (ulong)(domain & 0xFFu); break;
        case 1:  A1  ^= (ulong)(domain & 0xFFu); break;
        case 2:  A2  ^= (ulong)(domain & 0xFFu); break;
        case 3:  A3  ^= (ulong)(domain & 0xFFu); break;
        case 4:  A4  ^= (ulong)(domain & 0xFFu); break;
        case 5:  A5  ^= (ulong)(domain & 0xFFu); break;
        case 6:  A6  ^= (ulong)(domain & 0xFFu); break;
        case 7:  A7  ^= (ulong)(domain & 0xFFu); break;
        case 8:  A8  ^= (ulong)(domain & 0xFFu); break;
        case 9:  A9  ^= (ulong)(domain & 0xFFu); break;
        case 10: A10 ^= (ulong)(domain & 0xFFu); break;
        case 11: A11 ^= (ulong)(domain & 0xFFu); break;
        case 12: A12 ^= (ulong)(domain & 0xFFu); break;
        case 13: A13 ^= (ulong)(domain & 0xFFu); break;
        case 14: A14 ^= (ulong)(domain & 0xFFu); break;
        case 15: A15 ^= (ulong)(domain & 0xFFu); break;
        case 16: A16 ^= (ulong)(domain & 0xFFu); break;
        case 17: A17 ^= (ulong)(domain & 0xFFu); break;
        case 18: A18 ^= (ulong)(domain & 0xFFu); break;
        case 19: A19 ^= (ulong)(domain & 0xFFu); break;
        case 20: A20 ^= (ulong)(domain & 0xFFu); break;
        case 21: A21 ^= (ulong)(domain & 0xFFu); break;
        case 22: A22 ^= (ulong)(domain & 0xFFu); break;
        case 23: A23 ^= (ulong)(domain & 0xFFu); break;
        case 24: A24 ^= (ulong)(domain & 0xFFu); break;
    }

    switch (rate_lanes - 1u) {
        case 0:  A0  ^= 0x8000000000000000ul; break;
        case 1:  A1  ^= 0x8000000000000000ul; break;
        case 2:  A2  ^= 0x8000000000000000ul; break;
        case 3:  A3  ^= 0x8000000000000000ul; break;
        case 4:  A4  ^= 0x8000000000000000ul; break;
        case 5:  A5  ^= 0x8000000000000000ul; break;
        case 6:  A6  ^= 0x8000000000000000ul; break;
        case 7:  A7  ^= 0x8000000000000000ul; break;
        case 8:  A8  ^= 0x8000000000000000ul; break;
        case 9:  A9  ^= 0x8000000000000000ul; break;
        case 10: A10 ^= 0x8000000000000000ul; break;
        case 11: A11 ^= 0x8000000000000000ul; break;
        case 12: A12 ^= 0x8000000000000000ul; break;
        case 13: A13 ^= 0x8000000000000000ul; break;
        case 14: A14 ^= 0x8000000000000000ul; break;
        case 15: A15 ^= 0x8000000000000000ul; break;
        case 16: A16 ^= 0x8000000000000000ul; break;
        case 17: A17 ^= 0x8000000000000000ul; break;
        case 18: A18 ^= 0x8000000000000000ul; break;
        case 19: A19 ^= 0x8000000000000000ul; break;
        case 20: A20 ^= 0x8000000000000000ul; break;
        case 21: A21 ^= 0x8000000000000000ul; break;
        case 22: A22 ^= 0x8000000000000000ul; break;
        case 23: A23 ^= 0x8000000000000000ul; break;
        case 24: A24 ^= 0x8000000000000000ul; break;
    }

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        #pragma unroll(24)
        for (uint r = 0u; r < 24u; ++r) {
            ROUND(KECCAK_RC[r]);
        }

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        uint out_offset = out_base + written;
        
        switch (take) {
            case 24: out_data[out_offset + 23] = A23; [[fallthrough]];
            case 23: out_data[out_offset + 22] = A22; [[fallthrough]];
            case 22: out_data[out_offset + 21] = A21; [[fallthrough]];
            case 21: out_data[out_offset + 20] = A20; [[fallthrough]];
            case 20: out_data[out_offset + 19] = A19; [[fallthrough]];
            case 19: out_data[out_offset + 18] = A18; [[fallthrough]];
            case 18: out_data[out_offset + 17] = A17; [[fallthrough]];
            case 17: out_data[out_offset + 16] = A16; [[fallthrough]];
            case 16: out_data[out_offset + 15] = A15; [[fallthrough]];
            case 15: out_data[out_offset + 14] = A14; [[fallthrough]];
            case 14: out_data[out_offset + 13] = A13; [[fallthrough]];
            case 13: out_data[out_offset + 12] = A12; [[fallthrough]];
            case 12: out_data[out_offset + 11] = A11; [[fallthrough]];
            case 11: out_data[out_offset + 10] = A10; [[fallthrough]];
            case 10: out_data[out_offset + 9]  = A9;  [[fallthrough]];
            case 9:  out_data[out_offset + 8]  = A8;  [[fallthrough]];
            case 8:  out_data[out_offset + 7]  = A7;  [[fallthrough]];
            case 7:  out_data[out_offset + 6]  = A6;  [[fallthrough]];
            case 6:  out_data[out_offset + 5]  = A5;  [[fallthrough]];
            case 5:  out_data[out_offset + 4]  = A4;  [[fallthrough]];
            case 4:  out_data[out_offset + 3]  = A3;  [[fallthrough]];
            case 3:  out_data[out_offset + 2]  = A2;  [[fallthrough]];
            case 2:  out_data[out_offset + 1]  = A1;  [[fallthrough]];
            case 1:  out_data[out_offset + 0]  = A0;  [[fallthrough]];
            case 0: break;
        }

        written += take;
        if (written >= out_lanes) break;
    }
}
```