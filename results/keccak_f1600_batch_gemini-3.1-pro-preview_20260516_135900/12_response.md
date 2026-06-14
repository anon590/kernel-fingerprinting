To push the performance of the batch Keccak-f[1600] permutation, we apply three primary optimizations:

1. **Ping-Pong Register Updates (A $\to$ E $\to$ A):** The previous version computed temporaries `B0..B24` and immediately wrote them back to `A0..A24`. This introduces unnecessary move/rename overhead. By explicitly unwinding the loop to operate on two alternating register banks (`A` and `E`), we completely eliminate register moves, saving 600 instructions per instance.
2. **Precomputed Round 0:** For the `SHA3-256` and `SHAKE128` fast paths, the input state contains 20 constant zero-lanes. We explicitly resolve Round 0 at compile time, eliminating ~15 bitwise XORs and directly loading the output into `E` to initiate the ping-pong loop.
3. **Strictly-Aligned Vector Memory Access:** Message strings and outputs are guaranteed to be 32-byte aligned (`idx * 32`). We cast `in_data` and `out_data` to `device ulong4*` to leverage full 256-bit memory transactions, maximizing memory bandwidth without cache-line fragmentation.

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

#define RND_CORE(in, out, RC) do { \
    ulong C0 = in##0 ^ in##5 ^ in##10 ^ in##15 ^ in##20; \
    ulong C1 = in##1 ^ in##6 ^ in##11 ^ in##16 ^ in##21; \
    ulong C2 = in##2 ^ in##7 ^ in##12 ^ in##17 ^ in##22; \
    ulong C3 = in##3 ^ in##8 ^ in##13 ^ in##18 ^ in##23; \
    ulong C4 = in##4 ^ in##9 ^ in##14 ^ in##19 ^ in##24; \
    \
    ulong D0 = C4 ^ ROTL64(C1, 1u); \
    ulong D1 = C0 ^ ROTL64(C2, 1u); \
    ulong D2 = C1 ^ ROTL64(C3, 1u); \
    ulong D3 = C2 ^ ROTL64(C4, 1u); \
    ulong D4 = C3 ^ ROTL64(C0, 1u); \
    \
    ulong b0  = in##0 ^ D0; \
    ulong b1  = ROTL64(in##1 ^ D1, 1u); \
    ulong b2  = ROTL64(in##2 ^ D2, 62u); \
    ulong b3  = ROTL64(in##3 ^ D3, 28u); \
    ulong b4  = ROTL64(in##4 ^ D4, 27u); \
    ulong b5  = ROTL64(in##5 ^ D0, 36u); \
    ulong b6  = ROTL64(in##6 ^ D1, 44u); \
    ulong b7  = ROTL64(in##7 ^ D2, 6u); \
    ulong b8  = ROTL64(in##8 ^ D3, 55u); \
    ulong b9  = ROTL64(in##9 ^ D4, 20u); \
    ulong b10 = ROTL64(in##10 ^ D0, 3u); \
    ulong b11 = ROTL64(in##11 ^ D1, 10u); \
    ulong b12 = ROTL64(in##12 ^ D2, 43u); \
    ulong b13 = ROTL64(in##13 ^ D3, 25u); \
    ulong b14 = ROTL64(in##14 ^ D4, 39u); \
    ulong b15 = ROTL64(in##15 ^ D0, 41u); \
    ulong b16 = ROTL64(in##16 ^ D1, 45u); \
    ulong b17 = ROTL64(in##17 ^ D2, 15u); \
    ulong b18 = ROTL64(in##18 ^ D3, 21u); \
    ulong b19 = ROTL64(in##19 ^ D4, 8u); \
    ulong b20 = ROTL64(in##20 ^ D0, 18u); \
    ulong b21 = ROTL64(in##21 ^ D1, 2u); \
    ulong b22 = ROTL64(in##22 ^ D2, 61u); \
    ulong b23 = ROTL64(in##23 ^ D3, 56u); \
    ulong b24 = ROTL64(in##24 ^ D4, 14u); \
    \
    out##0 = b0 ^ (b12 & ~b6) ^ RC; \
    out##1 = b6 ^ (b18 & ~b12); \
    out##2 = b12 ^ (b24 & ~b18); \
    out##3 = b18 ^ (b0 & ~b24); \
    out##4 = b24 ^ (b6 & ~b0); \
    \
    out##5 = b3 ^ (b10 & ~b9); \
    out##6 = b9 ^ (b16 & ~b10); \
    out##7 = b10 ^ (b22 & ~b16); \
    out##8 = b16 ^ (b3 & ~b22); \
    out##9 = b22 ^ (b9 & ~b3); \
    \
    out##10 = b1 ^ (b13 & ~b7); \
    out##11 = b7 ^ (b19 & ~b13); \
    out##12 = b13 ^ (b20 & ~b19); \
    out##13 = b19 ^ (b1 & ~b20); \
    out##14 = b20 ^ (b7 & ~b1); \
    \
    out##15 = b4 ^ (b11 & ~b5); \
    out##16 = b5 ^ (b17 & ~b11); \
    out##17 = b11 ^ (b23 & ~b17); \
    out##18 = b17 ^ (b4 & ~b23); \
    out##19 = b23 ^ (b5 & ~b4); \
    \
    out##20 = b2 ^ (b14 & ~b8); \
    out##21 = b8 ^ (b15 & ~b14); \
    out##22 = b14 ^ (b21 & ~b15); \
    out##23 = b15 ^ (b2 & ~b21); \
    out##24 = b21 ^ (b8 & ~b2); \
} while(0)

#define PRECOMP_OUT_E(RC) do { \
    E0 = b0 ^ (b12 & ~b6) ^ RC; \
    E1 = b6 ^ (b18 & ~b12); \
    E2 = b12 ^ (b24 & ~b18); \
    E3 = b18 ^ (b0 & ~b24); \
    E4 = b24 ^ (b6 & ~b0); \
    E5 = b3 ^ (b10 & ~b9); \
    E6 = b9 ^ (b16 & ~b10); \
    E7 = b10 ^ (b22 & ~b16); \
    E8 = b16 ^ (b3 & ~b22); \
    E9 = b22 ^ (b9 & ~b3); \
    E10 = b1 ^ (b13 & ~b7); \
    E11 = b7 ^ (b19 & ~b13); \
    E12 = b13 ^ (b20 & ~b19); \
    E13 = b19 ^ (b1 & ~b20); \
    E14 = b20 ^ (b7 & ~b1); \
    E15 = b4 ^ (b11 & ~b5); \
    E16 = b5 ^ (b17 & ~b11); \
    E17 = b11 ^ (b23 & ~b17); \
    E18 = b17 ^ (b4 & ~b23); \
    E19 = b23 ^ (b5 & ~b4); \
    E20 = b2 ^ (b14 & ~b8); \
    E21 = b8 ^ (b15 & ~b14); \
    E22 = b14 ^ (b21 & ~b15); \
    E23 = b15 ^ (b2 & ~b21); \
    E24 = b21 ^ (b8 & ~b2); \
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

    if (msg_bytes == 32 && rate_bytes == 136 && out_bytes == 32) {
        ulong4 in_val = ((device const ulong4 *)in_data)[idx];
        ulong A0 = in_val[0];
        ulong A1 = in_val[1];
        ulong A2 = in_val[2];
        ulong A3 = in_val[3];
        ulong A4 = (ulong)(domain & 0xFFu);
        ulong A5, A6, A7, A8, A9, A10, A11, A12, A13, A14;
        ulong A15, A16, A17, A18, A19, A20, A21, A22, A23, A24;
        ulong E0, E1, E2, E3, E4, E5, E6, E7, E8, E9;
        ulong E10, E11, E12, E13, E14, E15, E16, E17, E18, E19;
        ulong E20, E21, E22, E23, E24;

        {
            ulong D0 = A4 ^ ROTL64(A1 ^ 0x8000000000000000ul, 1u);
            ulong D1 = A0 ^ ROTL64(A2, 1u);
            ulong D2 = (A1 ^ 0x8000000000000000ul) ^ ROTL64(A3, 1u);
            ulong D3 = A2 ^ ROTL64(A4, 1u);
            ulong D4 = A3 ^ ROTL64(A0, 1u);
            
            ulong b0  = A0 ^ D0;
            ulong b1  = ROTL64(A1 ^ D1, 1u);
            ulong b2  = ROTL64(A2 ^ D2, 62u);
            ulong b3  = ROTL64(A3 ^ D3, 28u);
            ulong b4  = ROTL64(A4 ^ D4, 27u);
            ulong b5  = ROTL64(D0, 36u);
            ulong b6  = ROTL64(D1, 44u);
            ulong b7  = ROTL64(D2, 6u);
            ulong b8  = ROTL64(D3, 55u);
            ulong b9  = ROTL64(D4, 20u);
            ulong b10 = ROTL64(D0, 3u);
            ulong b11 = ROTL64(D1, 10u);
            ulong b12 = ROTL64(D2, 43u);
            ulong b13 = ROTL64(D3, 25u);
            ulong b14 = ROTL64(D4, 39u);
            ulong b15 = ROTL64(D0, 41u);
            ulong b16 = ROTL64(0x8000000000000000ul ^ D1, 45u);
            ulong b17 = ROTL64(D2, 15u);
            ulong b18 = ROTL64(D3, 21u);
            ulong b19 = ROTL64(D4, 8u);
            ulong b20 = ROTL64(D0, 18u);
            ulong b21 = ROTL64(D1, 2u);
            ulong b22 = ROTL64(D2, 61u);
            ulong b23 = ROTL64(D3, 56u);
            ulong b24 = ROTL64(D4, 14u);
            
            PRECOMP_OUT_E(KECCAK_RC[0]);
        }

        #pragma unroll(11)
        for (uint r = 1u; r < 23u; r += 2u) {
            RND_CORE(E, A, KECCAK_RC[r]);
            RND_CORE(A, E, KECCAK_RC[r+1]);
        }
        RND_CORE(E, A, KECCAK_RC[23]);

        ((device ulong4 *)out_data)[idx] = ulong4(A0, A1, A2, A3);
        return;
    }

    if (msg_bytes == 32 && rate_bytes == 168 && out_bytes == 256) {
        ulong4 in_val = ((device const ulong4 *)in_data)[idx];
        ulong A0 = in_val[0];
        ulong A1 = in_val[1];
        ulong A2 = in_val[2];
        ulong A3 = in_val[3];
        ulong A4 = (ulong)(domain & 0xFFu);
        ulong A5, A6, A7, A8, A9, A10, A11, A12, A13, A14;
        ulong A15, A16, A17, A18, A19, A20, A21, A22, A23, A24;
        ulong E0, E1, E2, E3, E4, E5, E6, E7, E8, E9;
        ulong E10, E11, E12, E13, E14, E15, E16, E17, E18, E19;
        ulong E20, E21, E22, E23, E24;

        {
            ulong D0 = A4 ^ ROTL64(A1, 1u);
            ulong D1 = (A0 ^ 0x8000000000000000ul) ^ ROTL64(A2, 1u);
            ulong D2 = A1 ^ ROTL64(A3, 1u);
            ulong D3 = A2 ^ ROTL64(A4, 1u);
            ulong D4 = A3 ^ ROTL64(A0 ^ 0x8000000000000000ul, 1u);
            
            ulong b0  = A0 ^ D0;
            ulong b1  = ROTL64(A1 ^ D1, 1u);
            ulong b2  = ROTL64(A2 ^ D2, 62u);
            ulong b3  = ROTL64(A3 ^ D3, 28u);
            ulong b4  = ROTL64(A4 ^ D4, 27u);
            ulong b5  = ROTL64(D0, 36u);
            ulong b6  = ROTL64(D1, 44u);
            ulong b7  = ROTL64(D2, 6u);
            ulong b8  = ROTL64(D3, 55u);
            ulong b9  = ROTL64(D4, 20u);
            ulong b10 = ROTL64(D0, 3u);
            ulong b11 = ROTL64(D1, 10u);
            ulong b12 = ROTL64(D2, 43u);
            ulong b13 = ROTL64(D3, 25u);
            ulong b14 = ROTL64(D4, 39u);
            ulong b15 = ROTL64(D0, 41u);
            ulong b16 = ROTL64(D1, 45u);
            ulong b17 = ROTL64(D2, 15u);
            ulong b18 = ROTL64(D3, 21u);
            ulong b19 = ROTL64(D4, 8u);
            ulong b20 = ROTL64(0x8000000000000000ul ^ D0, 18u);
            ulong b21 = ROTL64(D1, 2u);
            ulong b22 = ROTL64(D2, 61u);
            ulong b23 = ROTL64(D3, 56u);
            ulong b24 = ROTL64(D4, 14u);
            
            PRECOMP_OUT_E(KECCAK_RC[0]);
        }

        #pragma unroll(11)
        for (uint r = 1u; r < 23u; r += 2u) {
            RND_CORE(E, A, KECCAK_RC[r]);
            RND_CORE(A, E, KECCAK_RC[r+1]);
        }
        RND_CORE(E, A, KECCAK_RC[23]);

        uint out_base = idx * 32;
        device ulong4 *ptr = (device ulong4 *)(out_data + out_base);
        ptr[0] = ulong4(A0, A1, A2, A3);
        ptr[1] = ulong4(A4, A5, A6, A7);
        ptr[2] = ulong4(A8, A9, A10, A11);
        ptr[3] = ulong4(A12, A13, A14, A15);
        ptr[4] = ulong4(A16, A17, A18, A19);
        ulong saved_A20 = A20;

        #pragma unroll(12)
        for (uint r = 0u; r < 24u; r += 2u) {
            RND_CORE(A, E, KECCAK_RC[r]);
            RND_CORE(E, A, KECCAK_RC[r+1]);
        }

        device ulong4 *ptr2 = (device ulong4 *)(out_data + out_base + 20);
        ptr2[0] = ulong4(saved_A20, A0, A1, A2);
        ptr2[1] = ulong4(A3, A4, A5, A6);
        ptr2[2] = ulong4(A7, A8, A9, A10);
        return;
    }

    ulong A0 = 0, A1 = 0, A2 = 0, A3 = 0, A4 = 0;
    ulong A5 = 0, A6 = 0, A7 = 0, A8 = 0, A9 = 0;
    ulong A10 = 0, A11 = 0, A12 = 0, A13 = 0, A14 = 0;
    ulong A15 = 0, A16 = 0, A17 = 0, A18 = 0, A19 = 0;
    ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;
    ulong E0, E1, E2, E3, E4, E5, E6, E7, E8, E9;
    ulong E10, E11, E12, E13, E14, E15, E16, E17, E18, E19;
    ulong E20, E21, E22, E23, E24;

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
        #pragma unroll(12)
        for (uint r = 0u; r < 24u; r += 2u) {
            RND_CORE(A, E, KECCAK_RC[r]);
            RND_CORE(E, A, KECCAK_RC[r+1]);
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