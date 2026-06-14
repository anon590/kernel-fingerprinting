#include <metal_stdlib>
using namespace metal;

inline ulong rotl(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    constexpr ulong RC[24] = {
        0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
        0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
        0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
        0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
        0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
        0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
        0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
        0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
    };

    #pragma unroll 24
    for (uint r = 0; r < 24; ++r) {
        // --- Theta ---
        ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D = C4 ^ rotl(C1, 1u);
        A[0] ^= D; A[5] ^= D; A[10] ^= D; A[15] ^= D; A[20] ^= D;

        D = C0 ^ rotl(C2, 1u);
        A[1] ^= D; A[6] ^= D; A[11] ^= D; A[16] ^= D; A[21] ^= D;

        D = C1 ^ rotl(C3, 1u);
        A[2] ^= D; A[7] ^= D; A[12] ^= D; A[17] ^= D; A[22] ^= D;

        D = C2 ^ rotl(C4, 1u);
        A[3] ^= D; A[8] ^= D; A[13] ^= D; A[18] ^= D; A[23] ^= D;

        D = C3 ^ rotl(C0, 1u);
        A[4] ^= D; A[9] ^= D; A[14] ^= D; A[19] ^= D; A[24] ^= D;

        // --- Rho and Pi ---
        ulong current = A[1];
        ulong next;
        next = A[10]; A[10] = rotl(current, 1u);  current = next;
        next = A[7];  A[7]  = rotl(current, 3u);  current = next;
        next = A[11]; A[11] = rotl(current, 6u);  current = next;
        next = A[17]; A[17] = rotl(current, 10u); current = next;
        next = A[18]; A[18] = rotl(current, 15u); current = next;
        next = A[3];  A[3]  = rotl(current, 21u); current = next;
        next = A[5];  A[5]  = rotl(current, 28u); current = next;
        next = A[16]; A[16] = rotl(current, 36u); current = next;
        next = A[8];  A[8]  = rotl(current, 45u); current = next;
        next = A[21]; A[21] = rotl(current, 55u); current = next;
        next = A[24]; A[24] = rotl(current, 2u);  current = next;
        next = A[4];  A[4]  = rotl(current, 14u); current = next;
        next = A[15]; A[15] = rotl(current, 27u); current = next;
        next = A[23]; A[23] = rotl(current, 41u); current = next;
        next = A[19]; A[19] = rotl(current, 56u); current = next;
        next = A[13]; A[13] = rotl(current, 8u);  current = next;
        next = A[12]; A[12] = rotl(current, 25u); current = next;
        next = A[2];  A[2]  = rotl(current, 43u); current = next;
        next = A[20]; A[20] = rotl(current, 62u); current = next;
        next = A[14]; A[14] = rotl(current, 18u); current = next;
        next = A[22]; A[22] = rotl(current, 39u); current = next;
        next = A[9];  A[9]  = rotl(current, 61u); current = next;
        next = A[6];  A[6]  = rotl(current, 20u); current = next;
                      A[1]  = rotl(current, 44u);

        // --- Chi ---
        ulong T0, T1, T2, T3, T4;

        T0 = A[0]; T1 = A[1]; T2 = A[2]; T3 = A[3]; T4 = A[4];
        A[0] = T0 ^ (~T1 & T2);
        A[1] = T1 ^ (~T2 & T3);
        A[2] = T2 ^ (~T3 & T4);
        A[3] = T3 ^ (~T4 & T0);
        A[4] = T4 ^ (~T0 & T1);

        T0 = A[5]; T1 = A[6]; T2 = A[7]; T3 = A[8]; T4 = A[9];
        A[5] = T0 ^ (~T1 & T2);
        A[6] = T1 ^ (~T2 & T3);
        A[7] = T2 ^ (~T3 & T4);
        A[8] = T3 ^ (~T4 & T0);
        A[9] = T4 ^ (~T0 & T1);

        T0 = A[10]; T1 = A[11]; T2 = A[12]; T3 = A[13]; T4 = A[14];
        A[10] = T0 ^ (~T1 & T2);
        A[11] = T1 ^ (~T2 & T3);
        A[12] = T2 ^ (~T3 & T4);
        A[13] = T3 ^ (~T4 & T0);
        A[14] = T4 ^ (~T0 & T1);

        T0 = A[15]; T1 = A[16]; T2 = A[17]; T3 = A[18]; T4 = A[19];
        A[15] = T0 ^ (~T1 & T2);
        A[16] = T1 ^ (~T2 & T3);
        A[17] = T2 ^ (~T3 & T4);
        A[18] = T3 ^ (~T4 & T0);
        A[19] = T4 ^ (~T0 & T1);

        T0 = A[20]; T1 = A[21]; T2 = A[22]; T3 = A[23]; T4 = A[24];
        A[20] = T0 ^ (~T1 & T2);
        A[21] = T1 ^ (~T2 & T3);
        A[22] = T2 ^ (~T3 & T4);
        A[23] = T3 ^ (~T4 & T0);
        A[24] = T4 ^ (~T0 & T1);

        // --- Iota ---
        A[0] ^= RC[r];
    }
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

    ulong A[25] = {0};

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong domain_val = (ulong)(domain & 0xFFu);

    // Fast-path: saturate bandwidth for uniform 32-byte loads using vectors
    if (msg_lanes == 4) {
        device const ulong4 *in_data4 = (device const ulong4 *)in_data;
        ulong4 val = in_data4[idx];
        A[0] ^= val.x;
        A[1] ^= val.y;
        A[2] ^= val.z;
        A[3] ^= val.w;
        A[4] ^= domain_val;
    } else {
        uint in_base = idx * msg_lanes;
        for (uint i = 0; i < msg_lanes; ++i) {
            A[i] ^= in_data[in_base + i];
        }
        A[msg_lanes] ^= domain_val;
    }

    A[rate_lanes - 1] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;

    // Fast-path: saturate bandwidth for standard 32-byte and 64-byte outputs
    if (out_lanes == 4) {
        keccak_f1600(A);
        device ulong4 *out_data4 = (device ulong4 *)out_data;
        out_data4[idx] = ulong4(A[0], A[1], A[2], A[3]);
    } else if (out_lanes == 8) {
        keccak_f1600(A);
        device ulong4 *out_data4 = (device ulong4 *)out_data;
        out_data4[idx * 2]     = ulong4(A[0], A[1], A[2], A[3]);
        out_data4[idx * 2 + 1] = ulong4(A[4], A[5], A[6], A[7]);
    } else {
        uint written = 0;
        for (;;) {
            keccak_f1600(A);
            uint remaining = out_lanes - written;
            uint take = remaining < rate_lanes ? remaining : rate_lanes;
            
            for (uint i = 0; i < take; ++i) {
                out_data[out_base + written + i] = A[i];
            }

            written += take;
            if (written >= out_lanes) break;
        }
    }
}