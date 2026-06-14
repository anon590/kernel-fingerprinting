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

inline ulong rotl(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

inline void keccak_f1600(thread ulong *A) {
    #pragma unroll 24
    for (uint r = 0; r < 24; ++r) {
        // --- Theta ---
        ulong C0 = A[0] ^ A[5] ^ A[10] ^ A[15] ^ A[20];
        ulong C1 = A[1] ^ A[6] ^ A[11] ^ A[16] ^ A[21];
        ulong C2 = A[2] ^ A[7] ^ A[12] ^ A[17] ^ A[22];
        ulong C3 = A[3] ^ A[8] ^ A[13] ^ A[18] ^ A[23];
        ulong C4 = A[4] ^ A[9] ^ A[14] ^ A[19] ^ A[24];

        ulong D0 = C4 ^ rotl(C1, 1u);
        ulong D1 = C0 ^ rotl(C2, 1u);
        ulong D2 = C1 ^ rotl(C3, 1u);
        ulong D3 = C2 ^ rotl(C4, 1u);
        ulong D4 = C3 ^ rotl(C0, 1u);

        A[0] ^= D0; A[5] ^= D0; A[10] ^= D0; A[15] ^= D0; A[20] ^= D0;
        A[1] ^= D1; A[6] ^= D1; A[11] ^= D1; A[16] ^= D1; A[21] ^= D1;
        A[2] ^= D2; A[7] ^= D2; A[12] ^= D2; A[17] ^= D2; A[22] ^= D2;
        A[3] ^= D3; A[8] ^= D3; A[13] ^= D3; A[18] ^= D3; A[23] ^= D3;
        A[4] ^= D4; A[9] ^= D4; A[14] ^= D4; A[19] ^= D4; A[24] ^= D4;

        // --- Rho and Pi (In-place 24-element cycle) ---
        ulong temp = A[1];
        ulong next;
        next = A[10]; A[10] = rotl(temp, 1u);  temp = next;
        next = A[7];  A[7]  = rotl(temp, 3u);  temp = next;
        next = A[11]; A[11] = rotl(temp, 6u);  temp = next;
        next = A[17]; A[17] = rotl(temp, 10u); temp = next;
        next = A[18]; A[18] = rotl(temp, 15u); temp = next;
        next = A[3];  A[3]  = rotl(temp, 21u); temp = next;
        next = A[5];  A[5]  = rotl(temp, 28u); temp = next;
        next = A[16]; A[16] = rotl(temp, 36u); temp = next;
        next = A[8];  A[8]  = rotl(temp, 45u); temp = next;
        next = A[21]; A[21] = rotl(temp, 55u); temp = next;
        next = A[24]; A[24] = rotl(temp, 2u);  temp = next;
        next = A[4];  A[4]  = rotl(temp, 14u); temp = next;
        next = A[15]; A[15] = rotl(temp, 27u); temp = next;
        next = A[23]; A[23] = rotl(temp, 41u); temp = next;
        next = A[19]; A[19] = rotl(temp, 56u); temp = next;
        next = A[13]; A[13] = rotl(temp, 8u);  temp = next;
        next = A[12]; A[12] = rotl(temp, 25u); temp = next;
        next = A[2];  A[2]  = rotl(temp, 43u); temp = next;
        next = A[20]; A[20] = rotl(temp, 62u); temp = next;
        next = A[14]; A[14] = rotl(temp, 18u); temp = next;
        next = A[22]; A[22] = rotl(temp, 39u); temp = next;
        next = A[9];  A[9]  = rotl(temp, 61u); temp = next;
        next = A[6];  A[6]  = rotl(temp, 20u); temp = next;
        A[1] = rotl(temp, 44u);

        // --- Chi ---
        {
            ulong T0 = A[0], T1 = A[1], T2 = A[2], T3 = A[3], T4 = A[4];
            A[0] = T0 ^ (~T1 & T2);
            A[1] = T1 ^ (~T2 & T3);
            A[2] = T2 ^ (~T3 & T4);
            A[3] = T3 ^ (~T4 & T0);
            A[4] = T4 ^ (~T0 & T1);
        }
        {
            ulong T0 = A[5], T1 = A[6], T2 = A[7], T3 = A[8], T4 = A[9];
            A[5] = T0 ^ (~T1 & T2);
            A[6] = T1 ^ (~T2 & T3);
            A[7] = T2 ^ (~T3 & T4);
            A[8] = T3 ^ (~T4 & T0);
            A[9] = T4 ^ (~T0 & T1);
        }
        {
            ulong T0 = A[10], T1 = A[11], T2 = A[12], T3 = A[13], T4 = A[14];
            A[10] = T0 ^ (~T1 & T2);
            A[11] = T1 ^ (~T2 & T3);
            A[12] = T2 ^ (~T3 & T4);
            A[13] = T3 ^ (~T4 & T0);
            A[14] = T4 ^ (~T0 & T1);
        }
        {
            ulong T0 = A[15], T1 = A[16], T2 = A[17], T3 = A[18], T4 = A[19];
            A[15] = T0 ^ (~T1 & T2);
            A[16] = T1 ^ (~T2 & T3);
            A[17] = T2 ^ (~T3 & T4);
            A[18] = T3 ^ (~T4 & T0);
            A[19] = T4 ^ (~T0 & T1);
        }
        {
            ulong T0 = A[20], T1 = A[21], T2 = A[22], T3 = A[23], T4 = A[24];
            A[20] = T0 ^ (~T1 & T2);
            A[21] = T1 ^ (~T2 & T3);
            A[22] = T2 ^ (~T3 & T4);
            A[23] = T3 ^ (~T4 & T0);
            A[24] = T4 ^ (~T0 & T1);
        }

        // --- Iota ---
        A[0] ^= KECCAK_RC[r];
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

    ulong A[25];

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;
    ulong domain_val = (ulong)(domain & 0xFFu);

    // Fast-path vectorized init
    if (msg_lanes == 4) {
        ulong4 msg = ((device const ulong4*)in_data)[idx];
        A[0] = msg.x;
        A[1] = msg.y;
        A[2] = msg.z;
        A[3] = msg.w;
        A[4] = domain_val;
        
        #pragma unroll
        for (uint i = 5; i < 25; ++i) A[i] = 0;
    } else {
        #pragma unroll
        for (uint i = 0; i < 25; ++i) A[i] = 0;

        uint in_base = idx * msg_lanes;
        switch (msg_lanes) {
            case 24: A[23] ^= in_data[in_base + 23];
            case 23: A[22] ^= in_data[in_base + 22];
            case 22: A[21] ^= in_data[in_base + 21];
            case 21: A[20] ^= in_data[in_base + 20];
            case 20: A[19] ^= in_data[in_base + 19];
            case 19: A[18] ^= in_data[in_base + 18];
            case 18: A[17] ^= in_data[in_base + 17];
            case 17: A[16] ^= in_data[in_base + 16];
            case 16: A[15] ^= in_data[in_base + 15];
            case 15: A[14] ^= in_data[in_base + 14];
            case 14: A[13] ^= in_data[in_base + 13];
            case 13: A[12] ^= in_data[in_base + 12];
            case 12: A[11] ^= in_data[in_base + 11];
            case 11: A[10] ^= in_data[in_base + 10];
            case 10: A[ 9] ^= in_data[in_base +  9];
            case  9: A[ 8] ^= in_data[in_base +  8];
            case  8: A[ 7] ^= in_data[in_base +  7];
            case  7: A[ 6] ^= in_data[in_base +  6];
            case  6: A[ 5] ^= in_data[in_base +  5];
            case  5: A[ 4] ^= in_data[in_base +  4];
            case  4: A[ 3] ^= in_data[in_base +  3];
            case  3: A[ 2] ^= in_data[in_base +  2];
            case  2: A[ 1] ^= in_data[in_base +  1];
            case  1: A[ 0] ^= in_data[in_base +  0]; break;
            default: break;
        }
        switch (msg_lanes) {
            case 24: A[24] ^= domain_val; break;
            case 23: A[23] ^= domain_val; break;
            case 22: A[22] ^= domain_val; break;
            case 21: A[21] ^= domain_val; break;
            case 20: A[20] ^= domain_val; break;
            case 19: A[19] ^= domain_val; break;
            case 18: A[18] ^= domain_val; break;
            case 17: A[17] ^= domain_val; break;
            case 16: A[16] ^= domain_val; break;
            case 15: A[15] ^= domain_val; break;
            case 14: A[14] ^= domain_val; break;
            case 13: A[13] ^= domain_val; break;
            case 12: A[12] ^= domain_val; break;
            case 11: A[11] ^= domain_val; break;
            case 10: A[10] ^= domain_val; break;
            case  9: A[ 9] ^= domain_val; break;
            case  8: A[ 8] ^= domain_val; break;
            case  7: A[ 7] ^= domain_val; break;
            case  6: A[ 6] ^= domain_val; break;
            case  5: A[ 5] ^= domain_val; break;
            case  4: A[ 4] ^= domain_val; break;
            case  3: A[ 3] ^= domain_val; break;
            case  2: A[ 2] ^= domain_val; break;
            case  1: A[ 1] ^= domain_val; break;
            case  0: A[ 0] ^= domain_val; break;
            default: break;
        }
    }

    // Direct explicit rate padding
    if (rate_lanes == 17) {
        A[16] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 21) {
        A[20] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 9) {
        A[8] ^= 0x8000000000000000ul;
    } else if (rate_lanes == 13) {
        A[12] ^= 0x8000000000000000ul;
    } else {
        switch (rate_lanes - 1) {
            case 24: A[24] ^= 0x8000000000000000ul; break;
            case 23: A[23] ^= 0x8000000000000000ul; break;
            case 22: A[22] ^= 0x8000000000000000ul; break;
            case 21: A[21] ^= 0x8000000000000000ul; break;
            case 20: A[20] ^= 0x8000000000000000ul; break;
            case 19: A[19] ^= 0x8000000000000000ul; break;
            case 18: A[18] ^= 0x8000000000000000ul; break;
            case 17: A[17] ^= 0x8000000000000000ul; break;
            case 16: A[16] ^= 0x8000000000000000ul; break;
            case 15: A[15] ^= 0x8000000000000000ul; break;
            case 14: A[14] ^= 0x8000000000000000ul; break;
            case 13: A[13] ^= 0x8000000000000000ul; break;
            case 12: A[12] ^= 0x8000000000000000ul; break;
            case 11: A[11] ^= 0x8000000000000000ul; break;
            case 10: A[10] ^= 0x8000000000000000ul; break;
            case  9: A[ 9] ^= 0x8000000000000000ul; break;
            case  8: A[ 8] ^= 0x8000000000000000ul; break;
            case  7: A[ 7] ^= 0x8000000000000000ul; break;
            case  6: A[ 6] ^= 0x8000000000000000ul; break;
            case  5: A[ 5] ^= 0x8000000000000000ul; break;
            case  4: A[ 4] ^= 0x8000000000000000ul; break;
            case  3: A[ 3] ^= 0x8000000000000000ul; break;
            case  2: A[ 2] ^= 0x8000000000000000ul; break;
            case  1: A[ 1] ^= 0x8000000000000000ul; break;
            case  0: A[ 0] ^= 0x8000000000000000ul; break;
            default: break;
        }
    }

    uint out_base = idx * out_lanes;
    uint written = 0;

    for (;;) {
        keccak_f1600(A);
        
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        
        if (take == 4 && written == 0 && out_lanes == 4) {
            ((device ulong4*)out_data)[idx] = { A[0], A[1], A[2], A[3] };
        } else if (take == 8 && written == 0 && out_lanes == 8) {
            ((device ulong4*)out_data)[idx * 2 + 0] = { A[0], A[1], A[2], A[3] };
            ((device ulong4*)out_data)[idx * 2 + 1] = { A[4], A[5], A[6], A[7] };
        } else if (take == 16 && written == 0 && out_lanes == 16) {
            ((device ulong4*)out_data)[idx * 4 + 0] = { A[0], A[1], A[2], A[3] };
            ((device ulong4*)out_data)[idx * 4 + 1] = { A[4], A[5], A[6], A[7] };
            ((device ulong4*)out_data)[idx * 4 + 2] = { A[8], A[9], A[10], A[11] };
            ((device ulong4*)out_data)[idx * 4 + 3] = { A[12], A[13], A[14], A[15] };
        } else {
            switch (take) {
                case 25: out_data[out_base + written + 24] = A[24];
                case 24: out_data[out_base + written + 23] = A[23];
                case 23: out_data[out_base + written + 22] = A[22];
                case 22: out_data[out_base + written + 21] = A[21];
                case 21: out_data[out_base + written + 20] = A[20];
                case 20: out_data[out_base + written + 19] = A[19];
                case 19: out_data[out_base + written + 18] = A[18];
                case 18: out_data[out_base + written + 17] = A[17];
                case 17: out_data[out_base + written + 16] = A[16];
                case 16: out_data[out_base + written + 15] = A[15];
                case 15: out_data[out_base + written + 14] = A[14];
                case 14: out_data[out_base + written + 13] = A[13];
                case 13: out_data[out_base + written + 12] = A[12];
                case 12: out_data[out_base + written + 11] = A[11];
                case 11: out_data[out_base + written + 10] = A[10];
                case 10: out_data[out_base + written +  9] = A[ 9];
                case  9: out_data[out_base + written +  8] = A[ 8];
                case  8: out_data[out_base + written +  7] = A[ 7];
                case  7: out_data[out_base + written +  6] = A[ 6];
                case  6: out_data[out_base + written +  5] = A[ 5];
                case  5: out_data[out_base + written +  4] = A[ 4];
                case  4: out_data[out_base + written +  3] = A[ 3];
                case  3: out_data[out_base + written +  2] = A[ 2];
                case  2: out_data[out_base + written +  1] = A[ 1];
                case  1: out_data[out_base + written +  0] = A[ 0]; break;
                default: break;
            }
        }

        written += take;
        if (written >= out_lanes) break;
    }
}