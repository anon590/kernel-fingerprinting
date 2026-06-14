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

inline ulong ROL(ulong x, uint k) {
    return rotate(x, (ulong)k);
}

kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_chains) return;

    uint n_lanes = n_bytes >> 3;
    uint base = idx * n_lanes;

    // Load seed into chunk[0..n_lanes-1]
    ulong chunk[17];
    for (uint k = 0u; k < 17u; ++k) chunk[k] = 0;
    for (uint k = 0u; k < n_lanes; ++k) chunk[k] = seeds[base + k];

    // SHA3 padding bytes into chunk[n_lanes] (low byte) — applied each step.
    // Lane 16 always gets the high bit pad. Since n_lanes < 17, dom_lane < 17.
    uint dom_lane = n_lanes;

    for (uint step = 0u; step < w; ++step) {
        // Apply padding to absorbed block (lanes 0..16 of state come from chunk[]).
        chunk[dom_lane] ^= 0x06ul;
        // chunk[16] gets the 0x80 in byte 7. dom_lane != 16 since n_lanes < 17 and
        // n_bytes < rate => n_lanes <= 16; but n_lanes is bytes/8 with n_bytes<136 => n_lanes<=16.
        // If dom_lane == 16, both XORs go on lane 16: handle with separate XOR.
        ulong lane16_pad = 0x8000000000000000ul;

        // Initialize 25-lane state. Capacity lanes (17..24) are zero.
        ulong A00 = chunk[0];
        ulong A01 = chunk[1];
        ulong A02 = chunk[2];
        ulong A03 = chunk[3];
        ulong A04 = chunk[4];
        ulong A05 = chunk[5];
        ulong A06 = chunk[6];
        ulong A07 = chunk[7];
        ulong A08 = chunk[8];
        ulong A09 = chunk[9];
        ulong A10 = chunk[10];
        ulong A11 = chunk[11];
        ulong A12 = chunk[12];
        ulong A13 = chunk[13];
        ulong A14 = chunk[14];
        ulong A15 = chunk[15];
        ulong A16 = chunk[16] ^ lane16_pad;
        ulong A17 = 0, A18 = 0, A19 = 0;
        ulong A20 = 0, A21 = 0, A22 = 0, A23 = 0, A24 = 0;

        // Undo the in-place XORs so chunk[] is reusable next step
        chunk[dom_lane] ^= 0x06ul;

        // 24 rounds of Keccak-f1600
        for (uint r = 0u; r < 24u; ++r) {
            ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;
            ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;
            ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;
            ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;
            ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            // theta + rho + pi combined: B[dst] = ROL(A[src] ^ D[x_src], rho[src])
            ulong B00 =      (A00 ^ D0)       ;
            ulong B10 = ROL( (A01 ^ D1),  1);
            ulong B20 = ROL( (A02 ^ D2), 62);
            ulong B05 = ROL( (A03 ^ D3), 28);
            ulong B15 = ROL( (A04 ^ D4), 27);

            ulong B16 = ROL( (A05 ^ D0), 36);
            ulong B01 = ROL( (A06 ^ D1), 44);
            ulong B11 = ROL( (A07 ^ D2),  6);
            ulong B21 = ROL( (A08 ^ D3), 55);
            ulong B06 = ROL( (A09 ^ D4), 20);

            ulong B07 = ROL( (A10 ^ D0),  3);
            ulong B17 = ROL( (A11 ^ D1), 10);
            ulong B02 = ROL( (A12 ^ D2), 43);
            ulong B12 = ROL( (A13 ^ D3), 25);
            ulong B22 = ROL( (A14 ^ D4), 39);

            ulong B23 = ROL( (A15 ^ D0), 41);
            ulong B08 = ROL( (A16 ^ D1), 45);
            ulong B18 = ROL( (A17 ^ D2), 15);
            ulong B03 = ROL( (A18 ^ D3), 21);
            ulong B13 = ROL( (A19 ^ D4),  8);

            ulong B14 = ROL( (A20 ^ D0), 18);
            ulong B24 = ROL( (A21 ^ D1),  2);
            ulong B09 = ROL( (A22 ^ D2), 61);
            ulong B19 = ROL( (A23 ^ D3), 56);
            ulong B04 = ROL( (A24 ^ D4), 14);

            // chi
            A00 = B00 ^ ((~B01) & B02);
            A01 = B01 ^ ((~B02) & B03);
            A02 = B02 ^ ((~B03) & B04);
            A03 = B03 ^ ((~B04) & B00);
            A04 = B04 ^ ((~B00) & B01);

            A05 = B05 ^ ((~B06) & B07);
            A06 = B06 ^ ((~B07) & B08);
            A07 = B07 ^ ((~B08) & B09);
            A08 = B08 ^ ((~B09) & B05);
            A09 = B09 ^ ((~B05) & B06);

            A10 = B10 ^ ((~B11) & B12);
            A11 = B11 ^ ((~B12) & B13);
            A12 = B12 ^ ((~B13) & B14);
            A13 = B13 ^ ((~B14) & B10);
            A14 = B14 ^ ((~B10) & B11);

            A15 = B15 ^ ((~B16) & B17);
            A16 = B16 ^ ((~B17) & B18);
            A17 = B17 ^ ((~B18) & B19);
            A18 = B18 ^ ((~B19) & B15);
            A19 = B19 ^ ((~B15) & B16);

            A20 = B20 ^ ((~B21) & B22);
            A21 = B21 ^ ((~B22) & B23);
            A22 = B22 ^ ((~B23) & B24);
            A23 = B23 ^ ((~B24) & B20);
            A24 = B24 ^ ((~B20) & B21);

            // iota
            A00 ^= KECCAK_RC[r];
        }

        // Squeeze first n_lanes lanes back into chunk[] for next iteration.
        chunk[0]  = A00;
        chunk[1]  = A01;
        chunk[2]  = A02;
        chunk[3]  = A03;
        chunk[4]  = A04;
        chunk[5]  = A05;
        chunk[6]  = A06;
        chunk[7]  = A07;
        chunk[8]  = A08;
        chunk[9]  = A09;
        chunk[10] = A10;
        chunk[11] = A11;
        chunk[12] = A12;
        chunk[13] = A13;
        chunk[14] = A14;
        chunk[15] = A15;
        chunk[16] = A16;
        // Zero out the lanes beyond n_lanes so the next step's absorb is clean.
        for (uint k = n_lanes; k < 17u; ++k) chunk[k] = 0;
    }

    // Write tip
    for (uint k = 0u; k < n_lanes; ++k) {
        tips[base + k] = chunk[k];
    }
}