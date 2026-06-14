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

    ulong A00=0, A01=0, A02=0, A03=0, A04=0;
    ulong A05=0, A06=0, A07=0, A08=0, A09=0;
    ulong A10=0, A11=0, A12=0, A13=0, A14=0;
    ulong A15=0, A16=0, A17=0, A18=0, A19=0;
    ulong A20=0, A21=0, A22=0, A23=0, A24=0;

    // Load seed straight into lanes (no intermediate s[]).
    if (n_lanes > 0u)  A00 = seeds[base + 0];
    if (n_lanes > 1u)  A01 = seeds[base + 1];
    if (n_lanes > 2u)  A02 = seeds[base + 2];
    if (n_lanes > 3u)  A03 = seeds[base + 3];
    if (n_lanes > 4u)  A04 = seeds[base + 4];
    if (n_lanes > 5u)  A05 = seeds[base + 5];
    if (n_lanes > 6u)  A06 = seeds[base + 6];
    if (n_lanes > 7u)  A07 = seeds[base + 7];
    if (n_lanes > 8u)  A08 = seeds[base + 8];
    if (n_lanes > 9u)  A09 = seeds[base + 9];
    if (n_lanes > 10u) A10 = seeds[base + 10];
    if (n_lanes > 11u) A11 = seeds[base + 11];
    if (n_lanes > 12u) A12 = seeds[base + 12];
    if (n_lanes > 13u) A13 = seeds[base + 13];
    if (n_lanes > 14u) A14 = seeds[base + 14];
    if (n_lanes > 15u) A15 = seeds[base + 15];

    const uint dom_lane = n_lanes;
    const ulong pad_dom = 0x06ul;
    const ulong pad_fin = 0x8000000000000000ul;

    // Precompute padding XOR pattern as a mask we apply each step.
    // After Keccak-f, lanes 0..n_lanes-1 are the new chunk; lanes n_lanes..24
    // must be cleared. We clear lanes n_lanes..15 unconditionally and lanes
    // 16..24 unconditionally, then apply pad XORs.

    for (uint step = 0u; step < w; ++step) {
        // Zero lanes n_lanes..15 (rate region above the chunk) and 16..24 (capacity).
        if (n_lanes < 16u) {
            if (n_lanes <= 0u)  A00 = 0;
            if (n_lanes <= 1u)  A01 = 0;
            if (n_lanes <= 2u)  A02 = 0;
            if (n_lanes <= 3u)  A03 = 0;
            if (n_lanes <= 4u)  A04 = 0;
            if (n_lanes <= 5u)  A05 = 0;
            if (n_lanes <= 6u)  A06 = 0;
            if (n_lanes <= 7u)  A07 = 0;
            if (n_lanes <= 8u)  A08 = 0;
            if (n_lanes <= 9u)  A09 = 0;
            if (n_lanes <= 10u) A10 = 0;
            if (n_lanes <= 11u) A11 = 0;
            if (n_lanes <= 12u) A12 = 0;
            if (n_lanes <= 13u) A13 = 0;
            if (n_lanes <= 14u) A14 = 0;
            A15 = 0;
        }
        A16 = pad_fin;  // capacity + final pad in one shot
        A17 = 0; A18 = 0; A19 = 0;
        A20 = 0; A21 = 0; A22 = 0; A23 = 0; A24 = 0;

        // Domain pad byte 0x06 at lane = n_lanes. n_lanes <= 16 here.
        switch (dom_lane) {
            case 0:  A00 ^= pad_dom; break;
            case 1:  A01 ^= pad_dom; break;
            case 2:  A02 ^= pad_dom; break;
            case 3:  A03 ^= pad_dom; break;
            case 4:  A04 ^= pad_dom; break;
            case 5:  A05 ^= pad_dom; break;
            case 6:  A06 ^= pad_dom; break;
            case 7:  A07 ^= pad_dom; break;
            case 8:  A08 ^= pad_dom; break;
            case 9:  A09 ^= pad_dom; break;
            case 10: A10 ^= pad_dom; break;
            case 11: A11 ^= pad_dom; break;
            case 12: A12 ^= pad_dom; break;
            case 13: A13 ^= pad_dom; break;
            case 14: A14 ^= pad_dom; break;
            case 15: A15 ^= pad_dom; break;
            case 16: A16 ^= pad_dom; break;
            default: break;
        }

        // ----- Keccak-f[1600], 24 rounds -----
        // Unroll by 2: round r processes A->A' producing same-layout state.
        // We use an explicit double-step to let the compiler see more ILP
        // across the round boundary and to halve the loop overhead.
        for (uint r = 0u; r < 24u; r += 2u) {
            // ===== Round r =====
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

            // Fuse theta XOR into rho+pi source reads.
            ulong B00 =      (A00 ^ D0);
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

            // chi - use bic pattern: andnot(b,c) = c & ~b
            A00 = B00 ^ (B02 & ~B01);
            A01 = B01 ^ (B03 & ~B02);
            A02 = B02 ^ (B04 & ~B03);
            A03 = B03 ^ (B00 & ~B04);
            A04 = B04 ^ (B01 & ~B00);
            A05 = B05 ^ (B07 & ~B06);
            A06 = B06 ^ (B08 & ~B07);
            A07 = B07 ^ (B09 & ~B08);
            A08 = B08 ^ (B05 & ~B09);
            A09 = B09 ^ (B06 & ~B05);
            A10 = B10 ^ (B12 & ~B11);
            A11 = B11 ^ (B13 & ~B12);
            A12 = B12 ^ (B14 & ~B13);
            A13 = B13 ^ (B10 & ~B14);
            A14 = B14 ^ (B11 & ~B10);
            A15 = B15 ^ (B17 & ~B16);
            A16 = B16 ^ (B18 & ~B17);
            A17 = B17 ^ (B19 & ~B18);
            A18 = B18 ^ (B15 & ~B19);
            A19 = B19 ^ (B16 & ~B15);
            A20 = B20 ^ (B22 & ~B21);
            A21 = B21 ^ (B23 & ~B22);
            A22 = B22 ^ (B24 & ~B23);
            A23 = B23 ^ (B20 & ~B24);
            A24 = B24 ^ (B21 & ~B20);

            A00 ^= KECCAK_RC[r];

            // ===== Round r+1 =====
            C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;
            C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;
            C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;
            C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;
            C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;

            D0 = C4 ^ ROL(C1, 1);
            D1 = C0 ^ ROL(C2, 1);
            D2 = C1 ^ ROL(C3, 1);
            D3 = C2 ^ ROL(C4, 1);
            D4 = C3 ^ ROL(C0, 1);

            B00 =      (A00 ^ D0);
            B10 = ROL( (A01 ^ D1),  1);
            B20 = ROL( (A02 ^ D2), 62);
            B05 = ROL( (A03 ^ D3), 28);
            B15 = ROL( (A04 ^ D4), 27);
            B16 = ROL( (A05 ^ D0), 36);
            B01 = ROL( (A06 ^ D1), 44);
            B11 = ROL( (A07 ^ D2),  6);
            B21 = ROL( (A08 ^ D3), 55);
            B06 = ROL( (A09 ^ D4), 20);
            B07 = ROL( (A10 ^ D0),  3);
            B17 = ROL( (A11 ^ D1), 10);
            B02 = ROL( (A12 ^ D2), 43);
            B12 = ROL( (A13 ^ D3), 25);
            B22 = ROL( (A14 ^ D4), 39);
            B23 = ROL( (A15 ^ D0), 41);
            B08 = ROL( (A16 ^ D1), 45);
            B18 = ROL( (A17 ^ D2), 15);
            B03 = ROL( (A18 ^ D3), 21);
            B13 = ROL( (A19 ^ D4),  8);
            B14 = ROL( (A20 ^ D0), 18);
            B24 = ROL( (A21 ^ D1),  2);
            B09 = ROL( (A22 ^ D2), 61);
            B19 = ROL( (A23 ^ D3), 56);
            B04 = ROL( (A24 ^ D4), 14);

            A00 = B00 ^ (B02 & ~B01);
            A01 = B01 ^ (B03 & ~B02);
            A02 = B02 ^ (B04 & ~B03);
            A03 = B03 ^ (B00 & ~B04);
            A04 = B04 ^ (B01 & ~B00);
            A05 = B05 ^ (B07 & ~B06);
            A06 = B06 ^ (B08 & ~B07);
            A07 = B07 ^ (B09 & ~B08);
            A08 = B08 ^ (B05 & ~B09);
            A09 = B09 ^ (B06 & ~B05);
            A10 = B10 ^ (B12 & ~B11);
            A11 = B11 ^ (B13 & ~B12);
            A12 = B12 ^ (B14 & ~B13);
            A13 = B13 ^ (B10 & ~B14);
            A14 = B14 ^ (B11 & ~B10);
            A15 = B15 ^ (B17 & ~B16);
            A16 = B16 ^ (B18 & ~B17);
            A17 = B17 ^ (B19 & ~B18);
            A18 = B18 ^ (B15 & ~B19);
            A19 = B19 ^ (B16 & ~B15);
            A20 = B20 ^ (B22 & ~B21);
            A21 = B21 ^ (B23 & ~B22);
            A22 = B22 ^ (B24 & ~B23);
            A23 = B23 ^ (B20 & ~B24);
            A24 = B24 ^ (B21 & ~B20);

            A00 ^= KECCAK_RC[r + 1u];
        }
    }

    // Squeeze.
    if (n_lanes > 0u)  tips[base + 0]  = A00;
    if (n_lanes > 1u)  tips[base + 1]  = A01;
    if (n_lanes > 2u)  tips[base + 2]  = A02;
    if (n_lanes > 3u)  tips[base + 3]  = A03;
    if (n_lanes > 4u)  tips[base + 4]  = A04;
    if (n_lanes > 5u)  tips[base + 5]  = A05;
    if (n_lanes > 6u)  tips[base + 6]  = A06;
    if (n_lanes > 7u)  tips[base + 7]  = A07;
    if (n_lanes > 8u)  tips[base + 8]  = A08;
    if (n_lanes > 9u)  tips[base + 9]  = A09;
    if (n_lanes > 10u) tips[base + 10] = A10;
    if (n_lanes > 11u) tips[base + 11] = A11;
    if (n_lanes > 12u) tips[base + 12] = A12;
    if (n_lanes > 13u) tips[base + 13] = A13;
    if (n_lanes > 14u) tips[base + 14] = A14;
    if (n_lanes > 15u) tips[base + 15] = A15;
}