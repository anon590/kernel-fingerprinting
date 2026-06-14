#include <metal_stdlib>
using namespace metal;

constant ulong KRC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

// Lane Complement Technique (LCT) round constants: certain lanes are kept
// in inverted form across rounds, so chi becomes `x ^ (y | z)` or
// `x ^ (~y & z)` selectively, and only a small fixed mask is XORed into
// iota to keep things consistent. We follow the Bertoni/Daemen LCT layout
// where lanes 1, 2, 8, 12, 17, 20 are stored inverted.

// chi with LCT: for each row, identify which lanes are inverted (stored complemented).
// Row 0: lanes (0,1,2,3,4). Inverted: {1,2}.
// Row 1: lanes (5,6,7,8,9). Inverted: {8}.
// Row 2: lanes (10,11,12,13,14). Inverted: {12}.
// Row 3: lanes (15,16,17,18,19). Inverted: {17}.
// Row 4: lanes (20,21,22,23,24). Inverted: {20}.
// After chi, the same set must remain inverted (this is a property of LCT).

// For each lane y in a row [a,b,c,d,e] (cyclic), output = a ^ ((~b) & c).
// If a stored value `a` is actually ~A (true), and similarly bm for b means stored=~B,
// then:
//   stored_a_new = stored_a ^ ((~stored_b if !bm else stored_b) & (stored_c if !cm else ~stored_c))
// For each output lane we precompute the formula in terms of stored values.

inline ulong ROLk(ulong x, uint k) {
    return (x << k) | (x >> (64 - k));
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

    const uint n_lanes = n_bytes >> 3;
    const uint base = idx * n_lanes;
    const uint W = w;

    // Load seed lanes into a small array; we'll reseed each chain step.
    ulong M[17];
    for (uint k = 0u; k < 17u; ++k) M[k] = 0;
    for (uint k = 0u; k < n_lanes; ++k) M[k] = seeds[base + k];

    // Lane variables.
    ulong A00, A01, A02, A03, A04;
    ulong A05, A06, A07, A08, A09;
    ulong A10, A11, A12, A13, A14;
    ulong A15, A16, A17, A18, A19;
    ulong A20, A21, A22, A23, A24;

    for (uint step = 0u; step < W; ++step) {
        // Initialize state from M with SHA3 padding.
        A00 = M[0];  A01 = M[1];  A02 = M[2];  A03 = M[3];  A04 = M[4];
        A05 = M[5];  A06 = M[6];  A07 = M[7];  A08 = M[8];  A09 = M[9];
        A10 = M[10]; A11 = M[11]; A12 = M[12]; A13 = M[13]; A14 = M[14];
        A15 = M[15]; A16 = M[16];
        A17 = 0; A18 = 0; A19 = 0;
        A20 = 0; A21 = 0; A22 = 0; A23 = 0; A24 = 0;

        // Domain-pad XOR 0x06 into lane[n_lanes].
        switch (n_lanes) {
            case 0:  A00 ^= 0x06ul; break;
            case 1:  A01 ^= 0x06ul; break;
            case 2:  A02 ^= 0x06ul; break;
            case 3:  A03 ^= 0x06ul; break;
            case 4:  A04 ^= 0x06ul; break;
            case 5:  A05 ^= 0x06ul; break;
            case 6:  A06 ^= 0x06ul; break;
            case 7:  A07 ^= 0x06ul; break;
            case 8:  A08 ^= 0x06ul; break;
            case 9:  A09 ^= 0x06ul; break;
            case 10: A10 ^= 0x06ul; break;
            case 11: A11 ^= 0x06ul; break;
            case 12: A12 ^= 0x06ul; break;
            case 13: A13 ^= 0x06ul; break;
            case 14: A14 ^= 0x06ul; break;
            case 15: A15 ^= 0x06ul; break;
            case 16: A16 ^= 0x06ul; break;
            default: break;
        }
        // Final pad byte 0x80 at byte 7 of lane 16.
        A16 ^= 0x8000000000000000ul;

        // ---- Apply LCT: invert lanes {1,2,8,12,17,20} once at start. ----
        A01 = ~A01;
        A02 = ~A02;
        A08 = ~A08;
        A12 = ~A12;
        A17 = ~A17;
        A20 = ~A20;

        // 24 Keccak-f rounds. We work with stored (possibly inverted) lanes.
        // After theta + rho+pi, the lane identities are permuted but the
        // inversion bits move with them; we handle chi accordingly.
        //
        // Strategy: do standard theta and rho+pi on stored values. Theta is
        // linear so it commutes with global complement of any lane: if any
        // of the 5 lanes in a column-XOR was inverted, the column parity
        // is just flipped, which is then XORed back equally everywhere ->
        // no net effect on the inversion pattern (since each lane gets
        // XORed by the same D vector, and D is a XOR sum so its effect on
        // inversion is 0 mod 2 if even number of inverted contributors).
        //
        // To keep things bit-exact and simple, we'll just clear LCT at the
        // start of each round before chi (un-invert), do plain chi, and
        // re-invert. The "savings" come from a different angle: we instead
        // fully unroll the round structure and let the optimizer schedule.
        //
        // So actually undo LCT here -- LCT was a red herring for u64 width
        // on Apple GPUs since `~` is free as part of `andn`. Revert.
        A01 = ~A01; A02 = ~A02; A08 = ~A08; A12 = ~A12; A17 = ~A17; A20 = ~A20;

        // Manually unrolled 24 rounds.
        #define ROUND(RC) {                                              \
            ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;                      \
            ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;                      \
            ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;                      \
            ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;                      \
            ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;                      \
            ulong D0 = C4 ^ ROLk(C1, 1);                                 \
            ulong D1 = C0 ^ ROLk(C2, 1);                                 \
            ulong D2 = C1 ^ ROLk(C3, 1);                                 \
            ulong D3 = C2 ^ ROLk(C4, 1);                                 \
            ulong D4 = C3 ^ ROLk(C0, 1);                                 \
            ulong T00 = A00 ^ D0;                                        \
            ulong T01 = A01 ^ D1;                                        \
            ulong T02 = A02 ^ D2;                                        \
            ulong T03 = A03 ^ D3;                                        \
            ulong T04 = A04 ^ D4;                                        \
            ulong T05 = A05 ^ D0;                                        \
            ulong T06 = A06 ^ D1;                                        \
            ulong T07 = A07 ^ D2;                                        \
            ulong T08 = A08 ^ D3;                                        \
            ulong T09 = A09 ^ D4;                                        \
            ulong T10 = A10 ^ D0;                                        \
            ulong T11 = A11 ^ D1;                                        \
            ulong T12 = A12 ^ D2;                                        \
            ulong T13 = A13 ^ D3;                                        \
            ulong T14 = A14 ^ D4;                                        \
            ulong T15 = A15 ^ D0;                                        \
            ulong T16 = A16 ^ D1;                                        \
            ulong T17 = A17 ^ D2;                                        \
            ulong T18 = A18 ^ D3;                                        \
            ulong T19 = A19 ^ D4;                                        \
            ulong T20 = A20 ^ D0;                                        \
            ulong T21 = A21 ^ D1;                                        \
            ulong T22 = A22 ^ D2;                                        \
            ulong T23 = A23 ^ D3;                                        \
            ulong T24 = A24 ^ D4;                                        \
            ulong B00 = T00;                                             \
            ulong B10 = ROLk(T01,  1);                                   \
            ulong B20 = ROLk(T02, 62);                                   \
            ulong B05 = ROLk(T03, 28);                                   \
            ulong B15 = ROLk(T04, 27);                                   \
            ulong B16 = ROLk(T05, 36);                                   \
            ulong B01 = ROLk(T06, 44);                                   \
            ulong B11 = ROLk(T07,  6);                                   \
            ulong B21 = ROLk(T08, 55);                                   \
            ulong B06 = ROLk(T09, 20);                                   \
            ulong B07 = ROLk(T10,  3);                                   \
            ulong B17 = ROLk(T11, 10);                                   \
            ulong B02 = ROLk(T12, 43);                                   \
            ulong B12 = ROLk(T13, 25);                                   \
            ulong B22 = ROLk(T14, 39);                                   \
            ulong B23 = ROLk(T15, 41);                                   \
            ulong B08 = ROLk(T16, 45);                                   \
            ulong B18 = ROLk(T17, 15);                                   \
            ulong B03 = ROLk(T18, 21);                                   \
            ulong B13 = ROLk(T19,  8);                                   \
            ulong B14 = ROLk(T20, 18);                                   \
            ulong B24 = ROLk(T21,  2);                                   \
            ulong B09 = ROLk(T22, 61);                                   \
            ulong B19 = ROLk(T23, 56);                                   \
            ulong B04 = ROLk(T24, 14);                                   \
            A00 = B00 ^ ((~B01) & B02) ^ (RC);                           \
            A01 = B01 ^ ((~B02) & B03);                                  \
            A02 = B02 ^ ((~B03) & B04);                                  \
            A03 = B03 ^ ((~B04) & B00);                                  \
            A04 = B04 ^ ((~B00) & B01);                                  \
            A05 = B05 ^ ((~B06) & B07);                                  \
            A06 = B06 ^ ((~B07) & B08);                                  \
            A07 = B07 ^ ((~B08) & B09);                                  \
            A08 = B08 ^ ((~B09) & B05);                                  \
            A09 = B09 ^ ((~B05) & B06);                                  \
            A10 = B10 ^ ((~B11) & B12);                                  \
            A11 = B11 ^ ((~B12) & B13);                                  \
            A12 = B12 ^ ((~B13) & B14);                                  \
            A13 = B13 ^ ((~B14) & B10);                                  \
            A14 = B14 ^ ((~B10) & B11);                                  \
            A15 = B15 ^ ((~B16) & B17);                                  \
            A16 = B16 ^ ((~B17) & B18);                                  \
            A17 = B17 ^ ((~B18) & B19);                                  \
            A18 = B18 ^ ((~B19) & B15);                                  \
            A19 = B19 ^ ((~B15) & B16);                                  \
            A20 = B20 ^ ((~B21) & B22);                                  \
            A21 = B21 ^ ((~B22) & B23);                                  \
            A22 = B22 ^ ((~B23) & B24);                                  \
            A23 = B23 ^ ((~B24) & B20);                                  \
            A24 = B24 ^ ((~B20) & B21);                                  \
        }

        ROUND(KRC[0]);  ROUND(KRC[1]);  ROUND(KRC[2]);  ROUND(KRC[3]);
        ROUND(KRC[4]);  ROUND(KRC[5]);  ROUND(KRC[6]);  ROUND(KRC[7]);
        ROUND(KRC[8]);  ROUND(KRC[9]);  ROUND(KRC[10]); ROUND(KRC[11]);
        ROUND(KRC[12]); ROUND(KRC[13]); ROUND(KRC[14]); ROUND(KRC[15]);
        ROUND(KRC[16]); ROUND(KRC[17]); ROUND(KRC[18]); ROUND(KRC[19]);
        ROUND(KRC[20]); ROUND(KRC[21]); ROUND(KRC[22]); ROUND(KRC[23]);

        #undef ROUND

        // Truncate output for next step (only first n_lanes matter).
        M[0]  = A00; M[1]  = A01; M[2]  = A02; M[3]  = A03;
        M[4]  = A04; M[5]  = A05; M[6]  = A06; M[7]  = A07;
        M[8]  = A08; M[9]  = A09; M[10] = A10; M[11] = A11;
        M[12] = A12; M[13] = A13; M[14] = A14; M[15] = A15;
    }

    for (uint k = 0u; k < n_lanes; ++k) tips[base + k] = M[k];
}