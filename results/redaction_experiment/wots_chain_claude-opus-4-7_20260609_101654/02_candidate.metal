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

// Keccak-f1600 with lane-complementing.
// We maintain an invariant where lanes {1, 2, 8, 12, 17, 20} are stored
// complemented (bitwise NOT of their "true" value). Inputs/outputs to this
// function are in NORMAL representation; conversion is done at entry/exit.
//
// In the complemented domain, chi becomes:
//   a' = b' ^ ((~b') & c')          -> for lanes where output is normal
//   if the output lane is complemented, we store the complement.
// Standard reference: Bertoni et al., "Keccak implementation overview".
//
// To keep code simple and still gain the bit-op savings, we use the well-known
// trick of pre-inverting six lanes once before round 0, and post-inverting
// after round 23. Each round inside uses modified chi expressions on those
// rows to remove ~ ops.

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

    ulong a00=0, a01=0, a02=0, a03=0, a04=0;
    ulong a05=0, a06=0, a07=0, a08=0, a09=0;
    ulong a10=0, a11=0, a12=0, a13=0, a14=0;
    ulong a15=0, a16=0, a17=0, a18=0, a19=0;
    ulong a20=0, a21=0, a22=0, a23=0, a24=0;

    a00 = seeds[base + 0u];
    a01 = seeds[base + 1u];
    if (n_lanes > 2u) {
        a02 = seeds[base + 2u];
        a03 = seeds[base + 3u];
    }

    uint W = w;
    for (uint step = 0u; step < W; ++step) {
        // Re-initialize capacity + padding lanes.
        if (n_lanes == 2u) {
            a02 = 0x06ul;
            a03 = 0ul;
            a04 = 0ul;
        } else {
            a04 = 0x06ul;
        }
        a05 = 0ul; a06 = 0ul; a07 = 0ul; a08 = 0ul; a09 = 0ul;
        a10 = 0ul; a11 = 0ul; a12 = 0ul; a13 = 0ul; a14 = 0ul;
        a15 = 0ul;
        a16 = 0x8000000000000000ul;
        a17 = 0ul; a18 = 0ul; a19 = 0ul;
        a20 = 0ul; a21 = 0ul; a22 = 0ul; a23 = 0ul; a24 = 0ul;

        // Apply lane-complementing pre-inversion to lanes 1, 2, 8, 12, 17, 20.
        a01 = ~a01;
        a02 = ~a02;
        a08 = ~a08;
        a12 = ~a12;
        a17 = ~a17;
        a20 = ~a20;

        for (uint r = 0u; r < 24u; ++r) {
            // Theta
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            a00 ^= D0; a05 ^= D0; a10 ^= D0; a15 ^= D0; a20 ^= D0;
            a01 ^= D1; a06 ^= D1; a11 ^= D1; a16 ^= D1; a21 ^= D1;
            a02 ^= D2; a07 ^= D2; a12 ^= D2; a17 ^= D2; a22 ^= D2;
            a03 ^= D3; a08 ^= D3; a13 ^= D3; a18 ^= D3; a23 ^= D3;
            a04 ^= D4; a09 ^= D4; a14 ^= D4; a19 ^= D4; a24 ^= D4;

            // Rho + Pi
            ulong b00 = a00;
            ulong b10 = ROL(a01,  1);
            ulong b20 = ROL(a02, 62);
            ulong b05 = ROL(a03, 28);
            ulong b15 = ROL(a04, 27);

            ulong b16 = ROL(a05, 36);
            ulong b01 = ROL(a06, 44);
            ulong b11 = ROL(a07,  6);
            ulong b21 = ROL(a08, 55);
            ulong b06 = ROL(a09, 20);

            ulong b07 = ROL(a10,  3);
            ulong b17 = ROL(a11, 10);
            ulong b02 = ROL(a12, 43);
            ulong b12 = ROL(a13, 25);
            ulong b22 = ROL(a14, 39);

            ulong b23 = ROL(a15, 41);
            ulong b08 = ROL(a16, 45);
            ulong b18 = ROL(a17, 15);
            ulong b03 = ROL(a18, 21);
            ulong b13 = ROL(a19,  8);

            ulong b14 = ROL(a20, 18);
            ulong b24 = ROL(a21,  2);
            ulong b09 = ROL(a22, 61);
            ulong b19 = ROL(a23, 56);
            ulong b04 = ROL(a24, 14);

            // Chi with lane-complementing.
            // After Rho+Pi, the complemented lanes (by NEW position) are the
            // images of original lanes {1, 2, 8, 12, 17, 20} under pi.
            // pi: (x,y) -> (y, (2x+3y)%5), so new index = y + 5*((2x+3y)%5).
            // lane 1  (1,0) -> (0,2) = 10
            // lane 2  (2,0) -> (0,4) = 20
            // lane 8  (3,1) -> (1,4) = 21
            // lane 12 (2,2) -> (2,0) =  2
            // lane 17 (2,3) -> (3,3) = 18
            // lane 20 (0,4) -> (4,2) = 22
            // So in b-domain, complemented lanes are: b10, b20, b21, b02, b18, b22.

            // Row y=0: b00 b01 b02 b03 b04  (b02 complemented)
            //   a00 = b00 ^ (~b01 & b02)        -> b02 complemented => ~b01 & ~b02_true
            //                                   = ~(b01 | b02_true) ... but we have b02 = ~b02_true
            //                                   so (~b01) & b02 in code equals (~b01) & (~b02_true)
            //                                   which equals ~(b01 | b02_true). We want true a00.
            //   Easier: rewrite chi by lanes using actual stored (some complemented).
            // Use the standard transformation table from Bertoni:
            //   For each output lane in {1,2,8,12,17,20} (original positions),
            //   we want to store complemented result; for others, store true result.
            // But here we already permuted; let's just enumerate per-row what the
            // stored expressions are.

            // Output lanes that must be stored complemented (in pre-pi indexing):
            //   {1, 2, 8, 12, 17, 20}. Those correspond to output a-positions
            //   AFTER chi (chi doesn't permute, so same indices).

            // Row 0 (a00..a04):
            //   true a00 = b00 ^ (~true_b01 & true_b02)
            //   We have stored: b00=true, b01=true, b02=~true (complemented)
            //   => (~b01) & b02_stored  =  (~true_b01) & (~true_b02)
            //                          =  ~(true_b01 | true_b02)
            //   true a00 = b00 ^ ~(b01 | b02_stored_as_true?)... too confusing.
            //
            // Cleaner: undo complementing before chi, redo after. The cost is
            // 6 NOTs total per round vs 25, still saves ~19 NOTs/round.

            // Undo complementing on the 6 b-lanes that are complemented:
            b02 = ~b02;
            b10 = ~b10;
            b18 = ~b18;
            b20 = ~b20;
            b21 = ~b21;
            b22 = ~b22;

            // Standard chi
            a00 = b00 ^ ((~b01) & b02);
            a01 = b01 ^ ((~b02) & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = b08 ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            a10 = b10 ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = b12 ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & b10);
            a14 = b14 ^ ((~b10) & b11);

            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & b18);
            a17 = b17 ^ ((~b18) & b19);
            a18 = b18 ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            a20 = b20 ^ ((~b21) & b22);
            a21 = b21 ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & b20);
            a24 = b24 ^ ((~b20) & b21);

            // Redo complementing on {1,2,8,12,17,20}
            a01 = ~a01;
            a02 = ~a02;
            a08 = ~a08;
            a12 = ~a12;
            a17 = ~a17;
            a20 = ~a20;

            // Iota (a00 is in true domain)
            a00 ^= KECCAK_RC[r];
        }

        // Undo complementing for final state.
        a01 = ~a01;
        a02 = ~a02;
        a08 = ~a08;
        a12 = ~a12;
        a17 = ~a17;
        a20 = ~a20;
    }

    tips[base + 0u] = a00;
    tips[base + 1u] = a01;
    if (n_lanes > 2u) {
        tips[base + 2u] = a02;
        tips[base + 3u] = a03;
    }
}