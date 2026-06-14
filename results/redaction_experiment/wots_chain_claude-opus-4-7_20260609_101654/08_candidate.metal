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

// Bertoni lane-complement Keccak round.
// Persistent complemented set S = {1, 2, 8, 12, 17} on a-lanes between rounds.
// Saves ~5 NOTs per round in Chi by tracking which lanes are stored as complement.

inline ulong ROL(ulong x, uint k) { return rotate(x, (ulong)k); }

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
    bool n2 = (n_lanes == 2u);

    // Carry chained truncation between sponge calls (NORMAL form).
    ulong c0 = seeds[base + 0u];
    ulong c1 = seeds[base + 1u];
    ulong c2 = n2 ? 0ul : seeds[base + 2u];
    ulong c3 = n2 ? 0ul : seeds[base + 3u];

    for (uint step = 0u; step < w; ++step) {
        // Build initial state. Real lanes:
        //   a0..a(n_lanes-1) = chained
        //   lane n_lanes byte0 = 0x06
        //   lane 16 byte7 = 0x80
        //   rest = 0
        // Stored form: S = {1,2,8,12,17} are complemented (XOR with all-ones).
        // For zero-real lanes in S, stored = ~0. For nonzero real lanes in S, stored = ~real.

        ulong a00 = c0;
        ulong a01 = ~c1;              // S
        ulong a02, a03, a04;
        if (n2) {
            a02 = ~0x06ul;            // S, real=0x06
            a03 = 0ul;
            a04 = 0ul;
        } else {
            a02 = ~c2;                // S
            a03 = c3;
            a04 = 0x06ul;
        }
        ulong a05 = 0, a06 = 0, a07 = 0;
        ulong a08 = ~0ul;             // S, real=0
        ulong a09 = 0, a10 = 0, a11 = 0;
        ulong a12 = ~0ul;             // S, real=0
        ulong a13 = 0, a14 = 0, a15 = 0;
        ulong a16 = 0x8000000000000000ul;
        ulong a17 = ~0ul;             // S, real=0
        ulong a18 = 0, a19 = 0;
        ulong a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

        // Unrolled 24 rounds.
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 24u; ++r) {
            // ---- Theta ----
            // Column parity. Number of S-lanes per column:
            //  col0 (00,05,10,15,20): 0  -> C0 = real
            //  col1 (01,06,11,16,21): 1 (a01)  -> C1 = ~real
            //  col2 (02,07,12,17,22): 3 (a02,a12,a17) -> C2 = ~real
            //  col3 (03,08,13,18,23): 1 (a08) -> C3 = ~real
            //  col4 (04,09,14,19,24): 0  -> C4 = real
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            // D_x_real = realC_{x-1} ^ ROL(realC_{x+1}, 1)
            // D0_real = realC4 ^ ROL(realC1,1) = C4 ^ ~ROL(C1,1) = ~(C4 ^ ROL(C1,1))
            // D1_real = realC0 ^ ROL(realC2,1) = C0 ^ ~ROL(C2,1) = ~(C0 ^ ROL(C2,1))
            // D2_real = realC1 ^ ROL(realC3,1) = ~C1 ^ ~ROL(C3,1) = C1 ^ ROL(C3,1)
            // D3_real = realC2 ^ ROL(realC4,1) = ~C2 ^ ROL(C4,1) = ~(C2 ^ ROL(C4,1))
            // D4_real = realC3 ^ ROL(realC0,1) = ~C3 ^ ROL(C0,1) = ~(C3 ^ ROL(C0,1))
            //
            // When XORing D into stored lanes: XOR with D_real is same on stored form
            // (since stored = real ^ K, K constant). So we can use D_real directly.
            // To avoid extra NOTs, pre-complement D where convenient — actually we can
            // just compute D as written and XOR. The NOTs collapse with subsequent ops.
            // Trick: instead of computing ~X then XORing into 5 lanes, XOR the all-ones
            // into ONE lane per column (we choose to absorb it into the column parity
            // contribution). But simpler: just compute D_real and XOR.

            ulong D0 = ~(C4 ^ ROL(C1, 1));
            ulong D1 = ~(C0 ^ ROL(C2, 1));
            ulong D2 =   C1 ^ ROL(C3, 1);
            ulong D3 = ~(C2 ^ ROL(C4, 1));
            ulong D4 = ~(C3 ^ ROL(C0, 1));

            a00 ^= D0; a05 ^= D0; a10 ^= D0; a15 ^= D0; a20 ^= D0;
            a01 ^= D1; a06 ^= D1; a11 ^= D1; a16 ^= D1; a21 ^= D1;
            a02 ^= D2; a07 ^= D2; a12 ^= D2; a17 ^= D2; a22 ^= D2;
            a03 ^= D3; a08 ^= D3; a13 ^= D3; a18 ^= D3; a23 ^= D3;
            a04 ^= D4; a09 ^= D4; a14 ^= D4; a19 ^= D4; a24 ^= D4;

            // ---- Rho + Pi ----
            // Source lane stored-complement set: S = {1,2,8,12,17}.
            // Destination b-lane complement set B = pi(S) = {10,20,21,2,18}.
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

            // ---- Chi ----
            // For each output a'[i,y] we want stored form with S-set complemented.
            // Computation derived in derivation comments (verified by previous attempt).
            //
            // Row 0: B-comp on this row = {b02}. S-target on a-row = {1,2}.
            a00 = b00 ^ ((~b01) & (~b02));
            a01 = (~b01) ^ (b02 & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            // Row 1: B-comp = {}. S-target = {8}.
            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = (~b08) ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            // Row 2: B-comp = {b10}. S-target = {12}.
            a10 = (~b10) ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = (~b12) ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & (~b10));
            a14 = b14 ^ (b10 & b11);

            // Row 3: B-comp = {b18}. S-target = {17}.
            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & (~b18));
            a17 = (~b17) ^ (b18 & b19);
            a18 = (~b18) ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            // Row 4: B-comp = {b20, b21}. S-target = {}.
            a20 = (~b20) ^ (b21 & b22);
            a21 = (~b21) ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & (~b20));
            a24 = b24 ^ (b20 & (~b21));

            // ---- Iota ----
            a00 ^= KECCAK_RC[r];
        }

        // Decomplement output lanes we need (a00..a(n_lanes-1)).
        // a00 normal, a01 complemented, a02 complemented, a03 normal.
        c0 = a00;
        c1 = ~a01;
        if (!n2) {
            c2 = ~a02;
            c3 = a03;
        }
    }

    tips[base + 0u] = c0;
    tips[base + 1u] = c1;
    if (!n2) {
        tips[base + 2u] = c2;
        tips[base + 3u] = c3;
    }
}