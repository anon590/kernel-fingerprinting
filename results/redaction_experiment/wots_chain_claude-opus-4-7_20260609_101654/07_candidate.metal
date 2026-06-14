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

// Lane-complement Keccak (Bertoni et al.).
// Persistent complemented set at the *start* of each round (on lanes a..):
//   S = {1, 2, 8, 12, 17}
// After Rho+Pi (b = pi(rho(a))), the complemented set on b-lanes is:
//   pi maps lane index L=x+5y -> L' = y + 5*((2x+3y)%5)
//   1->10, 2->20, 8->21, 12->2, 17->18  => B = {10, 20, 21, 2, 18}
// After Chi (a' = b ^ (~b_next & b_next2)), per row we determine which
// a'-lanes end up complemented. We want a' to again have S = {1,2,8,12,17}
// complemented; we'll prove the round transform achieves that.
//
// Chi row x (lanes b[x,y], b[x+1,y], b[x+2,y] for fixed y, varying x in {0..4}):
//   row y=0: b00,b01,b02,b03,b04   complemented in B: {b02}            -> {2}
//   row y=1: b05,b06,b07,b08,b09   complemented in B: {}                -> {}
//   row y=2: b10,b11,b12,b13,b14   complemented in B: {b10}             -> {10}
//   row y=3: b15,b16,b17,b18,b19   complemented in B: {b18}             -> {18}
//   row y=4: b20,b21,b22,b23,b24   complemented in B: {b20, b21}        -> {20,21}
//
// For each output a'[x,y] = b[x,y] ^ ( (~b[x+1,y]) & b[x+2,y] ):
//   Let f(b) = b for stored-as-normal, f(b) = ~b for stored-as-complement.
//   Real value of b is f(stored). We compute using stored values, adjusting NOTs.
//
// Case analysis per term t = (~B_mid) & B_hi, where B_mid stored = m, B_hi stored = h:
//   If mid not complemented and hi not complemented: t = (~m) & h
//   If mid not complemented and hi complemented:     t = (~m) & ~h  -> need ~h on stored; t_stored = (~m)&(~h), real = same
//   If mid complemented and hi not complemented:     real_mid = ~m, so ~real_mid = m; t = m & h
//   If mid complemented and hi complemented:         t = m & ~h
//
// Output a'[x,y] real = real_lo ^ t. lo stored is l; real_lo = l if lo not complemented, ~l if lo complemented.
// We want to STORE a'[x,y] possibly complemented per the target set S.
// stored_out = real if (x+5y) not in S, else ~real.
//
// Below I enumerate Chi for all 25 outputs with the stored values, choosing
// expressions that naturally produce the desired stored form (avoiding extra NOTs).

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

    // Carry chained truncation (normal form) between sponge calls.
    ulong c0 = seeds[base + 0u];
    ulong c1 = seeds[base + 1u];
    ulong c2 = n2 ? 0ul : seeds[base + 2u];
    ulong c3 = n2 ? 0ul : seeds[base + 3u];

    const ulong ALLONES = 0xFFFFFFFFFFFFFFFFul;

    for (uint step = 0u; step < w; ++step) {
        // Build initial state in NORMAL form, then complement lanes in S = {1,2,8,12,17}.
        // Normal init: a00..a(n_lanes-1) = chained; lane n_lanes byte0 ^= 0x06; lane 16 byte7 ^= 0x80; rest 0.
        ulong a00 = c0;
        ulong a01_n = c1;                                  // normal
        ulong a02_n = n2 ? 0x06ul : c2;                    // normal
        ulong a03   = n2 ? 0ul     : c3;
        ulong a04   = n2 ? 0ul     : 0x06ul;
        ulong a05 = 0, a06 = 0, a07 = 0;
        ulong a08_n = 0;                                   // normal (will complement)
        ulong a09 = 0, a10 = 0, a11 = 0;
        ulong a12_n = 0;                                   // normal (will complement)
        ulong a13 = 0, a14 = 0, a15 = 0;
        ulong a16 = 0x8000000000000000ul;
        ulong a17_n = 0;                                   // normal (will complement)
        ulong a18 = 0, a19 = 0, a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0;

        // Store complemented form for S = {1,2,8,12,17}.
        ulong a01 = ~a01_n;
        ulong a02 = ~a02_n;
        ulong a08 = ~a08_n;
        ulong a12 = ~a12_n;
        ulong a17 = ~a17_n;

        for (uint r = 0u; r < 24u; ++r) {
            // -------- Theta --------
            // Column parities. Some lanes stored as complement; XOR of an odd
            // number of complemented values flips parity. Per column:
            //  col0 (00,05,10,15,20): S-members = {}      -> parity correct
            //  col1 (01,06,11,16,21): S-members = {01}    -> XOR has extra ~ once -> result is ~real_C1
            //  col2 (02,07,12,17,22): S-members = {02,12,17} -> 3 flips -> ~real_C2
            //  col3 (03,08,13,18,23): S-members = {08}    -> ~real_C3
            //  col4 (04,09,14,19,24): S-members = {}      -> correct
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;   // = ~realC1
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;   // = ~realC2
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;   // = ~realC3
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            // D_x = realC_{x-1} ^ ROL(realC_{x+1},1).
            // Using stored: where stored = ~real, both XOR'd-in flips cancel
            // when both stale, or produce a single ~ if only one is stale.
            // D0 = realC4 ^ ROL(realC1,1) = C4 ^ ROL(~C1,1) = C4 ^ ~ROL(C1,1) = ~(C4 ^ ROL(C1,1))
            // D1 = realC0 ^ ROL(realC2,1) = C0 ^ ~ROL(C2,1) = ~(C0 ^ ROL(C2,1))
            // D2 = realC1 ^ ROL(realC3,1) = ~C1 ^ ~ROL(C3,1) = C1 ^ ROL(C3,1)
            // D3 = realC2 ^ ROL(realC4,1) = ~C2 ^ ROL(C4,1) = ~(C2 ^ ROL(C4,1))
            // D4 = realC3 ^ ROL(realC0,1) = ~C3 ^ ROL(C0,1) = ~(C3 ^ ROL(C0,1))
            //
            // We must XOR D into all 5 lanes of each column. For complemented
            // stored lanes (in S), XORing real_D works the same on stored form
            // (since stored = real ^ ALLONES; XOR with D leaves the ALLONES flip).
            // For non-S lanes, XOR with real_D directly. So we need real_D values.
            // Computing ~D for cols 0,1,3,4 means doing the ^ ALLONES once and
            // applying to 5 lanes (= 5 NOTs eliminated by absorbing into D itself).

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

            // -------- Rho + Pi --------
            // After this, complemented set on b-lanes is B = {2, 10, 18, 20, 21}.
            ulong b00 = a00;
            ulong b10 = ROL(a01,  1);   // from a01 (complemented) -> b10 complemented
            ulong b20 = ROL(a02, 62);   // from a02 (complemented) -> b20 complemented
            ulong b05 = ROL(a03, 28);
            ulong b15 = ROL(a04, 27);
            ulong b16 = ROL(a05, 36);
            ulong b01 = ROL(a06, 44);
            ulong b11 = ROL(a07,  6);
            ulong b21 = ROL(a08, 55);   // from a08 -> b21 complemented
            ulong b06 = ROL(a09, 20);
            ulong b07 = ROL(a10,  3);
            ulong b17 = ROL(a11, 10);
            ulong b02 = ROL(a12, 43);   // from a12 -> b02 complemented
            ulong b12 = ROL(a13, 25);
            ulong b22 = ROL(a14, 39);
            ulong b23 = ROL(a15, 41);
            ulong b08 = ROL(a16, 45);
            ulong b18 = ROL(a17, 15);   // from a17 -> b18 complemented
            ulong b03 = ROL(a18, 21);
            ulong b13 = ROL(a19,  8);
            ulong b14 = ROL(a20, 18);
            ulong b24 = ROL(a21,  2);
            ulong b09 = ROL(a22, 61);
            ulong b19 = ROL(a23, 56);
            ulong b04 = ROL(a24, 14);

            // -------- Chi (lane-complement form) --------
            // For each row, output a'[i] real = b[i]_real ^ ( (~b[i+1]_real) & b[i+2]_real ).
            // We want stored a' with S = {1,2,8,12,17} complemented.
            //
            // Row y=0: lanes 0..4. B-complemented: {b02}. S-target on this row: {1, 2}.
            //   real a00 = real_b00 ^ (~real_b01 & real_b02)
            //            = b00 ^ (~b01 & ~b02) = b00 ^ ((~b01) & ~b02)
            //     stored target: a00 (normal) = real
            //     => a00 = b00 ^ ((~b01) & ~b02)
            //   real a01 = real_b01 ^ (~real_b02 & real_b03)
            //            = b01 ^ (~~b02 & b03) = b01 ^ (b02 & b03)
            //     stored target: a01 (complemented) = ~real
            //     => a01 = ~(b01 ^ (b02 & b03)) = ~b01 ^ (b02 & b03)  -- one NOT
            //   real a02 = real_b02 ^ (~real_b03 & real_b04)
            //            = ~b02 ^ ((~b03) & b04)
            //     stored target: a02 complemented = ~real
            //     => a02 = b02 ^ ((~b03) & b04)         -- the ~ on b02 cancels with target ~
            //   real a03 = real_b03 ^ (~real_b04 & real_b00)
            //            = b03 ^ ((~b04) & b00)
            //     stored: normal
            //     => a03 = b03 ^ ((~b04) & b00)
            //   real a04 = real_b04 ^ (~real_b00 & real_b01)
            //            = b04 ^ ((~b00) & b01)
            //     stored: normal
            //     => a04 = b04 ^ ((~b00) & b01)
            //
            // Row y=1: lanes 5..9. B-complemented: {}. S-target on this row: {8}.
            //   a05 = b05 ^ ((~b06) & b07)
            //   a06 = b06 ^ ((~b07) & b08)
            //   a07 = b07 ^ ((~b08) & b09)
            //   a08 stored complemented = ~(b08 ^ ((~b09) & b05))
            //                            = ~b08 ^ ((~b09) & b05)
            //   a09 = b09 ^ ((~b05) & b06)
            //
            // Row y=2: lanes 10..14. B-complemented: {b10}. S-target: {12}.
            //   real a10 = ~b10 ^ ((~b11) & b12)
            //     stored: normal = real => a10 = (~b10) ^ ((~b11) & b12)
            //   real a11 = b11 ^ (~~b12_wait no, mid here is b12 which is NOT complemented)
            //     Wait: for row y=2 the b's involved are b10,b11,b12,b13,b14. Only b10 is complemented.
            //   a11 real = real_b11 ^ (~real_b12 & real_b13) = b11 ^ ((~b12) & b13)   stored normal
            //   a12 real = b12 ^ ((~b13) & b14)               stored complemented => a12 = ~b12 ^ ((~b13)&b14)
            //   a13 real = b13 ^ ((~b14) & real_b10) = b13 ^ ((~b14) & ~b10)         stored normal
            //   a14 real = b14 ^ ((~real_b10) & b11) = b14 ^ (b10 & b11)              stored normal
            //
            // Row y=3: lanes 15..19. B-complemented: {b18}. S-target: {17}.
            //   a15 real = b15 ^ ((~b16) & b17)                  stored normal
            //   a16 real = b16 ^ ((~b17) & real_b18) = b16 ^ ((~b17) & ~b18)
            //   a17 real = b17 ^ ((~real_b18) & b19) = b17 ^ (b18 & b19)
            //            stored complemented => a17 = ~b17 ^ (b18 & b19)
            //   a18 real = real_b18 ^ ((~b19) & b15) = ~b18 ^ ((~b19) & b15)        stored normal
            //   a19 real = b19 ^ ((~b15) & b16)                  stored normal
            //
            // Row y=4: lanes 20..24. B-complemented: {b20, b21}. S-target: {}.
            //   a20 real = ~b20 ^ ((~real_b21) & b22) = ~b20 ^ (b21 & b22)
            //            stored normal
            //   a21 real = real_b21 ^ (~b22 & b23) = ~b21 ^ ((~b22) & b23)
            //            stored normal
            //   a22 real = b22 ^ ((~b23) & b24)                   stored normal
            //   a23 real = b23 ^ ((~b24) & real_b20) = b23 ^ ((~b24) & ~b20)
            //   a24 real = b24 ^ ((~real_b20) & real_b21) = b24 ^ (b20 & ~b21)
            //            = b24 ^ (b20 & ~b21)   -- one NOT

            a00 = b00 ^ ((~b01) & (~b02));
            a01 = (~b01) ^ (b02 & b03);
            a02 = b02 ^ ((~b03) & b04);
            a03 = b03 ^ ((~b04) & b00);
            a04 = b04 ^ ((~b00) & b01);

            a05 = b05 ^ ((~b06) & b07);
            a06 = b06 ^ ((~b07) & b08);
            a07 = b07 ^ ((~b08) & b09);
            a08 = (~b08) ^ ((~b09) & b05);
            a09 = b09 ^ ((~b05) & b06);

            a10 = (~b10) ^ ((~b11) & b12);
            a11 = b11 ^ ((~b12) & b13);
            a12 = (~b12) ^ ((~b13) & b14);
            a13 = b13 ^ ((~b14) & (~b10));
            a14 = b14 ^ (b10 & b11);

            a15 = b15 ^ ((~b16) & b17);
            a16 = b16 ^ ((~b17) & (~b18));
            a17 = (~b17) ^ (b18 & b19);
            a18 = (~b18) ^ ((~b19) & b15);
            a19 = b19 ^ ((~b15) & b16);

            a20 = (~b20) ^ (b21 & b22);
            a21 = (~b21) ^ ((~b22) & b23);
            a22 = b22 ^ ((~b23) & b24);
            a23 = b23 ^ ((~b24) & (~b20));
            a24 = b24 ^ (b20 & (~b21));

            // Iota: XOR with RC affects only a00 (not complemented). Safe.
            a00 ^= KECCAK_RC[r];
        }

        // Undo complements on lanes we need to read out (a00..a(n_lanes-1)).
        // S members within first 4 lanes: a01, a02 always; a03 never.
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