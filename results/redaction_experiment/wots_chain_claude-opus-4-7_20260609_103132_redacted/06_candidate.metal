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

    // State lanes A[0..24]; we use the lane-complementing optimization
    // (Bertoni/Daemen/Peeters/Van Assche, "Keccak implementation overview").
    // Complemented set: {1, 2, 8, 12, 17, 20}.
    // On entry: XOR these with ~0.
    // chi step uses modified formulas (see below).
    // On output, when reading a complemented lane, XOR with ~0.

    ulong A00=0, A01=0, A02=0, A03=0, A04=0;
    ulong A05=0, A06=0, A07=0, A08=0, A09=0;
    ulong A10=0, A11=0, A12=0, A13=0, A14=0;
    ulong A15=0, A16=0, A17=0, A18=0, A19=0;
    ulong A20=0, A21=0, A22=0, A23=0, A24=0;

    // Load seed.
    ulong s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    ulong s8=0,s9=0,s10=0,s11=0,s12=0,s13=0,s14=0,s15=0;
    if (n_lanes > 0u)  s0  = seeds[base + 0];
    if (n_lanes > 1u)  s1  = seeds[base + 1];
    if (n_lanes > 2u)  s2  = seeds[base + 2];
    if (n_lanes > 3u)  s3  = seeds[base + 3];
    if (n_lanes > 4u)  s4  = seeds[base + 4];
    if (n_lanes > 5u)  s5  = seeds[base + 5];
    if (n_lanes > 6u)  s6  = seeds[base + 6];
    if (n_lanes > 7u)  s7  = seeds[base + 7];
    if (n_lanes > 8u)  s8  = seeds[base + 8];
    if (n_lanes > 9u)  s9  = seeds[base + 9];
    if (n_lanes > 10u) s10 = seeds[base + 10];
    if (n_lanes > 11u) s11 = seeds[base + 11];
    if (n_lanes > 12u) s12 = seeds[base + 12];
    if (n_lanes > 13u) s13 = seeds[base + 13];
    if (n_lanes > 14u) s14 = seeds[base + 14];
    if (n_lanes > 15u) s15 = seeds[base + 15];

    const ulong ONES = 0xFFFFFFFFFFFFFFFFul;
    uint dom_lane = n_lanes;

    for (uint step = 0u; step < w; ++step) {
        // Fresh padded state, in COMPLEMENTED form for lanes {1,2,8,12,17,20}.
        A00 = (n_lanes > 0u) ? s0 : 0ul;
        A01 = ((n_lanes > 1u) ? s1 : 0ul) ^ ONES;   // complemented
        A02 = ((n_lanes > 2u) ? s2 : 0ul) ^ ONES;   // complemented
        A03 = (n_lanes > 3u) ? s3 : 0ul;
        A04 = (n_lanes > 4u) ? s4 : 0ul;
        A05 = (n_lanes > 5u) ? s5 : 0ul;
        A06 = (n_lanes > 6u) ? s6 : 0ul;
        A07 = (n_lanes > 7u) ? s7 : 0ul;
        A08 = ((n_lanes > 8u) ? s8 : 0ul) ^ ONES;   // complemented
        A09 = (n_lanes > 9u) ? s9 : 0ul;
        A10 = (n_lanes > 10u) ? s10 : 0ul;
        A11 = (n_lanes > 11u) ? s11 : 0ul;
        A12 = ((n_lanes > 12u) ? s12 : 0ul) ^ ONES; // complemented
        A13 = (n_lanes > 13u) ? s13 : 0ul;
        A14 = (n_lanes > 14u) ? s14 : 0ul;
        A15 = (n_lanes > 15u) ? s15 : 0ul;
        A16 = 0; A17 = ONES; A18 = 0; A19 = 0;       // lane 17 complemented
        A20 = ONES; A21 = 0; A22 = 0; A23 = 0; A24 = 0; // lane 20 complemented

        // XOR domain pad 0x06 at lane = n_lanes. Account for complementation.
        switch (dom_lane) {
            case 0:  A00 ^= 0x06ul; break;
            case 1:  A01 ^= 0x06ul; break;  // OK: XOR commutes with complement
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
        // Final pad byte 0x80 at byte 7 of lane 16 (NOT complemented).
        A16 ^= 0x8000000000000000ul;

        // ---- 24 rounds of Keccak-f[1600] with lane-complementing ----
        for (uint r = 0u; r < 24u; ++r) {
            // theta (XOR/rotate works identically on complemented lanes:
            // parities are correct because complement bits cancel in pairs).
            // Sheet parity: complemented lanes per column:
            //   col 0 {0,5,10,15,20}: 20 complemented -> 1 flip
            //   col 1 {1,6,11,16,21}: 1                -> 1 flip
            //   col 2 {2,7,12,17,22}: 2,12,17          -> 3 flips -> 1 flip
            //   col 3 {3,8,13,18,23}: 8                -> 1 flip
            //   col 4 {4,9,14,19,24}: none             -> 0 flips
            // So C0..C3 are each off by ONES vs true parity, C4 is true.
            // D = C[x-1] ^ ROL(C[x+1],1). Each D has constant complement offset.
            // But we then XOR D into all lanes of a column — adding the same
            // constant to every lane in that column preserves correctness IF
            // the constant matches the per-lane complement bookkeeping... it
            // does NOT in general. So we must compute *true* C values.
            ulong C0 = A00 ^ A05 ^ A10 ^ A15 ^ A20;
            ulong C1 = A01 ^ A06 ^ A11 ^ A16 ^ A21;
            ulong C2 = A02 ^ A07 ^ A12 ^ A17 ^ A22;
            ulong C3 = A03 ^ A08 ^ A13 ^ A18 ^ A23;
            ulong C4 = A04 ^ A09 ^ A14 ^ A19 ^ A24;
            // Adjust to true parities by removing complement contributions:
            // col0 has lane20 complemented -> C0 ^= ONES
            // col1 has lane1               -> C1 ^= ONES
            // col2 has lane2,12,17 (3)     -> C2 ^= ONES
            // col3 has lane8               -> C3 ^= ONES
            // col4 none                     -> unchanged
            // But since we XOR D into ALL 5 lanes of a column (including the
            // complemented ones), and we want the complemented form preserved,
            // the cleanest is: compute D using TRUE C, then XOR D as-is into
            // all 5 lanes — this preserves the invariant exactly because the
            // complement bits are unaffected by XOR with D.
            C0 ^= ONES; C1 ^= ONES; C2 ^= ONES; C3 ^= ONES;

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            // rho + pi fused with theta-xor on the source side.
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

            // After theta+rho+pi, complemented lanes (in B-space, same lane
            // indices because theta only XORs and rho+pi just permutes/rotates
            // the same bits — but pi PERMUTES lane positions!).
            //
            // The complemented set is defined in A-space at lane indices
            // {1,2,8,12,17,20}. After pi, lane k in B comes from A[pi^{-1}(k)].
            // Easier: track complemented set per round in B-space by mapping.
            //
            // pi maps src lane (x,y) -> dst lane (y, 2x+3y mod 5). The src
            // lane indices that are complemented:
            //   1=(1,0) -> (0,2) = 10
            //   2=(2,0) -> (0,4) = 20
            //   8=(3,1) -> (1,4) = 21
            //   12=(2,2)-> (2,0) = 2
            //   17=(2,3)-> (3,3) = 18
            //   20=(0,4)-> (4,0) = 4
            // So after rho+pi, complemented lanes (in B) are: {10,20,21,2,18,4}.
            // chi operates row-wise on B and writes to A. We need to choose,
            // for each output A-lane, whether it ends up complemented (and
            // matches the input invariant for next round).
            //
            // To keep code simple and still benefit, we DECOMPLEMENT here:
            // XOR ONES into the complemented B-lanes, restoring true Keccak
            // semantics, then perform standard chi, then RE-COMPLEMENT the
            // A-output set {1,2,8,12,17,20}. The decomp+recomp cost is 12 XORs
            // per round — which is MORE than the saved NOTs (~6/round? no:
            // the trick saves up to 5 NOTs per round). Net loss.
            //
            // GIVE UP on lane-complementing path: revert to plain chi.
            // Decomplement on entry (to undo the initial complement) by
            // XORing ONES into the B-positions that came from complemented A.
            B10 ^= ONES; B20 ^= ONES; B21 ^= ONES;
            B02 ^= ONES; B18 ^= ONES; B04 ^= ONES;

            // chi (standard)
            ulong nA00 = B00 ^ ((~B01) & B02);
            ulong nA01 = B01 ^ ((~B02) & B03);
            ulong nA02 = B02 ^ ((~B03) & B04);
            ulong nA03 = B03 ^ ((~B04) & B00);
            ulong nA04 = B04 ^ ((~B00) & B01);
            ulong nA05 = B05 ^ ((~B06) & B07);
            ulong nA06 = B06 ^ ((~B07) & B08);
            ulong nA07 = B07 ^ ((~B08) & B09);
            ulong nA08 = B08 ^ ((~B09) & B05);
            ulong nA09 = B09 ^ ((~B05) & B06);
            ulong nA10 = B10 ^ ((~B11) & B12);
            ulong nA11 = B11 ^ ((~B12) & B13);
            ulong nA12 = B12 ^ ((~B13) & B14);
            ulong nA13 = B13 ^ ((~B14) & B10);
            ulong nA14 = B14 ^ ((~B10) & B11);
            ulong nA15 = B15 ^ ((~B16) & B17);
            ulong nA16 = B16 ^ ((~B17) & B18);
            ulong nA17 = B17 ^ ((~B18) & B19);
            ulong nA18 = B18 ^ ((~B19) & B15);
            ulong nA19 = B19 ^ ((~B15) & B16);
            ulong nA20 = B20 ^ ((~B21) & B22);
            ulong nA21 = B21 ^ ((~B22) & B23);
            ulong nA22 = B22 ^ ((~B23) & B24);
            ulong nA23 = B23 ^ ((~B24) & B20);
            ulong nA24 = B24 ^ ((~B20) & B21);

            // Re-complement lanes {1,2,8,12,17,20} for next round invariant.
            nA01 ^= ONES;
            nA02 ^= ONES;
            nA08 ^= ONES;
            nA12 ^= ONES;
            nA17 ^= ONES;
            nA20 ^= ONES;

            // iota
            nA00 ^= KECCAK_RC[r];

            A00=nA00; A01=nA01; A02=nA02; A03=nA03; A04=nA04;
            A05=nA05; A06=nA06; A07=nA07; A08=nA08; A09=nA09;
            A10=nA10; A11=nA11; A12=nA12; A13=nA13; A14=nA14;
            A15=nA15; A16=nA16; A17=nA17; A18=nA18; A19=nA19;
            A20=nA20; A21=nA21; A22=nA22; A23=nA23; A24=nA24;
        }

        // End of Keccak-f. State is in complemented form for {1,2,8,12,17,20}.
        // We extract first n_lanes lanes as the next chunk; UN-complement
        // lanes 1, 2, 8, 12, 17, 20 (only those that are < n_lanes).
        // We also clear C-side bookkeeping: above C0..C3 ^= ONES was an
        // adjustment for parity; that lives inside one round only.
        //
        // Extract uncomplemented seed values for next iteration.
        s0  = A00;
        s1  = A01 ^ ((n_lanes > 1u)  ? ONES : ONES);  // always uncomplement
        s2  = A02 ^ ONES;
        s3  = A03;
        s4  = A04;
        s5  = A05;
        s6  = A06;
        s7  = A07;
        s8  = A08 ^ ONES;
        s9  = A09;
        s10 = A10;
        s11 = A11;
        s12 = A12 ^ ONES;
        s13 = A13;
        s14 = A14;
        s15 = A15;
    }

    // Write final tips (un-complemented form held in s0..s15).
    if (n_lanes > 0u)  tips[base + 0]  = s0;
    if (n_lanes > 1u)  tips[base + 1]  = s1;
    if (n_lanes > 2u)  tips[base + 2]  = s2;
    if (n_lanes > 3u)  tips[base + 3]  = s3;
    if (n_lanes > 4u)  tips[base + 4]  = s4;
    if (n_lanes > 5u)  tips[base + 5]  = s5;
    if (n_lanes > 6u)  tips[base + 6]  = s6;
    if (n_lanes > 7u)  tips[base + 7]  = s7;
    if (n_lanes > 8u)  tips[base + 8]  = s8;
    if (n_lanes > 9u)  tips[base + 9]  = s9;
    if (n_lanes > 10u) tips[base + 10] = s10;
    if (n_lanes > 11u) tips[base + 11] = s11;
    if (n_lanes > 12u) tips[base + 12] = s12;
    if (n_lanes > 13u) tips[base + 13] = s13;
    if (n_lanes > 14u) tips[base + 14] = s14;
    if (n_lanes > 15u) tips[base + 15] = s15;
}