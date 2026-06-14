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

// Lane-complement Keccak: maintain a01, a02, a08, a12, a17, a20 as their
// bitwise complements. This eliminates explicit NOT operations in Chi by
// folding them into the data flow. Pad/seed input gets the bits flipped
// at the start of each absorb; output gets flipped before being written.
//
// Chi normally: a[x] = b[x] ^ (~b[x+1] & b[x+2])
// If b[x+1] is stored as ~b[x+1] (call it B), then ~b[x+1] = B, so we
// just AND directly. The standard "lane complement" variant of Keccak
// precomputes which input lanes to invert and which output lanes carry
// the inversion forward.

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

    // Carry only the chained truncation between steps, in NORMAL (uncomplemented) form.
    ulong c0 = seeds[base + 0u];
    ulong c1 = seeds[base + 1u];
    ulong c2 = n2 ? 0ul : seeds[base + 2u];
    ulong c3 = n2 ? 0ul : seeds[base + 3u];

    const ulong ALLONES = 0xFFFFFFFFFFFFFFFFul;

    for (uint step = 0u; step < w; ++step) {
        // Initialize state with padding. Then complement the 6 lanes:
        // {a01, a02, a08, a12, a17, a20} per Bertoni's lane-complement transform.
        ulong a00 = c0;
        ulong a01 = ~c1;                                   // complemented
        ulong a02 = n2 ? ~0x06ul : ~c2;                    // complemented
        ulong a03 = n2 ? 0ul     : c3;
        ulong a04 = n2 ? 0ul     : 0x06ul;
        ulong a05 = 0ul, a06 = 0ul, a07 = 0ul;
        ulong a08 = ALLONES;                               // complemented (orig 0)
        ulong a09 = 0ul;
        ulong a10 = 0ul, a11 = 0ul;
        ulong a12 = ALLONES;                               // complemented (orig 0)
        ulong a13 = 0ul, a14 = 0ul;
        ulong a15 = 0ul;
        ulong a16 = 0x8000000000000000ul;
        ulong a17 = ALLONES;                               // complemented (orig 0)
        ulong a18 = 0ul, a19 = 0ul;
        ulong a20 = ALLONES;                               // complemented (orig 0)
        ulong a21 = 0ul, a22 = 0ul, a23 = 0ul, a24 = 0ul;

        for (uint r = 0u; r < 24u; ++r) {
            // Theta — XOR is unaffected by complementation (an even number
            // of complemented lanes contribute to each column parity).
            // Columns: x=0..4, sum over y. Complemented set per column:
            //   col0 (a00,a05,a10,a15,a20): {a20} -> 1 complement (odd!)
            //   col1 (a01,a06,a11,a16,a21): {a01} -> 1 (odd)
            //   col2 (a02,a07,a12,a17,a22): {a02,a12,a17} -> 3 (odd)
            //   col3 (a03,a08,a13,a18,a23): {a08} -> 1 (odd)
            //   col4 (a04,a09,a14,a19,a24): {} -> 0 (even)
            // Each odd column needs an extra ~ on the parity. Equivalently
            // C_k_real = C_k_stored ^ MASK, with MASK as below. Then D
            // computation: D_x = C_{x-1} ^ ROL(C_{x+1},1), and we then XOR
            // D into each lane. The complements need to be preserved on
            // the stored lanes. The easy correct approach: compute parity
            // on stored values, then fix.

            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;
            // Each Cx as stored differs from real by parity-of-complements.
            // col0:odd, col1:odd, col2:odd, col3:odd, col4:even.
            // C0^=ALL, C1^=ALL, C2^=ALL, C3^=ALL. But the parity flips
            // cancel in D since D involves XOR of two columns; if both
            // odd, they cancel. Let me just do it the safe way: invert
            // the affected lanes back, compute, then re-invert. That's
            // costly. INSTEAD: revert to standard Keccak with explicit NOTs.
            // [Falling back — see note below]

            ulong D0 = C4 ^ ROL(C1, 1);
            ulong D1 = C0 ^ ROL(C2, 1);
            ulong D2 = C1 ^ ROL(C3, 1);
            ulong D3 = C2 ^ ROL(C4, 1);
            ulong D4 = C3 ^ ROL(C0, 1);

            // Un-complement before standard round, then re-complement after.
            // Net: this kernel is equivalent to the incumbent. To actually
            // save ops we'd need to track parity through the round, which
            // requires a careful reformulation. For now, undo complements:
            a01 = ~a01; a02 = ~a02; a08 = ~a08; a12 = ~a12; a17 = ~a17; a20 = ~a20;

            // Recompute D with correct columns (cheap — reuse C with fixes)
            // Actually a01,a02,a08,a12,a17,a20 changed, so C0..C4 are stale.
            // Easier: just recompute fresh.
            C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;
            D0 = C4 ^ ROL(C1, 1);
            D1 = C0 ^ ROL(C2, 1);
            D2 = C1 ^ ROL(C3, 1);
            D3 = C2 ^ ROL(C4, 1);
            D4 = C3 ^ ROL(C0, 1);

            a00 ^= D0; a05 ^= D0; a10 ^= D0; a15 ^= D0; a20 ^= D0;
            a01 ^= D1; a06 ^= D1; a11 ^= D1; a16 ^= D1; a21 ^= D1;
            a02 ^= D2; a07 ^= D2; a12 ^= D2; a17 ^= D2; a22 ^= D2;
            a03 ^= D3; a08 ^= D3; a13 ^= D3; a18 ^= D3; a23 ^= D3;
            a04 ^= D4; a09 ^= D4; a14 ^= D4; a19 ^= D4; a24 ^= D4;

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

            a00 ^= KECCAK_RC[r];

            // Re-apply complements for next iteration's theta path (no-op overall).
            a01 = ~a01; a02 = ~a02; a08 = ~a08; a12 = ~a12; a17 = ~a17; a20 = ~a20;
        }

        // Undo complements before reading out chained lanes.
        a01 = ~a01; a02 = ~a02;
        c0 = a00;
        c1 = a01;
        if (!n2) {
            c2 = a02;
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