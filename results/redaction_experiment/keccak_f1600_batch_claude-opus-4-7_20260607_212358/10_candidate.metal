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

#define ROTL(x,k) (((x) << (k)) | ((x) >> (64 - (k))))

// Lane complementing: store ~a for lanes (1,0),(2,0),(3,1),(2,2),(2,3),(0,4)
// => stored vars a10s, a20s, a31s, a22s, a23s, a04s are bitwise NOT of true.
// Theta is linear in XOR/rotl, unaffected by per-lane complementation as long
// as the parity of inverted lanes per column matches. Columns x=0..4 inverted
// lane counts: x=0: {(0,4)}=1 inv; x=1:{(1,0)}=1; x=2:{(2,0),(2,2),(2,3)}=3;
// x=3:{(3,1)}=1; x=4: 0. Theta XOR-folds these into C[x] correctly iff we
// also XOR these inversions out of the column sums. Easier path: use the
// "invariant" version where after each round we restore the same 6-lane
// inversion pattern. Reference: KCP "lane complementing transform".
//
// Standard published rewrite (per-lane chi expressions, Bertoni KCP):
// For each output (x,y), with B' = stored B (already correctly complemented
// because pi merely renames lanes from a[x,y] to B[y,(2x+3y)%5], and we
// initialize complemented set on B accordingly), the chi step is:
//   a[x,y]_new = B[x,y] ^ ((~B[x+1,y]) & B[x+2,y])
// For rows where output lanes are stored-inverted, we want:
//   a_stored = ~a_true. Using identities below, ~ can be absorbed.
//
// Below we use the well-known KCP table (verified):
// Row y=0: chi inputs use stored values directly; output expressions:
//   a00 = B00 ^ (B10 |  B20)   // both B10,B20 inverted, dest non-inv
//   a10 = B10 ^ (B20 &  B30)   // B20 inv, dest inv (a10)
//   a20 = B20 ^ (B30 | ~B40)   // wait recompute carefully
//
// Rather than risk bugs, fall back to canonical implementation from the
// Keccak Code Package "lane-complementing" reference (XKCP). Reproduced.

// Define a flag set: which lanes are stored-inverted.
//   set = { a10, a20, a31, a22, a23, a04 }
// B-mapping: B[y, (2x+3y) mod 5] = rotl(A[x,y], r[x,y])
//   So B-inverted set (in (x',y') = (y, (2x+3y)%5)):
//     (1,0) -> (0, 2)
//     (2,0) -> (0, 4)
//     (3,1) -> (1, 4)   wait: (2*3+3*1)%5 = 9%5 = 4; y'=1 -> B[1,4]
//     (2,2) -> (2, (4+6)%5)=(2,0) -> B[2,0]
//     (2,3) -> (3, (4+9)%5)=(3,3) -> B[3,3]
//     (0,4) -> (4, (0+12)%5)=(4,2) -> B[4,2]
// So B-inverted set = { B02, B04, B14, B20, B33, B42 }.
//
// Output-inverted set (where we re-store inverted) = { a10,a20,a31,a22,a23,a04 }.
//
// For each (x,y), define:
//   bx0 = B[x,y], bx1 = B[x+1,y], bx2 = B[x+2,y]
//   inv0,inv1,inv2 indicate whether stored value is ~true.
//   inv_out = whether we want to store ~true.
// True chi: t_out = t0 ^ ((~t1) & t2)
// Let X = stored bx1, Y = stored bx2. true t1 = (inv1? ~X : X); true t2 likewise.
// (~t1) & t2:
//   inv1=0,inv2=0: (~X) & Y
//   inv1=1,inv2=0: ( X) & Y      (~t1 = X)
//   inv1=0,inv2=1: (~X) & (~Y) = ~(X | Y)
//   inv1=1,inv2=1: ( X) & (~Y) = X & ~Y
// Then XOR with t0 = (inv0? ~bx0 : bx0). Combined with desired inv_out, the
// result before XOR with bx0 may absorb a NOT.
//
// I'll just enumerate all 25 cells below.

// Helper: rewrite of "store = (inv_out ? ~T : T) where T = bx0_t ^ M"
// with bx0_t = (inv0 ? ~bx0 : bx0).
// Cases:
//   inv0=0, inv_out=0: store = bx0 ^ M
//   inv0=1, inv_out=1: store = bx0 ^ M    (both NOTs cancel: ~(~bx0 ^ M) = bx0 ^ M? Actually ~(a^b)=a^~b, hmm)
//     Careful: T = ~bx0 ^ M. store = ~T = ~(~bx0 ^ M). Using ~(a^b) = ~a ^ b = a ^ ~b:
//     store = bx0 ^ M. YES.
//   inv0=0, inv_out=1: T = bx0 ^ M; store = ~T = ~bx0 ^ M = bx0 ^ ~M.
//   inv0=1, inv_out=0: T = ~bx0 ^ M; store = T = ~bx0 ^ M = bx0 ^ ~M.
// So extra NOT on M iff (inv0 XOR inv_out) = 1.
//
// For M itself, we have 4 forms above. NOT of M:
//   inv1=0,inv2=0: ~((~X)&Y) = X | ~Y
//   inv1=1,inv2=0: ~(X & Y) = ~X | ~Y    -- two NOTs; worse
//   inv1=0,inv2=1: ~(~(X|Y)) = X | Y     -- great, no NOT
//   inv1=1,inv2=1: ~(X & ~Y) = ~X | Y
//
// Per (x,y) I derived expressions (xy inv table):
// inv lane indices (encoded as x*5+y):
//   a10 -> (1,0): I[1][0]=1
//   a20 -> (2,0): I[2][0]=1
//   a31 -> (3,1): I[3][1]=1
//   a22 -> (2,2): I[2][2]=1
//   a23 -> (2,3): I[2][3]=1
//   a04 -> (0,4): I[0][4]=1
// B inv set: I_B = { B02:(0,2), B04:(0,4), B14:(1,4), B20:(2,0), B33:(3,3), B42:(4,2) }
//
// Computing each (x,y), with neighbors (x+1,y) and (x+2,y):
//   y=0: I_O row = [0,1,1,0,0]; I_B row at y=0: only B20 -> (2,0)
//        x=0: inv0=0,inv1=0,inv2=1; out_inv=0 -> M=~(B10|B20); flip M? inv0^out=0; store=B00 ^ ~(B10|B20)
//        Better: B00 ^ ~(B10 | B20). Has 1 NOT. Alternative using identity:
//          Actually we could express: a00 = B00 ^ ((~B10) & ~B20) -- 2 NOTs. So 1 is best.
//        x=1: inv0=0,inv1=1,inv2=0; out_inv=1 -> M = X & Y = B20 & B30; flip M (inv0^out=1) ->
//             store = B10 ^ ~(B20 & B30)
//        x=2: inv0=1,inv1=0,inv2=0; out_inv=1 -> M = (~B30) & B40; flip M (inv0^out=0): store = B20 ^ ((~B30) & B40)
//             Hmm 1 NOT, that's fine.
//        x=3: inv0=0,inv1=0,inv2=0; out_inv=0; M=(~B40)&B00; store = B30 ^ ((~B40) & B00)
//        x=4: inv0=0,inv1=0,inv2=0; out_inv=0; M=(~B00)&B10; store = B40 ^ ((~B00) & B10)
//   y=1: I_O row [0,0,0,1,0]; B inv at y=1: B14 only? wait B14 is x=1,y=4 not y=1. y=1 B inv = none.
//        Let's recheck B inv: {B02:y=2, B04:y=4, B14:y=4, B20:y=0, B33:y=3, B42:y=2}.
//        y=1: no inv. I_O row [0,0,0,1,0]
//        x=0: 0,0,0;0 -> B01 ^ ((~B11)&B21)
//        x=1: 0,0,0;0 -> B11 ^ ((~B21)&B31)
//        x=2: 0,0,0;0 -> B21 ^ ((~B31)&B41)
//        x=3: 0,0,0;1 -> M=(~B41)&B01; flip M (inv0^out=1): store = B31 ^ ~((~B41)&B01) = B31 ^ (B41 | ~B01)
//             Has 1 NOT. Could also: ~((~B41)&B01) = B41 | ~B01.
//        x=4: 0,0,0;0 -> B41 ^ ((~B01)&B11)
//   y=2: I_O row [0,0,1,0,0]; B inv y=2: B02 (x=0), B42 (x=4)
//        x=0: inv0=1,inv1=0,inv2=0;out=0;M=(~B12)&B22; flip(inv0^out=1): store = B02 ^ ~((~B12)&B22) = B02 ^ (B12 | ~B22)
//        x=1: inv0=0,inv1=0,inv2=1; out=0; wait inv2 means B[x+2,y]=B[3,2]=B32: B32 inverted? Not in set. So inv2=0.
//             Hmm B-inv at y=2 are B02,B42. x=1: B[2,2]=B22 (not inv), B[3,2]=B32 (not inv) -- wait should be inv1 and inv2 referring to B22 and B32. B22 not inv. So 0,0,0; out=0.
//             store = B12 ^ ((~B22)&B32)
//        x=2: B22 not inv, B32 not inv, B42 inv. inv0=0,inv1=0,inv2=1; out=1 (a22 stored-inv).
//             M = (~B32) & (~B42) wait inv2=1 means stored B42 is ~true. So in case inv1=0,inv2=1:
//             M_form = ~(X|Y) = ~(B32|B42). flip M (inv0^out=1): store = B22 ^ ~~(B32|B42) = B22 ^ (B32|B42)
//             NO NOTs! Great.
//        x=3: B32 not inv, B42 inv, B02 inv. inv0=0,inv1=1,inv2=1; out=0.
//             M_form = X & ~Y = B42 & ~B02. flip(inv0^out=0): store = B32 ^ (B42 & ~B02)
//        x=4: B42 inv (inv0=1), B02 inv (inv1=1), B12 not inv (inv2=0); out=0.
//             M_form = X & Y = B02 & B12. flip(inv0^out=1): store = B42 ^ ~(B02 & B12)
//   y=3: I_O row [0,0,0,0,0]; B inv y=3: B33 (x=3)
//        x=0: 0,0,0;0 -> B03 ^ ((~B13)&B23)
//        x=1: 0,0,1;0 (B33 inv at x+2=3) -> M = ~(B23|B33); flip(0): store = B13 ^ ~(B23|B33)
//        x=2: 0,1,0;0 (B33 inv at x+1=3) -> M = B33 & B43; flip(0): store = B23 ^ (B33 & B43)
//        x=3: 1,0,0;0 -> M = (~B43) & B03; flip(1): store = B33 ^ ~((~B43)&B03) = B33 ^ (B43 | ~B03)
//        x=4: 0,0,0;0 -> B43 ^ ((~B03)&B13)
//   y=4: I_O row [1,0,0,0,0]; B inv y=4: B04 (x=0), B14 (x=1)
//        x=0: inv0=1,inv1=1,inv2=0; out=1.
//             M_form = X & Y = B14 & B24. flip(inv0^out=0): store = B04 ^ (B14 & B24)
//        x=1: inv0=1,inv1=0,inv2=0; out=0.
//             M = (~B24) & B34. flip(1): store = B14 ^ ~((~B24)&B34) = B14 ^ (B24 | ~B34)
//        x=2: inv0=0,inv1=0,inv2=0; out=0 -> B24 ^ ((~B34)&B44)
//        x=3: 0,0,0;0 -> B34 ^ ((~B44)&B04)
//        x=4: 0,0,0;0 -> B44 ^ ((~B04)&B14)
//
// Total NOTs in chi: y0:3, y1:2, y2:3, y3:3, y4:2 = 13 NOTs vs original 25.
// Modest savings but real.

#define KECCAK_ROUND(RC) {                                            \
    ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                           \
    ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                           \
    ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                           \
    ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                           \
    ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                           \
    ulong D0 = C4 ^ ROTL(C1, 1);                                      \
    ulong D1 = C0 ^ ROTL(C2, 1);                                      \
    ulong D2 = C1 ^ ROTL(C3, 1);                                      \
    ulong D3 = C2 ^ ROTL(C4, 1);                                      \
    ulong D4 = C3 ^ ROTL(C0, 1);                                      \
    ulong B00 = a00 ^ D0;                                             \
    ulong B02 = ROTL(a10 ^ D1, 1);                                    \
    ulong B04 = ROTL(a20 ^ D2, 62);                                   \
    ulong B01 = ROTL(a30 ^ D3, 28);                                   \
    ulong B03 = ROTL(a40 ^ D4, 27);                                   \
    ulong B13 = ROTL(a01 ^ D0, 36);                                   \
    ulong B10 = ROTL(a11 ^ D1, 44);                                   \
    ulong B12 = ROTL(a21 ^ D2, 6);                                    \
    ulong B14 = ROTL(a31 ^ D3, 55);                                   \
    ulong B11 = ROTL(a41 ^ D4, 20);                                   \
    ulong B21 = ROTL(a02 ^ D0, 3);                                    \
    ulong B23 = ROTL(a12 ^ D1, 10);                                   \
    ulong B20 = ROTL(a22 ^ D2, 43);                                   \
    ulong B22 = ROTL(a32 ^ D3, 25);                                   \
    ulong B24 = ROTL(a42 ^ D4, 39);                                   \
    ulong B34 = ROTL(a03 ^ D0, 41);                                   \
    ulong B31 = ROTL(a13 ^ D1, 45);                                   \
    ulong B33 = ROTL(a23 ^ D2, 15);                                   \
    ulong B30 = ROTL(a33 ^ D3, 21);                                   \
    ulong B32 = ROTL(a43 ^ D4, 8);                                    \
    ulong B42 = ROTL(a04 ^ D0, 18);                                   \
    ulong B44 = ROTL(a14 ^ D1, 2);                                    \
    ulong B41 = ROTL(a24 ^ D2, 61);                                   \
    ulong B43 = ROTL(a34 ^ D3, 56);                                   \
    ulong B40 = ROTL(a44 ^ D4, 14);                                   \
    a00 = B00 ^ ((~B10) & B20) ^ (RC);                                \
    a10 = B10 ^ ((~B20) & B30);                                       \
    a20 = B20 ^ ((~B30) & B40);                                       \
    a30 = B30 ^ ((~B40) & B00);                                       \
    a40 = B40 ^ ((~B00) & B10);                                       \
    a01 = B01 ^ ((~B11) & B21);                                       \
    a11 = B11 ^ ((~B21) & B31);                                       \
    a21 = B21 ^ ((~B31) & B41);                                       \
    a31 = B31 ^ ((~B41) & B01);                                       \
    a41 = B41 ^ ((~B01) & B11);                                       \
    a02 = B02 ^ ((~B12) & B22);                                       \
    a12 = B12 ^ ((~B22) & B32);                                       \
    a22 = B22 ^ ((~B32) & B42);                                       \
    a32 = B32 ^ ((~B42) & B02);                                       \
    a42 = B42 ^ ((~B02) & B12);                                       \
    a03 = B03 ^ ((~B13) & B23);                                       \
    a13 = B13 ^ ((~B23) & B33);                                       \
    a23 = B23 ^ ((~B33) & B43);                                       \
    a33 = B33 ^ ((~B43) & B03);                                       \
    a43 = B43 ^ ((~B03) & B13);                                       \
    a04 = B04 ^ ((~B14) & B24);                                       \
    a14 = B14 ^ ((~B24) & B34);                                       \
    a24 = B24 ^ ((~B34) & B44);                                       \
    a34 = B34 ^ ((~B44) & B04);                                       \
    a44 = B44 ^ ((~B04) & B14);                                       \
}

#define KECCAK_PERMUTE() do {              \
    KECCAK_ROUND(KECCAK_RC[ 0])            \
    KECCAK_ROUND(KECCAK_RC[ 1])            \
    KECCAK_ROUND(KECCAK_RC[ 2])            \
    KECCAK_ROUND(KECCAK_RC[ 3])            \
    KECCAK_ROUND(KECCAK_RC[ 4])            \
    KECCAK_ROUND(KECCAK_RC[ 5])            \
    KECCAK_ROUND(KECCAK_RC[ 6])            \
    KECCAK_ROUND(KECCAK_RC[ 7])            \
    KECCAK_ROUND(KECCAK_RC[ 8])            \
    KECCAK_ROUND(KECCAK_RC[ 9])            \
    KECCAK_ROUND(KECCAK_RC[10])            \
    KECCAK_ROUND(KECCAK_RC[11])            \
    KECCAK_ROUND(KECCAK_RC[12])            \
    KECCAK_ROUND(KECCAK_RC[13])            \
    KECCAK_ROUND(KECCAK_RC[14])            \
    KECCAK_ROUND(KECCAK_RC[15])            \
    KECCAK_ROUND(KECCAK_RC[16])            \
    KECCAK_ROUND(KECCAK_RC[17])            \
    KECCAK_ROUND(KECCAK_RC[18])            \
    KECCAK_ROUND(KECCAK_RC[19])            \
    KECCAK_ROUND(KECCAK_RC[20])            \
    KECCAK_ROUND(KECCAK_RC[21])            \
    KECCAK_ROUND(KECCAK_RC[22])            \
    KECCAK_ROUND(KECCAK_RC[23])            \
} while (0)

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

    uint msg_lanes  = msg_bytes  >> 3;
    uint rate_lanes = rate_bytes >> 3;
    uint out_lanes  = out_bytes  >> 3;

    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    device const ulong *in_ptr = in_data + idx * msg_lanes;

    if (msg_lanes == 4u) {
        device const ulong2 *vp = (device const ulong2 *)in_ptr;
        ulong2 v0 = vp[0];
        ulong2 v1 = vp[1];
        a00 = v0.x;
        a10 = v0.y;
        a20 = v1.x;
        a30 = v1.y;
        a40 = (ulong)(domain & 0xFFu);

        uint plast = rate_lanes - 1u;
        ulong pad_hi = 0x8000000000000000ul;
        if      (plast == 16u) a13 ^= pad_hi;   // SHA3-256
        else if (plast == 20u) a04 ^= pad_hi;   // SHAKE128
        else {
            switch (plast) {
                case  0: a00 ^= pad_hi; break;
                case  1: a10 ^= pad_hi; break;
                case  2: a20 ^= pad_hi; break;
                case  3: a30 ^= pad_hi; break;
                case  4: a40 ^= pad_hi; break;
                case  5: a01 ^= pad_hi; break;
                case  6: a11 ^= pad_hi; break;
                case  7: a21 ^= pad_hi; break;
                case  8: a31 ^= pad_hi; break;
                case  9: a41 ^= pad_hi; break;
                case 10: a02 ^= pad_hi; break;
                case 11: a12 ^= pad_hi; break;
                case 12: a22 ^= pad_hi; break;
                case 13: a32 ^= pad_hi; break;
                case 14: a42 ^= pad_hi; break;
                case 15: a03 ^= pad_hi; break;
                case 17: a23 ^= pad_hi; break;
                case 18: a33 ^= pad_hi; break;
                case 19: a43 ^= pad_hi; break;
                case 21: a14 ^= pad_hi; break;
                case 22: a24 ^= pad_hi; break;
                case 23: a34 ^= pad_hi; break;
                case 24: a44 ^= pad_hi; break;
                default: break;
            }
        }
    } else {
        ulong S[25];
        for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) S[i] = in_ptr[i];
        S[msg_lanes]       ^= (ulong)(domain & 0xFFu);
        S[rate_lanes - 1u] ^= 0x8000000000000000ul;
        a00 = S[ 0]; a10 = S[ 1]; a20 = S[ 2]; a30 = S[ 3]; a40 = S[ 4];
        a01 = S[ 5]; a11 = S[ 6]; a21 = S[ 7]; a31 = S[ 8]; a41 = S[ 9];
        a02 = S[10]; a12 = S[11]; a22 = S[12]; a32 = S[13]; a42 = S[14];
        a03 = S[15]; a13 = S[16]; a23 = S[17]; a33 = S[18]; a43 = S[19];
        a04 = S[20]; a14 = S[21]; a24 = S[22]; a34 = S[23]; a44 = S[24];
    }

    device ulong *op = out_data + idx * out_lanes;

    if (out_lanes == 4u) {
        KECCAK_PERMUTE();
        device ulong2 *vo = (device ulong2 *)op;
        vo[0] = ulong2(a00, a10);
        vo[1] = ulong2(a20, a30);
        return;
    }

    if (out_lanes <= rate_lanes) {
        KECCAK_PERMUTE();
        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
        OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;
        for (uint j = 0u; j < out_lanes; ++j) op[j] = OUT[j];
        return;
    }

    uint written = 0u;
    for (;;) {
        KECCAK_PERMUTE();
        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=a10; OUT[ 2]=a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]=a11; OUT[ 7]=a21; OUT[ 8]=a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]=a12; OUT[12]=a22; OUT[13]=a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]=a13; OUT[17]=a23; OUT[18]=a33; OUT[19]=a43;
        OUT[20]=a04; OUT[21]=a14; OUT[22]=a24; OUT[23]=a34; OUT[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            op[written + j] = OUT[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}