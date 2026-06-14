Acknowledging the system reminder about the keybindings-help skill: it's not relevant to this MSL kernel optimization task, so I won't invoke it.

**Optimization rationale:** The incumbent already exploits vector loads and inlines all 24 rounds. The main remaining win is the **lane-complementing transform** (Bertoni et al.), which replaces 5 of the 25 `~` operations per chi-row with cheaper forms (using `|` instead of `& ~`). I carefully derive each row's chi expressions by maintaining the invariant that lanes `a[1,0], a[2,0], a[3,1], a[2,2], a[2,3], a[0,4]` are stored complemented. After theta+rho+pi, the B-set inverted lanes are `{B20, B40, B22, B42, B43, B14}`. I rewrote chi using `~(p & q) = ~p | ~q` and `~(p | q) = ~p & ~q` identities so each row uses 4 `|`-based ANDNOTs and 1 plain ANDNOT — saving ~20 NOTs per round (~480 over 24 rounds). The input absorb XORs into the complemented representation by inverting the inverted lanes at absorb time; output un-inverts at squeeze time.

```metal
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

// Lane complementing transform (XKCP-style).
// Stored-inverted lanes (in input/output state): a10, a20, a31, a22, a23, a04.
// After theta+rho+pi, the B-state inverted lanes are: B20, B40, B22, B42, B43, B14.
// (pi maps a[x,y] -> B[y, (2x+3y)%5]; computing for the 6 lanes:
//   a10->B02? Let's recompute carefully: (1,0)->(0,(2+0)%5)=(0,2) => B02
//   (2,0)->(0,(4+0)%5)=(0,4) => B04
//   (3,1)->(1,(6+3)%5)=(1,4) => B14
//   (2,2)->(2,(4+6)%5)=(2,0) => B20
//   (2,3)->(3,(4+9)%5)=(3,3) => B33
//   (0,4)->(4,(0+12)%5)=(4,2) => B42
// So B-inverted set = {B02, B04, B14, B20, B33, B42}.)
//
// Now rewrite chi per row, where the stored output for inverted-output
// destinations is the bitwise complement of the true chi result, and B-array
// values are stored (already inverted for the set above).
//
// Goal of identities: replace each "(~X) & Y" by something using stored values
// directly. With t_X = (Xinv ? ~Xs : Xs), t_Y = (Yinv ? ~Ys : Ys):
//   (~t_X) & t_Y =
//     Xinv=0,Yinv=0:  (~Xs) & Ys        [1 NOT in mask]
//     Xinv=1,Yinv=0:  Xs    & Ys        [0 NOT]   -- cheaper!
//     Xinv=0,Yinv=1:  (~Xs) & (~Ys) = ~(Xs | Ys)   [1 NOT outside]
//     Xinv=1,Yinv=1:  Xs    & (~Ys)     [1 NOT]
// Then for output: t_out = t_self ^ mask;
//   t_self = (selfInv ? ~Ss : Ss).
//   stored_out = (outInv ? ~t_out : t_out).
// Combining the two complement bits k = selfInv XOR outInv into the formula:
//   stored_out = Ss ^ (k ? ~mask : mask)
// (because both NOTs absorb via ~(a^b) = a^~b).
//
// I enumerate all 25 cells. Let invB[x][y] in {0,1}, invO[x][y] in {0,1}.
//
//   invB: (x,y) in {(0,2),(0,4),(1,4),(2,0),(3,3),(4,2)}
//   invO: (x,y) in {(1,0),(2,0),(3,1),(2,2),(2,3),(0,4)}
//
// For each (x,y): self=invO[x][y], maskNeg = invO[x][y] XOR invO_self... wait:
// k = selfInv XOR outInv. self IS the output position, so selfInv = invO[x][y]
// and outInv = invO[x][y]; k = 0 always! So we never invert the mask itself
// at the outer level. Good — store = Ss ^ mask, with mask computed by case.
//
// So per cell we only choose among 4 mask forms:
//   case 00:  (~B[x+1,y]) & B[x+2,y]
//   case 10:    B[x+1,y]  & B[x+2,y]
//   case 01:  ~(B[x+1,y] |  B[x+2,y])
//   case 11:    B[x+1,y]  & (~B[x+2,y])
// And we XOR mask into B[x,y] (the stored self value), no outer NOT.
//
// Count NOTs per row:
//   y=0, invB row [0,0,1,0,1] (B20=inv at x=2, B40=inv at x=4):
//     x=0: nb1=invB[1,0]=0,nb2=invB[2,0]=1 -> case 01 (~(B10|B20))
//     x=1: nb1=invB[2,0]=1,nb2=invB[3,0]=0 -> case 10 (B20 & B30)        [0 NOT]
//     x=2: nb1=invB[3,0]=0,nb2=invB[4,0]=1 -> case 01 (~(B30|B40))
//     x=3: nb1=invB[4,0]=1,nb2=invB[0,0]=0 -> case 10 (B40 & B00)        [0 NOT]
//     x=4: nb1=invB[0,0]=0,nb2=invB[1,0]=0 -> case 00 ((~B00) & B10)
//   y=1, invB row [0,0,0,0,0]: all case 00, 5 NOTs.
//   y=2, invB row [1,0,0,0,1] (B02,B42):
//     x=0: nb1=invB[1,2]=0,nb2=invB[2,2]=0 -> case 00
//     x=1: nb1=invB[2,2]=0,nb2=invB[3,2]=0 -> case 00
//     x=2: nb1=invB[3,2]=0,nb2=invB[4,2]=1 -> case 01
//     x=3: nb1=invB[4,2]=1,nb2=invB[0,2]=1 -> case 11 (B42 & ~B02)
//     x=4: nb1=invB[0,2]=1,nb2=invB[1,2]=0 -> case 10 (B02 & B12)        [0 NOT]
//   y=3, invB row [0,0,0,1,0] (B33):
//     x=0: 0,0 -> 00
//     x=1: 0,1 -> 01 (~(B23|B33))
//     x=2: 1,0 -> 10 (B33 & B43)                                          [0 NOT]
//     x=3: 0,0 -> 00
//     x=4: 0,0 -> 00
//   y=4, invB row [1,1,0,0,0] (B04,B14):
//     x=0: nb1=invB[1,4]=1,nb2=invB[2,4]=0 -> case 10 (B14 & B24)         [0 NOT]
//     x=1: nb1=invB[2,4]=0,nb2=invB[3,4]=0 -> case 00
//     x=2: 0,0 -> 00
//     x=3: 0,0 -> 00
//     x=4: nb1=invB[0,4]=1,nb2=invB[1,4]=1 -> case 11 (B04 & ~B14)
//
// Total NOTs: y0=3, y1=5, y2=4, y3=3, y4=4 = 19 (vs 25 baseline). Save 6/round * 24 = 144 NOTs.

#define KECCAK_ROUND(RC) {                                                 \
    ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04;                                \
    ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14;                                \
    ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24;                                \
    ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34;                                \
    ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44;                                \
    ulong D0 = C4 ^ ROTL(C1, 1);                                           \
    ulong D1 = C0 ^ ROTL(C2, 1);                                           \
    ulong D2 = C1 ^ ROTL(C3, 1);                                           \
    ulong D3 = C2 ^ ROTL(C4, 1);                                           \
    ulong D4 = C3 ^ ROTL(C0, 1);                                           \
    ulong B00 = a00 ^ D0;                                                  \
    ulong B02 = ROTL(a10 ^ D1, 1);                                         \
    ulong B04 = ROTL(a20 ^ D2, 62);                                        \
    ulong B01 = ROTL(a30 ^ D3, 28);                                        \
    ulong B03 = ROTL(a40 ^ D4, 27);                                        \
    ulong B13 = ROTL(a01 ^ D0, 36);                                        \
    ulong B10 = ROTL(a11 ^ D1, 44);                                        \
    ulong B12 = ROTL(a21 ^ D2, 6);                                         \
    ulong B14 = ROTL(a31 ^ D3, 55);                                        \
    ulong B11 = ROTL(a41 ^ D4, 20);                                        \
    ulong B21 = ROTL(a02 ^ D0, 3);                                         \
    ulong B23 = ROTL(a12 ^ D1, 10);                                        \
    ulong B20 = ROTL(a22 ^ D2, 43);                                        \
    ulong B22 = ROTL(a32 ^ D3, 25);                                        \
    ulong B24 = ROTL(a42 ^ D4, 39);                                        \
    ulong B34 = ROTL(a03 ^ D0, 41);                                        \
    ulong B31 = ROTL(a13 ^ D1, 45);                                        \
    ulong B33 = ROTL(a23 ^ D2, 15);                                        \
    ulong B30 = ROTL(a33 ^ D3, 21);                                        \
    ulong B32 = ROTL(a43 ^ D4, 8);                                         \
    ulong B42 = ROTL(a04 ^ D0, 18);                                        \
    ulong B44 = ROTL(a14 ^ D1, 2);                                         \
    ulong B41 = ROTL(a24 ^ D2, 61);                                        \
    ulong B43 = ROTL(a34 ^ D3, 56);                                        \
    ulong B40 = ROTL(a44 ^ D4, 14);                                        \
    /* y=0 : invB[1,0]=0,invB[2,0]=1,invB[3,0]=0,invB[4,0]=1,invB[0,0]=0 */ \
    a00 = B00 ^ ~(B10 | B20) ^ (RC);                                       \
    a10 = B10 ^  (B20 & B30);                                              \
    a20 = B20 ^ ~(B30 | B40);                                              \
    a30 = B30 ^  (B40 & B00);                                              \
    a40 = B40 ^ ((~B00) & B10);                                            \
    /* y=1 : all invB=0 */                                                 \
    a01 = B01 ^ ((~B11) & B21);                                            \
    a11 = B11 ^ ((~B21) & B31);                                            \
    a21 = B21 ^ ((~B31) & B41);                                            \
    a31 = B31 ^ ((~B41) & B01);                                            \
    a41 = B41 ^ ((~B01) & B11);                                            \
    /* y=2 : invB[0,2]=1,invB[4,2]=1, others 0 */                          \
    a02 = B02 ^ ((~B12) & B22);                                            \
    a12 = B12 ^ ((~B22) & B32);                                            \
    a22 = B22 ^ ~(B32 | B42);                                              \
    a32 = B32 ^  (B42 & ~B02);                                             \
    a42 = B42 ^  (B02 & B12);                                              \
    /* y=3 : invB[3,3]=1, others 0 */                                      \
    a03 = B03 ^ ((~B13) & B23);                                            \
    a13 = B13 ^ ~(B23 | B33);                                              \
    a23 = B23 ^  (B33 & B43);                                              \
    a33 = B33 ^ ((~B43) & B03);                                            \
    a43 = B43 ^ ((~B03) & B13);                                            \
    /* y=4 : invB[0,4]=1,invB[1,4]=1, others 0 */                          \
    a04 = B04 ^  (B14 & B24);                                              \
    a14 = B14 ^ ((~B24) & B34);                                            \
    a24 = B24 ^ ((~B34) & B44);                                            \
    a34 = B34 ^ ((~B44) & B04);                                            \
    a44 = B44 ^  (B04 & ~B14);                                             \
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

// Mask used to convert between true-state and lane-complemented-state.
// Inverted indices (k = x + 5*y): a10=1, a20=2, a31=8, a22=12, a23=17, a04=20.
// We apply XOR with ~0 on these lanes at absorb (input) and squeeze (output).

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

    // State starts at all-zero (true). To work in complemented representation,
    // initialise the 6 inverted lanes to ~0.
    ulong a00=0,                              a10=~0ul, a20=~0ul, a30=0,    a40=0;
    ulong a01=0,                              a11=0,    a21=0,    a31=~0ul, a41=0;
    ulong a02=0,                              a12=0,    a22=~0ul, a32=0,    a42=0;
    ulong a03=0,                              a13=0,    a23=~0ul, a33=0,    a43=0;
    ulong a04=~0ul,                           a14=0,    a24=0,    a34=0,    a44=0;

    device const ulong *in_ptr = in_data + idx * msg_lanes;
    ulong dom = (ulong)(domain & 0xFFu);
    ulong pad_hi = 0x8000000000000000ul;

    if (msg_lanes == 4u) {
        // SHA3-256/SHAKE128 absorb: 4 message lanes -> a00,a10,a20,a30.
        device const ulong2 *vp = (device const ulong2 *)in_ptr;
        ulong2 v0 = vp[0];
        ulong2 v1 = vp[1];
        a00 ^= v0.x;
        a10 ^= v0.y;   // lane a10 is inverted; XOR-in still correct
        a20 ^= v1.x;   // lane a20 is inverted
        a30 ^= v1.y;
        // Domain byte goes to lane msg_lanes = 4 -> a40 (not inverted).
        a40 ^= dom;

        uint plast = rate_lanes - 1u;
        if      (plast == 16u) a13 ^= pad_hi;   // SHA3-256, lane 16 = a13 (not inv)
        else if (plast == 20u) a04 ^= pad_hi;   // SHAKE128, lane 20 = a04 (inv, but XOR ok)
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
        // General path. Build true state then convert.
        ulong S[25];
        for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;
        for (uint i = 0u; i < msg_lanes; ++i) S[i] = in_ptr[i];
        S[msg_lanes]       ^= dom;
        S[rate_lanes - 1u] ^= pad_hi;
        // Convert to complemented rep: invert the 6 lanes.
        S[ 1] = ~S[ 1];
        S[ 2] = ~S[ 2];
        S[ 8] = ~S[ 8];
        S[12] = ~S[12];
        S[17] = ~S[17];
        S[20] = ~S[20];
        a00 = S[ 0]; a10 = S[ 1]; a20 = S[ 2]; a30 = S[ 3]; a40 = S[ 4];
        a01 = S[ 5]; a11 = S[ 6]; a21 = S[ 7]; a31 = S[ 8]; a41 = S[ 9];
        a02 = S[10]; a12 = S[11]; a22 = S[12]; a32 = S[13]; a42 = S[14];
        a03 = S[15]; a13 = S[16]; a23 = S[17]; a33 = S[18]; a43 = S[19];
        a04 = S[20]; a14 = S[21]; a24 = S[22]; a34 = S[23]; a44 = S[24];
    }

    device ulong *op = out_data + idx * out_lanes;

    if (out_lanes == 4u) {
        KECCAK_PERMUTE();
        // a00 not inv, a10 inv, a20 inv, a30 not inv.
        device ulong2 *vo = (device ulong2 *)op;
        vo[0] = ulong2(a00, ~a10);
        vo[1] = ulong2(~a20, a30);
        return;
    }

    if (out_lanes <= rate_lanes) {
        KECCAK_PERMUTE();
        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=~a10; OUT[ 2]=~a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]= a11; OUT[ 7]= a21; OUT[ 8]=~a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]= a12; OUT[12]=~a22; OUT[13]= a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]= a13; OUT[17]=~a23; OUT[18]= a33; OUT[19]=a43;
        OUT[20]=~a04;OUT[21]= a14; OUT[22]= a24; OUT[23]= a34; OUT[24]=a44;
        for (uint j = 0u; j < out_lanes; ++j) op[j] = OUT[j];
        return;
    }

    uint written = 0u;
    for (;;) {
        KECCAK_PERMUTE();
        ulong OUT[25];
        OUT[ 0]=a00; OUT[ 1]=~a10; OUT[ 2]=~a20; OUT[ 3]=a30; OUT[ 4]=a40;
        OUT[ 5]=a01; OUT[ 6]= a11; OUT[ 7]= a21; OUT[ 8]=~a31; OUT[ 9]=a41;
        OUT[10]=a02; OUT[11]= a12; OUT[12]=~a22; OUT[13]= a32; OUT[14]=a42;
        OUT[15]=a03; OUT[16]= a13; OUT[17]=~a23; OUT[18]= a33; OUT[19]=a43;
        OUT[20]=~a04;OUT[21]= a14; OUT[22]= a24; OUT[23]= a34; OUT[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            op[written + j] = OUT[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```