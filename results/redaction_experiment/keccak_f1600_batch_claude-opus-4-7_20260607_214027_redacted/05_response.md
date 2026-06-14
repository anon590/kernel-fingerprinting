**Optimization:** Going back to the known-correct incumbent and applying two targeted improvements: (1) replace the switch-statement absorb/squeeze with a direct-indexed scalar fan-out using a small register-backed temp array — but more importantly, since `msg_lanes` is always 4 in tests and the absorb pattern is fixed, I do a straight-line absorb (lanes 0..3 from in_data, then dynamic XOR for domain at lane `msg_lanes` and 0x80 at lane `rate_lanes-1`). (2) For the squeeze, write the state into a local `ulong[25]` once and copy out by index — avoiding the giant 25-case switch nested in a loop that the compiler was likely lowering poorly. (3) Keep chi in standard form (lane-complement trick is correct but didn't speed up last attempt and risked bugs). This should reduce instruction count in the absorb/squeeze hot path while keeping the verified-correct permutation.

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

#define ROTL64(x, k) (((x) << (k)) | ((x) >> (64 - (k))))

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

    // State as a local array; will be lifted to registers by the compiler
    // since all indexing into it is via small constant loops or short
    // variable indices that fan out.
    ulong S[25];
    #pragma unroll
    for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;

    // Absorb single block. msg_lanes is small (typically 4 for SHA3-256).
    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        S[i] = in_data[in_base + i];
    }
    S[msg_lanes]      ^= (ulong)(domain & 0xFFu);
    S[rate_lanes - 1u] ^= 0x8000000000000000ul;

    // Move into scalar registers for the permutation.
    ulong a00=S[ 0], a01=S[ 1], a02=S[ 2], a03=S[ 3], a04=S[ 4];
    ulong a05=S[ 5], a06=S[ 6], a07=S[ 7], a08=S[ 8], a09=S[ 9];
    ulong a10=S[10], a11=S[11], a12=S[12], a13=S[13], a14=S[14];
    ulong a15=S[15], a16=S[16], a17=S[17], a18=S[18], a19=S[19];
    ulong a20=S[20], a21=S[21], a22=S[22], a23=S[23], a24=S[24];

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        for (uint r = 0u; r < 24u; ++r) {
            // theta
            ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            ulong D0 = C4 ^ ROTL64(C1, 1);
            ulong D1 = C0 ^ ROTL64(C2, 1);
            ulong D2 = C1 ^ ROTL64(C3, 1);
            ulong D3 = C2 ^ ROTL64(C4, 1);
            ulong D4 = C3 ^ ROTL64(C0, 1);

            ulong t00 = a00 ^ D0;
            ulong t01 = a01 ^ D1;
            ulong t02 = a02 ^ D2;
            ulong t03 = a03 ^ D3;
            ulong t04 = a04 ^ D4;
            ulong t05 = a05 ^ D0;
            ulong t06 = a06 ^ D1;
            ulong t07 = a07 ^ D2;
            ulong t08 = a08 ^ D3;
            ulong t09 = a09 ^ D4;
            ulong t10 = a10 ^ D0;
            ulong t11 = a11 ^ D1;
            ulong t12 = a12 ^ D2;
            ulong t13 = a13 ^ D3;
            ulong t14 = a14 ^ D4;
            ulong t15 = a15 ^ D0;
            ulong t16 = a16 ^ D1;
            ulong t17 = a17 ^ D2;
            ulong t18 = a18 ^ D3;
            ulong t19 = a19 ^ D4;
            ulong t20 = a20 ^ D0;
            ulong t21 = a21 ^ D1;
            ulong t22 = a22 ^ D2;
            ulong t23 = a23 ^ D3;
            ulong t24 = a24 ^ D4;

            // rho + pi
            ulong b00 = t00;
            ulong b10 = ROTL64(t01,  1);
            ulong b20 = ROTL64(t02, 62);
            ulong b05 = ROTL64(t03, 28);
            ulong b15 = ROTL64(t04, 27);

            ulong b16 = ROTL64(t05, 36);
            ulong b01 = ROTL64(t06, 44);
            ulong b11 = ROTL64(t07,  6);
            ulong b21 = ROTL64(t08, 55);
            ulong b06 = ROTL64(t09, 20);

            ulong b07 = ROTL64(t10,  3);
            ulong b17 = ROTL64(t11, 10);
            ulong b02 = ROTL64(t12, 43);
            ulong b12 = ROTL64(t13, 25);
            ulong b22 = ROTL64(t14, 39);

            ulong b23 = ROTL64(t15, 41);
            ulong b08 = ROTL64(t16, 45);
            ulong b18 = ROTL64(t17, 15);
            ulong b03 = ROTL64(t18, 21);
            ulong b13 = ROTL64(t19,  8);

            ulong b14 = ROTL64(t20, 18);
            ulong b24 = ROTL64(t21,  2);
            ulong b09 = ROTL64(t22, 61);
            ulong b19 = ROTL64(t23, 56);
            ulong b04 = ROTL64(t24, 14);

            // chi
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

            // iota
            a00 ^= KECCAK_RC[r];
        }

        // Squeeze: dump scalar state into local array, then index by j.
        ulong O[25];
        O[ 0]=a00; O[ 1]=a01; O[ 2]=a02; O[ 3]=a03; O[ 4]=a04;
        O[ 5]=a05; O[ 6]=a06; O[ 7]=a07; O[ 8]=a08; O[ 9]=a09;
        O[10]=a10; O[11]=a11; O[12]=a12; O[13]=a13; O[14]=a14;
        O[15]=a15; O[16]=a16; O[17]=a17; O[18]=a18; O[19]=a19;
        O[20]=a20; O[21]=a21; O[22]=a22; O[23]=a23; O[24]=a24;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = O[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```