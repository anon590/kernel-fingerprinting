Acknowledging the system reminder about the keybindings-help skill — it's not relevant to this Metal kernel optimization task, so I'll proceed with the actual request.

**Structural change:** Instead of keeping all 25 lanes as scalar registers in one thread, I'll use a **lane-parallel approach** where 5 threads in a simdgroup cooperate on a single Keccak instance, each holding one *column* (5 lanes, one per y). This gives:
- Theta column sum becomes a 5-way XOR within a thread, and D[x] is fetched via `simd_shuffle` from neighbor columns — replacing 25-XOR theta with much less work per thread.
- Rho+pi: the 5 lanes within a thread go to different destinations; we use simd_shuffle on a per-y-row basis. Actually simplest: store state as 5 lanes per thread indexed by y, with the thread's x = lane_id % 5. Pi moves lanes between threads, which is a shuffle.
- Chi works across x: shuffle neighbors x+1, x+2 from sibling threads.
- 6 instances per simdgroup (30 lanes, 2 idle). Threads-per-grid still equals batch (one thread per instance externally), but we ignore that — wait, the spec says each thread = one instance. We must respect dispatch.

Given the constraint "each thread processes ONE instance end-to-end," I'll instead try a **register-pressure reduction via interleaved column processing** with `simd_shuffle` cooperation across sibling instances to share RC loads, and most importantly **eliminate the switch ladders** which generate massive branchy code. Use direct indexed local arrays (the compiler will keep them in registers for known sizes).

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

    // Keep state in a flat 25-element array; the compiler will promote
    // to registers since all indices used in the permutation are
    // compile-time constants.
    ulong S[25];
    for (uint i = 0u; i < 25u; ++i) S[i] = 0ul;

    // Absorb: data-dependent loop with data-dependent index. Use a
    // device pointer offset; the loop trip count msg_lanes is small.
    uint in_base = idx * msg_lanes;
    for (uint i = 0u; i < msg_lanes; ++i) {
        S[i] = in_data[in_base + i];
    }
    // Padding: domain at byte 0 of lane msg_lanes; 0x80 at byte 7 of lane rate_lanes-1.
    S[msg_lanes]      ^= (ulong)(domain & 0xFFu);
    S[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;
    uint written = 0u;

    for (;;) {
        // Pull the entire state into named locals so all subsequent
        // operations are register-resident with constant indices.
        ulong a00=S[ 0], a01=S[ 1], a02=S[ 2], a03=S[ 3], a04=S[ 4];
        ulong a05=S[ 5], a06=S[ 6], a07=S[ 7], a08=S[ 8], a09=S[ 9];
        ulong a10=S[10], a11=S[11], a12=S[12], a13=S[13], a14=S[14];
        ulong a15=S[15], a16=S[16], a17=S[17], a18=S[18], a19=S[19];
        ulong a20=S[20], a21=S[21], a22=S[22], a23=S[23], a24=S[24];

        // Unroll 2 rounds at a time, using b-buffer for round k and
        // ping-pong back to a for round k+1. This halves the round
        // overhead and gives the compiler more scheduling freedom.
        for (uint r = 0u; r < 24u; r += 2u) {
            // ----------- Round r -----------
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

            ulong b00 =        (a00 ^ D0);
            ulong b10 = ROTL64(a01 ^ D1,  1);
            ulong b20 = ROTL64(a02 ^ D2, 62);
            ulong b05 = ROTL64(a03 ^ D3, 28);
            ulong b15 = ROTL64(a04 ^ D4, 27);
            ulong b16 = ROTL64(a05 ^ D0, 36);
            ulong b01 = ROTL64(a06 ^ D1, 44);
            ulong b11 = ROTL64(a07 ^ D2,  6);
            ulong b21 = ROTL64(a08 ^ D3, 55);
            ulong b06 = ROTL64(a09 ^ D4, 20);
            ulong b07 = ROTL64(a10 ^ D0,  3);
            ulong b17 = ROTL64(a11 ^ D1, 10);
            ulong b02 = ROTL64(a12 ^ D2, 43);
            ulong b12 = ROTL64(a13 ^ D3, 25);
            ulong b22 = ROTL64(a14 ^ D4, 39);
            ulong b23 = ROTL64(a15 ^ D0, 41);
            ulong b08 = ROTL64(a16 ^ D1, 45);
            ulong b18 = ROTL64(a17 ^ D2, 15);
            ulong b03 = ROTL64(a18 ^ D3, 21);
            ulong b13 = ROTL64(a19 ^ D4,  8);
            ulong b14 = ROTL64(a20 ^ D0, 18);
            ulong b24 = ROTL64(a21 ^ D1,  2);
            ulong b09 = ROTL64(a22 ^ D2, 61);
            ulong b19 = ROTL64(a23 ^ D3, 56);
            ulong b04 = ROTL64(a24 ^ D4, 14);

            a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[r];
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

            // ----------- Round r+1 -----------
            C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20;
            C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21;
            C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22;
            C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23;
            C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24;

            D0 = C4 ^ ROTL64(C1, 1);
            D1 = C0 ^ ROTL64(C2, 1);
            D2 = C1 ^ ROTL64(C3, 1);
            D3 = C2 ^ ROTL64(C4, 1);
            D4 = C3 ^ ROTL64(C0, 1);

            b00 =        (a00 ^ D0);
            b10 = ROTL64(a01 ^ D1,  1);
            b20 = ROTL64(a02 ^ D2, 62);
            b05 = ROTL64(a03 ^ D3, 28);
            b15 = ROTL64(a04 ^ D4, 27);
            b16 = ROTL64(a05 ^ D0, 36);
            b01 = ROTL64(a06 ^ D1, 44);
            b11 = ROTL64(a07 ^ D2,  6);
            b21 = ROTL64(a08 ^ D3, 55);
            b06 = ROTL64(a09 ^ D4, 20);
            b07 = ROTL64(a10 ^ D0,  3);
            b17 = ROTL64(a11 ^ D1, 10);
            b02 = ROTL64(a12 ^ D2, 43);
            b12 = ROTL64(a13 ^ D3, 25);
            b22 = ROTL64(a14 ^ D4, 39);
            b23 = ROTL64(a15 ^ D0, 41);
            b08 = ROTL64(a16 ^ D1, 45);
            b18 = ROTL64(a17 ^ D2, 15);
            b03 = ROTL64(a18 ^ D3, 21);
            b13 = ROTL64(a19 ^ D4,  8);
            b14 = ROTL64(a20 ^ D0, 18);
            b24 = ROTL64(a21 ^ D1,  2);
            b09 = ROTL64(a22 ^ D2, 61);
            b19 = ROTL64(a23 ^ D3, 56);
            b04 = ROTL64(a24 ^ D4, 14);

            a00 = b00 ^ ((~b01) & b02) ^ KECCAK_RC[r + 1u];
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
        }

        // Spill back to S[] so we can do a data-dependent squeeze.
        S[ 0]=a00; S[ 1]=a01; S[ 2]=a02; S[ 3]=a03; S[ 4]=a04;
        S[ 5]=a05; S[ 6]=a06; S[ 7]=a07; S[ 8]=a08; S[ 9]=a09;
        S[10]=a10; S[11]=a11; S[12]=a12; S[13]=a13; S[14]=a14;
        S[15]=a15; S[16]=a16; S[17]=a17; S[18]=a18; S[19]=a19;
        S[20]=a20; S[21]=a21; S[22]=a22; S[23]=a23; S[24]=a24;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0u; j < take; ++j) {
            out_data[out_base + written + j] = S[j];
        }
        written += take;
        if (written >= out_lanes) break;
    }
}
```