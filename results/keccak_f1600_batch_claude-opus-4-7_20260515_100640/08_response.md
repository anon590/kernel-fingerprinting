I'll note the skill availability but it's not relevant to this MSL kernel task.

**Optimization**: Going back to the best-scoring approach (iter 5, 0.96). The current attempt regressed because of register pressure with 25 named locals plus 25 b-locals. The winning strategy was likely a tighter round with less spilling. I'll:
1. Use a `ulong a[25]` array kept in registers, and a temporary `b[25]` array, allowing the compiler to schedule freely.
2. Specialize the absorb for `msg_bytes=32` (4 lanes) which is the only test case, while keeping the runtime path correct for any value.
3. Avoid the giant inlined macro that bloats register pressure; instead use a tight loop-free round body.

Actually, the cleanest path: precompute `D[5]` then do theta+rho+pi into `b[25]` then chi+iota back into `a[25]`. The compiler handles this well. Plus mark with `#pragma unroll` selectively.

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong ROTL64(ulong x, uint k) {
    return (x << k) | (x >> (64 - k));
}

constant ulong RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul, 0x8000000080008000ul,
    0x000000000000808Bul, 0x0000000080000001ul, 0x8000000080008081ul, 0x8000000000008009ul,
    0x000000000000008Aul, 0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul, 0x8000000000008003ul,
    0x8000000000008002ul, 0x8000000000000080ul, 0x000000000000800Aul, 0x800000008000000Aul,
    0x8000000080008081ul, 0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul
};

static inline void keccak_f1600(thread ulong *a) {
    for (uint r = 0; r < 24; ++r) {
        ulong C0 = a[0]  ^ a[5]  ^ a[10] ^ a[15] ^ a[20];
        ulong C1 = a[1]  ^ a[6]  ^ a[11] ^ a[16] ^ a[21];
        ulong C2 = a[2]  ^ a[7]  ^ a[12] ^ a[17] ^ a[22];
        ulong C3 = a[3]  ^ a[8]  ^ a[13] ^ a[18] ^ a[23];
        ulong C4 = a[4]  ^ a[9]  ^ a[14] ^ a[19] ^ a[24];

        ulong D0 = C4 ^ ROTL64(C1, 1);
        ulong D1 = C0 ^ ROTL64(C2, 1);
        ulong D2 = C1 ^ ROTL64(C3, 1);
        ulong D3 = C2 ^ ROTL64(C4, 1);
        ulong D4 = C3 ^ ROTL64(C0, 1);

        ulong b00 =        (a[0]  ^ D0)     ;
        ulong b10 = ROTL64(a[1]  ^ D1,  1);
        ulong b20 = ROTL64(a[2]  ^ D2, 62);
        ulong b30 = ROTL64(a[3]  ^ D3, 28);
        ulong b40 = ROTL64(a[4]  ^ D4, 27);

        ulong b31 = ROTL64(a[5]  ^ D0, 36);
        ulong b41 = ROTL64(a[6]  ^ D1, 44);
        ulong b01 = ROTL64(a[7]  ^ D2,  6);
        ulong b11 = ROTL64(a[8]  ^ D3, 55);
        ulong b21 = ROTL64(a[9]  ^ D4, 20);

        ulong b12 = ROTL64(a[10] ^ D0,  3);
        ulong b22 = ROTL64(a[11] ^ D1, 10);
        ulong b32 = ROTL64(a[12] ^ D2, 43);
        ulong b42 = ROTL64(a[13] ^ D3, 25);
        ulong b02 = ROTL64(a[14] ^ D4, 39);

        ulong b43 = ROTL64(a[15] ^ D0, 41);
        ulong b03 = ROTL64(a[16] ^ D1, 45);
        ulong b13 = ROTL64(a[17] ^ D2, 15);
        ulong b23 = ROTL64(a[18] ^ D3, 21);
        ulong b33 = ROTL64(a[19] ^ D4,  8);

        ulong b24 = ROTL64(a[20] ^ D0, 18);
        ulong b34 = ROTL64(a[21] ^ D1,  2);
        ulong b44 = ROTL64(a[22] ^ D2, 61);
        ulong b04 = ROTL64(a[23] ^ D3, 56);
        ulong b14 = ROTL64(a[24] ^ D4, 14);

        a[0]  = b00 ^ ((~b10) & b20) ^ RC[r];
        a[1]  = b10 ^ ((~b20) & b30);
        a[2]  = b20 ^ ((~b30) & b40);
        a[3]  = b30 ^ ((~b40) & b00);
        a[4]  = b40 ^ ((~b00) & b10);

        a[5]  = b01 ^ ((~b11) & b21);
        a[6]  = b11 ^ ((~b21) & b31);
        a[7]  = b21 ^ ((~b31) & b41);
        a[8]  = b31 ^ ((~b41) & b01);
        a[9]  = b41 ^ ((~b01) & b11);

        a[10] = b02 ^ ((~b12) & b22);
        a[11] = b12 ^ ((~b22) & b32);
        a[12] = b22 ^ ((~b32) & b42);
        a[13] = b32 ^ ((~b42) & b02);
        a[14] = b42 ^ ((~b02) & b12);

        a[15] = b03 ^ ((~b13) & b23);
        a[16] = b13 ^ ((~b23) & b33);
        a[17] = b23 ^ ((~b33) & b43);
        a[18] = b33 ^ ((~b43) & b03);
        a[19] = b43 ^ ((~b03) & b13);

        a[20] = b04 ^ ((~b14) & b24);
        a[21] = b14 ^ ((~b24) & b34);
        a[22] = b24 ^ ((~b34) & b44);
        a[23] = b34 ^ ((~b44) & b04);
        a[24] = b44 ^ ((~b04) & b14);
    }
}

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

    ulong a[25];
    for (uint i = 0; i < 25; ++i) a[i] = 0ul;

    uint in_base = idx * msg_lanes;
    for (uint i = 0; i < msg_lanes; ++i) a[i] = in_data[in_base + i];
    a[msg_lanes]       ^= (ulong)(domain & 0xFFu);
    a[rate_lanes - 1u] ^= 0x8000000000000000ul;

    uint out_base = idx * out_lanes;

    keccak_f1600(a);

    if (out_lanes <= rate_lanes) {
        for (uint j = 0; j < out_lanes; ++j) {
            out_data[out_base + j] = a[j];
        }
        return;
    }

    uint written = 0u;
    for (;;) {
        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = a[j];
        }
        written += take;
        if (written >= out_lanes) break;
        keccak_f1600(a);
    }
}
```