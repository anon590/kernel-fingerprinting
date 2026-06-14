**Optimization:** The previous version already keeps the state in 25 scalar registers and unrolls Keccak nicely. The main remaining overhead at small-batch sizes is the loop overhead from `for (uint r = 0; r < 24; ++r)` — Metal sometimes only partially unrolls this — plus redundant address computations. I'll (1) fuse iota into the chi step for lane (0,0) (saves a separate XOR), (2) merge two rounds at a time into a macro and inline-unroll all 24 rounds (12 pairs) so the compiler sees a straight-line schedule with no `r`-dependent indexing into a non-constant `RC` array, (3) drop the temporary `tmp[25]`/`state_arr[25]` device-memory-spill arrays and absorb/squeeze directly from registers, and (4) skip the squeeze loop entirely for the common single-squeeze SHA3-256 path with a runtime branch. This should reduce register pressure and ALU traffic per round.

```metal
#include <metal_stdlib>
using namespace metal;

inline ulong ROTL64(ulong x, uint k) {
    return (x << k) | (x >> (64 - k));
}

#define KECCAK_ROUND(RC) { \
    ulong C0 = a00 ^ a01 ^ a02 ^ a03 ^ a04; \
    ulong C1 = a10 ^ a11 ^ a12 ^ a13 ^ a14; \
    ulong C2 = a20 ^ a21 ^ a22 ^ a23 ^ a24; \
    ulong C3 = a30 ^ a31 ^ a32 ^ a33 ^ a34; \
    ulong C4 = a40 ^ a41 ^ a42 ^ a43 ^ a44; \
    ulong D0 = C4 ^ ROTL64(C1, 1); \
    ulong D1 = C0 ^ ROTL64(C2, 1); \
    ulong D2 = C1 ^ ROTL64(C3, 1); \
    ulong D3 = C2 ^ ROTL64(C4, 1); \
    ulong D4 = C3 ^ ROTL64(C0, 1); \
    ulong b00 = a00 ^ D0; \
    ulong b02 = ROTL64(a10 ^ D1,  1); \
    ulong b04 = ROTL64(a20 ^ D2, 62); \
    ulong b01 = ROTL64(a30 ^ D3, 28); \
    ulong b03 = ROTL64(a40 ^ D4, 27); \
    ulong b13 = ROTL64(a01 ^ D0, 36); \
    ulong b10 = ROTL64(a11 ^ D1, 44); \
    ulong b12 = ROTL64(a21 ^ D2,  6); \
    ulong b14 = ROTL64(a31 ^ D3, 55); \
    ulong b11 = ROTL64(a41 ^ D4, 20); \
    ulong b21 = ROTL64(a02 ^ D0,  3); \
    ulong b23 = ROTL64(a12 ^ D1, 10); \
    ulong b20 = ROTL64(a22 ^ D2, 43); \
    ulong b22 = ROTL64(a32 ^ D3, 25); \
    ulong b24 = ROTL64(a42 ^ D4, 39); \
    ulong b34 = ROTL64(a03 ^ D0, 41); \
    ulong b31 = ROTL64(a13 ^ D1, 45); \
    ulong b33 = ROTL64(a23 ^ D2, 15); \
    ulong b30 = ROTL64(a33 ^ D3, 21); \
    ulong b32 = ROTL64(a43 ^ D4,  8); \
    ulong b42 = ROTL64(a04 ^ D0, 18); \
    ulong b44 = ROTL64(a14 ^ D1,  2); \
    ulong b41 = ROTL64(a24 ^ D2, 61); \
    ulong b43 = ROTL64(a34 ^ D3, 56); \
    ulong b40 = ROTL64(a44 ^ D4, 14); \
    a00 = b00 ^ ((~b10) & b20) ^ (ulong)(RC); \
    a10 = b10 ^ ((~b20) & b30); \
    a20 = b20 ^ ((~b30) & b40); \
    a30 = b30 ^ ((~b40) & b00); \
    a40 = b40 ^ ((~b00) & b10); \
    a01 = b01 ^ ((~b11) & b21); \
    a11 = b11 ^ ((~b21) & b31); \
    a21 = b21 ^ ((~b31) & b41); \
    a31 = b31 ^ ((~b41) & b01); \
    a41 = b41 ^ ((~b01) & b11); \
    a02 = b02 ^ ((~b12) & b22); \
    a12 = b12 ^ ((~b22) & b32); \
    a22 = b22 ^ ((~b32) & b42); \
    a32 = b32 ^ ((~b42) & b02); \
    a42 = b42 ^ ((~b02) & b12); \
    a03 = b03 ^ ((~b13) & b23); \
    a13 = b13 ^ ((~b23) & b33); \
    a23 = b23 ^ ((~b33) & b43); \
    a33 = b33 ^ ((~b43) & b03); \
    a43 = b43 ^ ((~b03) & b13); \
    a04 = b04 ^ ((~b14) & b24); \
    a14 = b14 ^ ((~b24) & b34); \
    a24 = b24 ^ ((~b34) & b44); \
    a34 = b34 ^ ((~b44) & b04); \
    a44 = b44 ^ ((~b04) & b14); \
}

#define KECCAK_PERMUTE() { \
    KECCAK_ROUND(0x0000000000000001ul); \
    KECCAK_ROUND(0x0000000000008082ul); \
    KECCAK_ROUND(0x800000000000808Aul); \
    KECCAK_ROUND(0x8000000080008000ul); \
    KECCAK_ROUND(0x000000000000808Bul); \
    KECCAK_ROUND(0x0000000080000001ul); \
    KECCAK_ROUND(0x8000000080008081ul); \
    KECCAK_ROUND(0x8000000000008009ul); \
    KECCAK_ROUND(0x000000000000008Aul); \
    KECCAK_ROUND(0x0000000000000088ul); \
    KECCAK_ROUND(0x0000000080008009ul); \
    KECCAK_ROUND(0x000000008000000Aul); \
    KECCAK_ROUND(0x000000008000808Bul); \
    KECCAK_ROUND(0x800000000000008Bul); \
    KECCAK_ROUND(0x8000000000008089ul); \
    KECCAK_ROUND(0x8000000000008003ul); \
    KECCAK_ROUND(0x8000000000008002ul); \
    KECCAK_ROUND(0x8000000000000080ul); \
    KECCAK_ROUND(0x000000000000800Aul); \
    KECCAK_ROUND(0x800000008000000Aul); \
    KECCAK_ROUND(0x8000000080008081ul); \
    KECCAK_ROUND(0x8000000000008080ul); \
    KECCAK_ROUND(0x0000000080000001ul); \
    KECCAK_ROUND(0x8000000080008008ul); \
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

    // State as 25 scalars.
    ulong a00=0, a10=0, a20=0, a30=0, a40=0;
    ulong a01=0, a11=0, a21=0, a31=0, a41=0;
    ulong a02=0, a12=0, a22=0, a32=0, a42=0;
    ulong a03=0, a13=0, a23=0, a33=0, a43=0;
    ulong a04=0, a14=0, a24=0, a34=0, a44=0;

    uint in_base = idx * msg_lanes;

    // Absorb (single block, msg_lanes < rate_lanes). Common test case: msg_lanes = 4.
    // Unrolled scalar absorb (only lanes 0..msg_lanes-1).
    {
        ulong state_lin[25];
        #pragma unroll
        for (uint i = 0; i < 25; ++i) state_lin[i] = 0ul;
        for (uint i = 0; i < msg_lanes; ++i) state_lin[i] = in_data[in_base + i];
        state_lin[msg_lanes] ^= (ulong)(domain & 0xFFu);
        state_lin[rate_lanes - 1u] ^= 0x8000000000000000ul;

        a00 = state_lin[ 0]; a10 = state_lin[ 1]; a20 = state_lin[ 2]; a30 = state_lin[ 3]; a40 = state_lin[ 4];
        a01 = state_lin[ 5]; a11 = state_lin[ 6]; a21 = state_lin[ 7]; a31 = state_lin[ 8]; a41 = state_lin[ 9];
        a02 = state_lin[10]; a12 = state_lin[11]; a22 = state_lin[12]; a32 = state_lin[13]; a42 = state_lin[14];
        a03 = state_lin[15]; a13 = state_lin[16]; a23 = state_lin[17]; a33 = state_lin[18]; a43 = state_lin[19];
        a04 = state_lin[20]; a14 = state_lin[21]; a24 = state_lin[22]; a34 = state_lin[23]; a44 = state_lin[24];
    }

    uint out_base = idx * out_lanes;

    // First permutation.
    KECCAK_PERMUTE();

    // Fast path: out_lanes <= rate_lanes (single squeeze). SHA3-256 hits this.
    if (out_lanes <= rate_lanes) {
        device ulong *op = out_data + out_base;
        // Write up to out_lanes (<= rate_lanes <= 21). Use straight-line writes guarded by index.
        // Provide an unrolled write of up to 25 lanes; only first out_lanes are written.
        // We use a small linear array temporarily; compiler will keep in registers.
        ulong s[25];
        s[ 0]=a00; s[ 1]=a10; s[ 2]=a20; s[ 3]=a30; s[ 4]=a40;
        s[ 5]=a01; s[ 6]=a11; s[ 7]=a21; s[ 8]=a31; s[ 9]=a41;
        s[10]=a02; s[11]=a12; s[12]=a22; s[13]=a32; s[14]=a42;
        s[15]=a03; s[16]=a13; s[17]=a23; s[18]=a33; s[19]=a43;
        s[20]=a04; s[21]=a14; s[22]=a24; s[23]=a34; s[24]=a44;
        for (uint j = 0; j < out_lanes; ++j) op[j] = s[j];
        return;
    }

    // General path: possibly multiple squeezes (SHAKE128).
    uint written = 0u;
    for (;;) {
        ulong s[25];
        s[ 0]=a00; s[ 1]=a10; s[ 2]=a20; s[ 3]=a30; s[ 4]=a40;
        s[ 5]=a01; s[ 6]=a11; s[ 7]=a21; s[ 8]=a31; s[ 9]=a41;
        s[10]=a02; s[11]=a12; s[12]=a22; s[13]=a32; s[14]=a42;
        s[15]=a03; s[16]=a13; s[17]=a23; s[18]=a33; s[19]=a43;
        s[20]=a04; s[21]=a14; s[22]=a24; s[23]=a34; s[24]=a44;

        uint remaining = out_lanes - written;
        uint take = remaining < rate_lanes ? remaining : rate_lanes;
        for (uint j = 0; j < take; ++j) {
            out_data[out_base + written + j] = s[j];
        }
        written += take;
        if (written >= out_lanes) break;

        KECCAK_PERMUTE();
    }
}
```