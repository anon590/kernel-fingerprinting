I scalarize the 1600-bit state into `uint2` lane halves, fully unroll the 24 Keccak rounds, and perform rho+pi in-place via the standard 24-lane cycle. This removes local array indexing, dynamic rho lookups, and the 25-lane temporary `B` array, greatly reducing spills and loop overhead. Using 32-bit halves also better matches Apple GPU integer hardware while preserving exact 64-bit lane semantics.

```metal
#include <metal_stdlib>
using namespace metal;

inline uint2 split_u64(ulong x) {
    return uint2((uint)x, (uint)(x >> 32));
}

inline ulong join_u64(uint2 v) {
    return (((ulong)v.y) << 32) | ((ulong)v.x);
}

#define ROL1(v) uint2(((v).x << 1u) | ((v).y >> 31u), ((v).y << 1u) | ((v).x >> 31u))
#define ROL_LT32(v,n) uint2(((v).x << (n)) | ((v).y >> (32u - (n))), ((v).y << (n)) | ((v).x >> (32u - (n))))
#define ROL_GT32(v,n) uint2(((v).y << ((n) - 32u)) | ((v).x >> (64u - (n))), ((v).x << ((n) - 32u)) | ((v).y >> (64u - (n))))

#define KECCAK_ROUND(rcv) do { \
    uint2 c0 = a0 ^ a5 ^ a10 ^ a15 ^ a20; \
    uint2 c1 = a1 ^ a6 ^ a11 ^ a16 ^ a21; \
    uint2 c2 = a2 ^ a7 ^ a12 ^ a17 ^ a22; \
    uint2 c3 = a3 ^ a8 ^ a13 ^ a18 ^ a23; \
    uint2 c4 = a4 ^ a9 ^ a14 ^ a19 ^ a24; \
    uint2 d0 = c4 ^ ROL1(c1); \
    uint2 d1 = c0 ^ ROL1(c2); \
    uint2 d2 = c1 ^ ROL1(c3); \
    uint2 d3 = c2 ^ ROL1(c4); \
    uint2 d4 = c3 ^ ROL1(c0); \
    a0 ^= d0; a5 ^= d0; a10 ^= d0; a15 ^= d0; a20 ^= d0; \
    a1 ^= d1; a6 ^= d1; a11 ^= d1; a16 ^= d1; a21 ^= d1; \
    a2 ^= d2; a7 ^= d2; a12 ^= d2; a17 ^= d2; a22 ^= d2; \
    a3 ^= d3; a8 ^= d3; a13 ^= d3; a18 ^= d3; a23 ^= d3; \
    a4 ^= d4; a9 ^= d4; a14 ^= d4; a19 ^= d4; a24 ^= d4; \
    uint2 t = a1; \
    uint2 u; \
    u = a10; a10 = ROL1(t);        t = u; \
    u = a7;  a7  = ROL_LT32(t,3u); t = u; \
    u = a11; a11 = ROL_LT32(t,6u); t = u; \
    u = a17; a17 = ROL_LT32(t,10u); t = u; \
    u = a18; a18 = ROL_LT32(t,15u); t = u; \
    u = a3;  a3  = ROL_LT32(t,21u); t = u; \
    u = a5;  a5  = ROL_LT32(t,28u); t = u; \
    u = a16; a16 = ROL_GT32(t,36u); t = u; \
    u = a8;  a8  = ROL_GT32(t,45u); t = u; \
    u = a21; a21 = ROL_GT32(t,55u); t = u; \
    u = a24; a24 = ROL_LT32(t,2u);  t = u; \
    u = a4;  a4  = ROL_LT32(t,14u); t = u; \
    u = a15; a15 = ROL_LT32(t,27u); t = u; \
    u = a23; a23 = ROL_GT32(t,41u); t = u; \
    u = a19; a19 = ROL_GT32(t,56u); t = u; \
    u = a13; a13 = ROL_LT32(t,8u);  t = u; \
    u = a12; a12 = ROL_LT32(t,25u); t = u; \
    u = a2;  a2  = ROL_GT32(t,43u); t = u; \
    u = a20; a20 = ROL_GT32(t,62u); t = u; \
    u = a14; a14 = ROL_LT32(t,18u); t = u; \
    u = a22; a22 = ROL_GT32(t,39u); t = u; \
    u = a9;  a9  = ROL_GT32(t,61u); t = u; \
    u = a6;  a6  = ROL_LT32(t,20u); t = u; \
    a1 = ROL_GT32(t,44u); \
    c0 = a0; c1 = a1; c2 = a2; c3 = a3; c4 = a4; \
    a0 = c0 ^ ((~c1) & c2); \
    a1 = c1 ^ ((~c2) & c3); \
    a2 = c2 ^ ((~c3) & c4); \
    a3 = c3 ^ ((~c4) & c0); \
    a4 = c4 ^ ((~c0) & c1); \
    a0 ^= (rcv); \
    c0 = a5; c1 = a6; c2 = a7; c3 = a8; c4 = a9; \
    a5 = c0 ^ ((~c1) & c2); \
    a6 = c1 ^ ((~c2) & c3); \
    a7 = c2 ^ ((~c3) & c4); \
    a8 = c3 ^ ((~c4) & c0); \
    a9 = c4 ^ ((~c0) & c1); \
    c0 = a10; c1 = a11; c2 = a12; c3 = a13; c4 = a14; \
    a10 = c0 ^ ((~c1) & c2); \
    a11 = c1 ^ ((~c2) & c3); \
    a12 = c2 ^ ((~c3) & c4); \
    a13 = c3 ^ ((~c4) & c0); \
    a14 = c4 ^ ((~c0) & c1); \
    c0 = a15; c1 = a16; c2 = a17; c3 = a18; c4 = a19; \
    a15 = c0 ^ ((~c1) & c2); \
    a16 = c1 ^ ((~c2) & c3); \
    a17 = c2 ^ ((~c3) & c4); \
    a18 = c3 ^ ((~c4) & c0); \
    a19 = c4 ^ ((~c0) & c1); \
    c0 = a20; c1 = a21; c2 = a22; c3 = a23; c4 = a24; \
    a20 = c0 ^ ((~c1) & c2); \
    a21 = c1 ^ ((~c2) & c3); \
    a22 = c2 ^ ((~c3) & c4); \
    a23 = c3 ^ ((~c4) & c0); \
    a24 = c4 ^ ((~c0) & c1); \
} while(false)

#define KECCAK_PERMUTE() do { \
    KECCAK_ROUND(uint2(0x00000001u, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x00008082u, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x0000808Au, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x80008000u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x0000808Bu, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x80000001u, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x80008081u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x00008009u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x0000008Au, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x00000088u, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x80008009u, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x8000000Au, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x8000808Bu, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x0000008Bu, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x00008089u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x00008003u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x00008002u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x00000080u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x0000800Au, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x8000000Au, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x80008081u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x00008080u, 0x80000000u)); \
    KECCAK_ROUND(uint2(0x80000001u, 0x00000000u)); \
    KECCAK_ROUND(uint2(0x80008008u, 0x80000000u)); \
} while(false)

#define LOAD_LANE(i, ai) do { if (msg_lanes > (i)) { ai = split_u64(in_data[in_base + (i)]); } } while(false)
#define DOMAIN_LANE(i, ai) do { if (msg_lanes == (i)) { ai ^= domv; } } while(false)
#define PAD_LANE(i, ai) do { if (pad_lane == (i)) { ai ^= padv; } } while(false)

#define STORE_TAKE(dst_, take_) do { \
    uint _dst = (dst_); \
    uint _take = (take_); \
    if (_take > 0u)  out_data[_dst + 0u]  = join_u64(a0); \
    if (_take > 1u)  out_data[_dst + 1u]  = join_u64(a1); \
    if (_take > 2u)  out_data[_dst + 2u]  = join_u64(a2); \
    if (_take > 3u)  out_data[_dst + 3u]  = join_u64(a3); \
    if (_take > 4u)  out_data[_dst + 4u]  = join_u64(a4); \
    if (_take > 5u)  out_data[_dst + 5u]  = join_u64(a5); \
    if (_take > 6u)  out_data[_dst + 6u]  = join_u64(a6); \
    if (_take > 7u)  out_data[_dst + 7u]  = join_u64(a7); \
    if (_take > 8u)  out_data[_dst + 8u]  = join_u64(a8); \
    if (_take > 9u)  out_data[_dst + 9u]  = join_u64(a9); \
    if (_take > 10u) out_data[_dst + 10u] = join_u64(a10); \
    if (_take > 11u) out_data[_dst + 11u] = join_u64(a11); \
    if (_take > 12u) out_data[_dst + 12u] = join_u64(a12); \
    if (_take > 13u) out_data[_dst + 13u] = join_u64(a13); \
    if (_take > 14u) out_data[_dst + 14u] = join_u64(a14); \
    if (_take > 15u) out_data[_dst + 15u] = join_u64(a15); \
    if (_take > 16u) out_data[_dst + 16u] = join_u64(a16); \
    if (_take > 17u) out_data[_dst + 17u] = join_u64(a17); \
    if (_take > 18u) out_data[_dst + 18u] = join_u64(a18); \
    if (_take > 19u) out_data[_dst + 19u] = join_u64(a19); \
    if (_take > 20u) out_data[_dst + 20u] = join_u64(a20); \
    if (_take > 21u) out_data[_dst + 21u] = join_u64(a21); \
    if (_take > 22u) out_data[_dst + 22u] = join_u64(a22); \
    if (_take > 23u) out_data[_dst + 23u] = join_u64(a23); \
    if (_take > 24u) out_data[_dst + 24u] = join_u64(a24); \
} while(false)

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

    uint2 z = uint2(0u, 0u);
    uint2 a0 = z, a1 = z, a2 = z, a3 = z, a4 = z;
    uint2 a5 = z, a6 = z, a7 = z, a8 = z, a9 = z;
    uint2 a10 = z, a11 = z, a12 = z, a13 = z, a14 = z;
    uint2 a15 = z, a16 = z, a17 = z, a18 = z, a19 = z;
    uint2 a20 = z, a21 = z, a22 = z, a23 = z, a24 = z;

    uint2 domv = uint2(domain & 0xFFu, 0u);
    uint2 padv = uint2(0u, 0x80000000u);

    if (msg_lanes == 4u) {
        uint in_base4 = idx << 2;
        a0 = split_u64(in_data[in_base4 + 0u]);
        a1 = split_u64(in_data[in_base4 + 1u]);
        a2 = split_u64(in_data[in_base4 + 2u]);
        a3 = split_u64(in_data[in_base4 + 3u]);
        a4 ^= domv;
    } else {
        uint in_base = idx * msg_lanes;
        LOAD_LANE(0u, a0);   LOAD_LANE(1u, a1);   LOAD_LANE(2u, a2);   LOAD_LANE(3u, a3);   LOAD_LANE(4u, a4);
        LOAD_LANE(5u, a5);   LOAD_LANE(6u, a6);   LOAD_LANE(7u, a7);   LOAD_LANE(8u, a8);   LOAD_LANE(9u, a9);
        LOAD_LANE(10u, a10); LOAD_LANE(11u, a11); LOAD_LANE(12u, a12); LOAD_LANE(13u, a13); LOAD_LANE(14u, a14);
        LOAD_LANE(15u, a15); LOAD_LANE(16u, a16); LOAD_LANE(17u, a17); LOAD_LANE(18u, a18); LOAD_LANE(19u, a19);
        LOAD_LANE(20u, a20); LOAD_LANE(21u, a21); LOAD_LANE(22u, a22); LOAD_LANE(23u, a23); LOAD_LANE(24u, a24);

        DOMAIN_LANE(0u, a0);   DOMAIN_LANE(1u, a1);   DOMAIN_LANE(2u, a2);   DOMAIN_LANE(3u, a3);   DOMAIN_LANE(4u, a4);
        DOMAIN_LANE(5u, a5);   DOMAIN_LANE(6u, a6);   DOMAIN_LANE(7u, a7);   DOMAIN_LANE(8u, a8);   DOMAIN_LANE(9u, a9);
        DOMAIN_LANE(10u, a10); DOMAIN_LANE(11u, a11); DOMAIN_LANE(12u, a12); DOMAIN_LANE(13u, a13); DOMAIN_LANE(14u, a14);
        DOMAIN_LANE(15u, a15); DOMAIN_LANE(16u, a16); DOMAIN_LANE(17u, a17); DOMAIN_LANE(18u, a18); DOMAIN_LANE(19u, a19);
        DOMAIN_LANE(20u, a20); DOMAIN_LANE(21u, a21); DOMAIN_LANE(22u, a22); DOMAIN_LANE(23u, a23); DOMAIN_LANE(24u, a24);
    }

    if (rate_lanes == 17u) {
        a16 ^= padv;
    } else if (rate_lanes == 21u) {
        a20 ^= padv;
    } else if (rate_lanes == 9u) {
        a8 ^= padv;
    } else {
        uint pad_lane = rate_lanes - 1u;
        PAD_LANE(0u, a0);   PAD_LANE(1u, a1);   PAD_LANE(2u, a2);   PAD_LANE(3u, a3);   PAD_LANE(4u, a4);
        PAD_LANE(5u, a5);   PAD_LANE(6u, a6);   PAD_LANE(7u, a7);   PAD_LANE(8u, a8);   PAD_LANE(9u, a9);
        PAD_LANE(10u, a10); PAD_LANE(11u, a11); PAD_LANE(12u, a12); PAD_LANE(13u, a13); PAD_LANE(14u, a14);
        PAD_LANE(15u, a15); PAD_LANE(16u, a16); PAD_LANE(17u, a17); PAD_LANE(18u, a18); PAD_LANE(19u, a19);
        PAD_LANE(20u, a20); PAD_LANE(21u, a21); PAD_LANE(22u, a22); PAD_LANE(23u, a23); PAD_LANE(24u, a24);
    }

    if (out_lanes == 0u) return;

    KECCAK_PERMUTE();

    if (out_lanes == 4u && rate_lanes >= 4u) {
        uint out_base4 = idx << 2;
        out_data[out_base4 + 0u] = join_u64(a0);
        out_data[out_base4 + 1u] = join_u64(a1);
        out_data[out_base4 + 2u] = join_u64(a2);
        out_data[out_base4 + 3u] = join_u64(a3);
        return;
    }

    uint out_base = idx * out_lanes;
    uint take = out_lanes < rate_lanes ? out_lanes : rate_lanes;
    STORE_TAKE(out_base, take);

    uint written = take;
    while (written < out_lanes) {
        KECCAK_PERMUTE();
        uint remaining = out_lanes - written;
        take = remaining < rate_lanes ? remaining : rate_lanes;
        STORE_TAKE(out_base + written, take);
        written += take;
    }
}
```