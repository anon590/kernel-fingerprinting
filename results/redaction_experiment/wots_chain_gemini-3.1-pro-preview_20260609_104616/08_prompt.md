## Task: wots_chain

Batched WOTS+ / SPHINCS+-style hash chains. Given ``n_chains`` independent ``n_bytes``-byte seeds, apply the Keccak-256 inner hash ``w`` times in sequence per chain (each digest truncated to ``n_bytes`` bytes before feeding into the next iteration) and write the chain tip to the output. The chains are embarrassingly parallel; the ``w``-step iteration along each chain is strictly sequential.

Inner hash: Keccak-f[1600] with the FIPS 202 SHA3-256 sponge framing -- rate = 136 bytes (17 lanes), capacity = 64 bytes, domain pad byte = 0x06. State convention: the 1600-bit state is a 5x5 array of 64-bit lanes; lane k = x + 5*y holds bytes 8*k .. 8*k + 7 of the sponge state in little-endian.

All test sizes have ``n_bytes < rate_bytes`` (in-distribution n_bytes=16, held-out n_bytes=32; rate_bytes=136), so every chain step collapses to a single-block absorb + single-block squeeze of ``n_lanes = n_bytes / 8`` state lanes:
  state                          := 0
  state[lane 0..n_lanes-1]       := previous_chunk
  state[lane n_lanes, byte 0]    ^= 0x06   # SHA3 domain
  state[lane 16, byte 7]         ^= 0x80   # FIPS 202 final pad
  state                          := Keccak-f1600(state)
  next_chunk                     := state[lane 0..n_lanes-1]

On the first chain step the absorb is the seed; on every subsequent step the absorb is the n_lanes-lane truncation of the previous Keccak-f1600 output. After ``w`` steps the first n_lanes state lanes are written to the output as the chain tip.

The kernel must read ``n_bytes`` and ``w`` from the bound device buffers rather than treating them as compile-time constants; both vary across the test sizes (``w`` in {16, 64, 256} in the in-distribution sweep, ``n_bytes`` 16 -> 32 between in-distribution and held-out). Hardcoding either value silently produces wrong output, not just slow output.

Correctness is bit-exact against ``hashlib.sha3_256`` iterated ``w`` times with ``n_bytes``-byte truncation; any mismatched output ulong rejects the candidate.

## Required kernel signature(s)

```
kernel void wots_chain(
    device const ulong *seeds    [[buffer(0)]],
    device       ulong *tips     [[buffer(1)]],
    constant uint      &n_chains [[buffer(2)]],
    constant uint      &n_bytes  [[buffer(3)]],
    constant uint      &w        [[buffer(4)]],
    uint idx [[thread_position_in_grid]]);

Dispatch (host-fixed):
  threadsPerGrid        = (n_chains, 1, 1)
  threadsPerThreadgroup = (min(n_chains, 64), 1, 1)
Each thread processes ONE chain end-to-end; guard against idx >= n_chains (the grid is rounded up to a multiple of the TG width). ``seeds`` is laid out as n_chains consecutive runs of ``n_bytes / 8`` ulongs; ``tips`` likewise. The external buffer layout above must be preserved and the per-chain sequential semantics honored: each chain's step ``j+1`` must read the digest produced by its own step ``j`` (cross-chain mixing of intermediate digests would be a correctness bug).
```

## Your previous attempt

```metal
#include <metal_stdlib>
using namespace metal;

#define rotl64(x, k) (((x) << (k)) | ((x) >> (64u - (k))))

inline __attribute__((always_inline))
void keccak_f1600_scalar(
    thread ulong &a00, thread ulong &a01, thread ulong &a02, thread ulong &a03, thread ulong &a04,
    thread ulong &a05, thread ulong &a06, thread ulong &a07, thread ulong &a08, thread ulong &a09,
    thread ulong &a10, thread ulong &a11, thread ulong &a12, thread ulong &a13, thread ulong &a14,
    thread ulong &a15, thread ulong &a16, thread ulong &a17, thread ulong &a18, thread ulong &a19,
    thread ulong &a20, thread ulong &a21, thread ulong &a22, thread ulong &a23, thread ulong &a24)
{
    ulong b00, b01, b02, b03, b04;
    ulong b05, b06, b07, b08, b09;
    ulong b10, b11, b12, b13, b14;
    ulong b15, b16, b17, b18, b19;
    ulong b20, b21, b22, b23, b24;

    #define R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, \
              b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, rc) \
    do { \
        ulong C0 = a00 ^ a05 ^ a10 ^ a15 ^ a20; \
        ulong C1 = a01 ^ a06 ^ a11 ^ a16 ^ a21; \
        ulong C2 = a02 ^ a07 ^ a12 ^ a17 ^ a22; \
        ulong C3 = a03 ^ a08 ^ a13 ^ a18 ^ a23; \
        ulong C4 = a04 ^ a09 ^ a14 ^ a19 ^ a24; \
        ulong D0 = C4 ^ rotl64(C1, 1u); \
        ulong D1 = C0 ^ rotl64(C2, 1u); \
        ulong D2 = C1 ^ rotl64(C3, 1u); \
        ulong D3 = C2 ^ rotl64(C4, 1u); \
        ulong D4 = C3 ^ rotl64(C0, 1u); \
        ulong t0 = a00 ^ D0; \
        ulong t1 = rotl64(a06 ^ D1, 44u); \
        ulong t2 = rotl64(a12 ^ D2, 43u); \
        ulong t3 = rotl64(a18 ^ D3, 21u); \
        ulong t4 = rotl64(a24 ^ D4, 14u); \
        b00 = t0 ^ (t2 & ~t1) ^ rc; \
        b01 = t1 ^ (t3 & ~t2); \
        b02 = t2 ^ (t4 & ~t3); \
        b03 = t3 ^ (t0 & ~t4); \
        b04 = t4 ^ (t1 & ~t0); \
        t0 = rotl64(a03 ^ D3, 28u); \
        t1 = rotl64(a09 ^ D4, 20u); \
        t2 = rotl64(a10 ^ D0, 3u); \
        t3 = rotl64(a16 ^ D1, 45u); \
        t4 = rotl64(a22 ^ D2, 61u); \
        b05 = t0 ^ (t2 & ~t1); \
        b06 = t1 ^ (t3 & ~t2); \
        b07 = t2 ^ (t4 & ~t3); \
        b08 = t3 ^ (t0 & ~t4); \
        b09 = t4 ^ (t1 & ~t0); \
        t0 = rotl64(a01 ^ D1, 1u); \
        t1 = rotl64(a07 ^ D2, 6u); \
        t2 = rotl64(a13 ^ D3, 25u); \
        t3 = rotl64(a19 ^ D4, 8u); \
        t4 = rotl64(a20 ^ D0, 18u); \
        b10 = t0 ^ (t2 & ~t1); \
        b11 = t1 ^ (t3 & ~t2); \
        b12 = t2 ^ (t4 & ~t3); \
        b13 = t3 ^ (t0 & ~t4); \
        b14 = t4 ^ (t1 & ~t0); \
        t0 = rotl64(a04 ^ D4, 27u); \
        t1 = rotl64(a05 ^ D0, 36u); \
        t2 = rotl64(a11 ^ D1, 10u); \
        t3 = rotl64(a17 ^ D2, 15u); \
        t4 = rotl64(a23 ^ D3, 56u); \
        b15 = t0 ^ (t2 & ~t1); \
        b16 = t1 ^ (t3 & ~t2); \
        b17 = t2 ^ (t4 & ~t3); \
        b18 = t3 ^ (t0 & ~t4); \
        b19 = t4 ^ (t1 & ~t0); \
        t0 = rotl64(a02 ^ D2, 62u); \
        t1 = rotl64(a08 ^ D3, 55u); \
        t2 = rotl64(a14 ^ D4, 39u); \
        t3 = rotl64(a15 ^ D0, 41u); \
        t4 = rotl64(a21 ^ D1, 2u); \
        b20 = t0 ^ (t2 & ~t1); \
        b21 = t1 ^ (t3 & ~t2); \
        b22 = t2 ^ (t4 & ~t3); \
        b23 = t3 ^ (t0 & ~t4); \
        b24 = t4 ^ (t1 & ~t0); \
    } while(0)

    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x0000000000000001ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x0000000000008082ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x800000000000808Aul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x8000000080008000ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x000000000000808Bul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x0000000080000001ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x8000000080008081ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x8000000000008009ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x000000000000008Aul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x0000000000000088ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x0000000080008009ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x000000008000000Aul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x000000008000808Bul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x800000000000008Bul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x8000000000008089ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x8000000000008003ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x8000000000008002ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x8000000000000080ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x000000000000800Aul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x800000008000000Aul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x8000000080008081ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x8000000000008080ul);
    R(a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,
      b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24, 0x0000000080000001ul);
    R(b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,
      a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24, 0x8000000080008008ul);

    #undef R
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
    uint w_val = w;

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        
        ulong a00 = seed_val.x;
        ulong a01 = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            ulong a02 = 0x06ul;
            ulong a03 = 0ul; ulong a04 = 0ul; ulong a05 = 0ul; ulong a06 = 0ul;
            ulong a07 = 0ul; ulong a08 = 0ul; ulong a09 = 0ul; ulong a10 = 0ul;
            ulong a11 = 0ul; ulong a12 = 0ul; ulong a13 = 0ul; ulong a14 = 0ul;
            ulong a15 = 0ul; 
            ulong a16 = 0x8000000000000000ul;
            ulong a17 = 0ul; ulong a18 = 0ul; ulong a19 = 0ul; ulong a20 = 0ul;
            ulong a21 = 0ul; ulong a22 = 0ul; ulong a23 = 0ul; ulong a24 = 0ul;
            
            keccak_f1600_scalar(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09,
                                a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                a20, a21, a22, a23, a24);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(a00, a01);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        
        ulong a00 = seed_val.x;
        ulong a01 = seed_val.y;
        ulong a02 = seed_val.z;
        ulong a03 = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            ulong a04 = 0x06ul;
            ulong a05 = 0ul; ulong a06 = 0ul; ulong a07 = 0ul; ulong a08 = 0ul; 
            ulong a09 = 0ul; ulong a10 = 0ul; ulong a11 = 0ul; ulong a12 = 0ul; 
            ulong a13 = 0ul; ulong a14 = 0ul; ulong a15 = 0ul; 
            ulong a16 = 0x8000000000000000ul;
            ulong a17 = 0ul; ulong a18 = 0ul; ulong a19 = 0ul; ulong a20 = 0ul;
            ulong a21 = 0ul; ulong a22 = 0ul; ulong a23 = 0ul; ulong a24 = 0ul;
            
            keccak_f1600_scalar(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09,
                                a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
                                a20, a21, a22, a23, a24);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(a00, a01, a02, a03);

    } else {
        uint base = idx * n_lanes;
        ulong A[25];
        
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) A[i] = 0ul;
                }
            }
            A[n_lanes] ^= 0x06ul;
            A[16u]     ^= 0x8000000000000000ul;
            
            keccak_f1600_scalar(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9],
                                A[10], A[11], A[12], A[13], A[14], A[15], A[16], A[17], A[18], A[19],
                                A[20], A[21], A[22], A[23], A[24]);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            if (i < n_lanes) tips[base + i] = A[i];
        }
    }
}
```

Result of previous attempt:
          w16_C64K: correct, 5.68 ms, 686.4 Gbitops/s (u64) (61.0% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.73 ms, 718.1 Gbitops/s (u64) (63.8% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 87.32 ms, 714.8 Gbitops/s (u64) (63.5% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6278

## Current best (incumbent)

```metal
#include <metal_stdlib>
using namespace metal;

constexpr constant ulong KECCAK_RC[24] = {
    0x0000000000000001ul, 0x0000000000008082ul, 0x800000000000808Aul,
    0x8000000080008000ul, 0x000000000000808Bul, 0x0000000080000001ul,
    0x8000000080008081ul, 0x8000000000008009ul, 0x000000000000008Aul,
    0x0000000000000088ul, 0x0000000080008009ul, 0x000000008000000Aul,
    0x000000008000808Bul, 0x800000000000008Bul, 0x8000000000008089ul,
    0x8000000000008003ul, 0x8000000000008002ul, 0x8000000000000080ul,
    0x000000000000800Aul, 0x800000008000000Aul, 0x8000000080008081ul,
    0x8000000000008080ul, 0x0000000080000001ul, 0x8000000080008008ul,
};

inline __attribute__((always_inline))
ulong rotl64(ulong x, uint k) {
    return (x << k) | (x >> (64u - k));
}

#define ROUND(A, E, rc) do { \
    ulong C0 = (A[0]  ^ A[5])  ^ (A[10] ^ A[15]) ^ A[20]; \
    ulong C1 = (A[1]  ^ A[6])  ^ (A[11] ^ A[16]) ^ A[21]; \
    ulong C2 = (A[2]  ^ A[7])  ^ (A[12] ^ A[17]) ^ A[22]; \
    ulong C3 = (A[3]  ^ A[8])  ^ (A[13] ^ A[18]) ^ A[23]; \
    ulong C4 = (A[4]  ^ A[9])  ^ (A[14] ^ A[19]) ^ A[24]; \
    \
    ulong D0 = C4 ^ rotl64(C1, 1u); \
    ulong D1 = C0 ^ rotl64(C2, 1u); \
    ulong D2 = C1 ^ rotl64(C3, 1u); \
    ulong D3 = C2 ^ rotl64(C4, 1u); \
    ulong D4 = C3 ^ rotl64(C0, 1u); \
    \
    ulong b0 = A[0] ^ D0; \
    ulong b1 = rotl64(A[6] ^ D1, 44u); \
    ulong b2 = rotl64(A[12] ^ D2, 43u); \
    ulong b3 = rotl64(A[18] ^ D3, 21u); \
    ulong b4 = rotl64(A[24] ^ D4, 14u); \
    E[0] = b0 ^ (b2 & ~b1) ^ rc; \
    E[1] = b1 ^ (b3 & ~b2); \
    E[2] = b2 ^ (b4 & ~b3); \
    E[3] = b3 ^ (b0 & ~b4); \
    E[4] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[3] ^ D3, 28u); \
    b1 = rotl64(A[9] ^ D4, 20u); \
    b2 = rotl64(A[10] ^ D0, 3u); \
    b3 = rotl64(A[16] ^ D1, 45u); \
    b4 = rotl64(A[22] ^ D2, 61u); \
    E[5] = b0 ^ (b2 & ~b1); \
    E[6] = b1 ^ (b3 & ~b2); \
    E[7] = b2 ^ (b4 & ~b3); \
    E[8] = b3 ^ (b0 & ~b4); \
    E[9] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[1] ^ D1, 1u); \
    b1 = rotl64(A[7] ^ D2, 6u); \
    b2 = rotl64(A[13] ^ D3, 25u); \
    b3 = rotl64(A[19] ^ D4, 8u); \
    b4 = rotl64(A[20] ^ D0, 18u); \
    E[10] = b0 ^ (b2 & ~b1); \
    E[11] = b1 ^ (b3 & ~b2); \
    E[12] = b2 ^ (b4 & ~b3); \
    E[13] = b3 ^ (b0 & ~b4); \
    E[14] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[4] ^ D4, 27u); \
    b1 = rotl64(A[5] ^ D0, 36u); \
    b2 = rotl64(A[11] ^ D1, 10u); \
    b3 = rotl64(A[17] ^ D2, 15u); \
    b4 = rotl64(A[23] ^ D3, 56u); \
    E[15] = b0 ^ (b2 & ~b1); \
    E[16] = b1 ^ (b3 & ~b2); \
    E[17] = b2 ^ (b4 & ~b3); \
    E[18] = b3 ^ (b0 & ~b4); \
    E[19] = b4 ^ (b1 & ~b0); \
    \
    b0 = rotl64(A[2] ^ D2, 62u); \
    b1 = rotl64(A[8] ^ D3, 55u); \
    b2 = rotl64(A[14] ^ D4, 39u); \
    b3 = rotl64(A[15] ^ D0, 41u); \
    b4 = rotl64(A[21] ^ D1, 2u); \
    E[20] = b0 ^ (b2 & ~b1); \
    E[21] = b1 ^ (b3 & ~b2); \
    E[22] = b2 ^ (b4 & ~b3); \
    E[23] = b3 ^ (b0 & ~b4); \
    E[24] = b4 ^ (b1 & ~b0); \
} while(0)

struct KeccakState {
    ulong A[25];
};

inline __attribute__((always_inline))
KeccakState keccak_f1600(KeccakState s) {
    KeccakState e;
    #pragma unroll
    for (uint r = 0u; r < 24u; r += 2u) {
        ROUND(s.A, e.A, KECCAK_RC[r]);
        ROUND(e.A, s.A, KECCAK_RC[r+1]);
    }
    return s;
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
    uint w_val = w;

    if (n_lanes == 2u) {
        device const ulong2 *seeds_v = (device const ulong2 *)seeds;
        ulong2 seed_val = seeds_v[idx];
        
        KeccakState s;
        s.A[0] = seed_val.x;
        s.A[1] = seed_val.y;
        
        for (uint step = 0u; step < w_val; ++step) {
            s.A[2]  = 0x06ul;
            s.A[3]  = 0ul; s.A[4]  = 0ul; s.A[5]  = 0ul; s.A[6]  = 0ul;
            s.A[7]  = 0ul; s.A[8]  = 0ul; s.A[9]  = 0ul; s.A[10] = 0ul;
            s.A[11] = 0ul; s.A[12] = 0ul; s.A[13] = 0ul; s.A[14] = 0ul;
            s.A[15] = 0ul; 
            s.A[16] = 0x8000000000000000ul;
            s.A[17] = 0ul; s.A[18] = 0ul; s.A[19] = 0ul; s.A[20] = 0ul;
            s.A[21] = 0ul; s.A[22] = 0ul; s.A[23] = 0ul; s.A[24] = 0ul;
            
            s = keccak_f1600(s);
        }
        
        device ulong2 *tips_v = (device ulong2 *)tips;
        tips_v[idx] = ulong2(s.A[0], s.A[1]);

    } else if (n_lanes == 4u) {
        device const ulong4 *seeds_v = (device const ulong4 *)seeds;
        ulong4 seed_val = seeds_v[idx];
        
        KeccakState s;
        s.A[0] = seed_val.x;
        s.A[1] = seed_val.y;
        s.A[2] = seed_val.z;
        s.A[3] = seed_val.w;
        
        for (uint step = 0u; step < w_val; ++step) {
            s.A[4]  = 0x06ul;
            s.A[5]  = 0ul; s.A[6]  = 0ul; s.A[7]  = 0ul; s.A[8]  = 0ul; 
            s.A[9]  = 0ul; s.A[10] = 0ul; s.A[11] = 0ul; s.A[12] = 0ul; 
            s.A[13] = 0ul; s.A[14] = 0ul; s.A[15] = 0ul; 
            s.A[16] = 0x8000000000000000ul;
            s.A[17] = 0ul; s.A[18] = 0ul; s.A[19] = 0ul; s.A[20] = 0ul;
            s.A[21] = 0ul; s.A[22] = 0ul; s.A[23] = 0ul; s.A[24] = 0ul;
            
            s = keccak_f1600(s);
        }
        
        device ulong4 *tips_v = (device ulong4 *)tips;
        tips_v[idx] = ulong4(s.A[0], s.A[1], s.A[2], s.A[3]);

    } else {
        uint base = idx * n_lanes;
        KeccakState s;
        #pragma unroll
        for (uint i = 0u; i < 25u; ++i) {
            s.A[i] = (i < n_lanes) ? seeds[base + i] : 0ul;
        }
        
        for (uint step = 0u; step < w_val; ++step) {
            if (step > 0u) {
                #pragma unroll
                for (uint i = 0u; i < 25u; ++i) {
                    if (i >= n_lanes) s.A[i] = 0ul;
                }
            }
            s.A[n_lanes] ^= 0x06ul;
            s.A[16u]     ^= 0x8000000000000000ul;
            
            s = keccak_f1600(s);
        }
        
        #pragma unroll
        for (uint i = 0u; i < 17u; ++i) {
            if (i < n_lanes) tips[base + i] = s.A[i];
        }
    }
}
```

Incumbent result:
          w16_C64K: correct, 5.53 ms, 705.7 Gbitops/s (u64) (62.7% of 1125 Gops/s (u64 bitop, est))
          w64_C64K: correct, 21.50 ms, 725.7 Gbitops/s (u64) (64.5% of 1125 Gops/s (u64 bitop, est))
         w256_C64K: correct, 86.44 ms, 722.0 Gbitops/s (u64) (64.2% of 1125 Gops/s (u64 bitop, est))
  score (gmean of fraction): 0.6380

## History

- iter  0: compile=OK | correct=True | score=0.03891053861246893
- iter  1: compile=OK | correct=True | score=0.05283807961281304
- iter  2: compile=OK | correct=True | score=0.06711178931910174
- iter  3: compile=OK | correct=True | score=0.6293248516727208
- iter  4: compile=OK | correct=True | score=0.6302812102022003
- iter  5: compile=OK | correct=True | score=0.6098559015825479
- iter  6: compile=OK | correct=True | score=0.6379862316008869
- iter  7: compile=OK | correct=True | score=0.6278056530485507

## Instructions

Write an improved Metal kernel. Address the failure mode in the
previous attempt (if any), then push beyond the incumbent. Output ONE
fenced ```metal``` code block. Preserve kernel name(s) and buffer
indices.
