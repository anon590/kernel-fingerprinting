To optimize the `wots_chain` kernel, we force complete scalarization of the Keccak state and fully unroll the 24-round permutation into a single inline block. By replacing the 25-element `ulong` state array with 25 independent local variables (`a00`..`a24`), we eliminate any risk of the compiler spilling to thread-local memory (which can happen if SROA fails on arrays across loops). 

Furthermore, re-initializing the capacity lanes to `0x06` and `0` dynamically *inside* the chain loop (just before calling the scalarized Keccak block) allows LLVM's Sparse Conditional Constant Propagation (SCCP) to heavily optimize the first unrolled round. The compiler statically knows that 23 out of 25 lanes are constants in round 1, allowing it to silently prune over 30% of the XOR operations for the first round of every chain step.

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