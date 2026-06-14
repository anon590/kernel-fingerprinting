// Naive seed for the LogUp lookup-argument running product (Z7).
//
// Computes:
//   (1) multiplicities[j] = #{ i : witness_idx[i] == j }   for j in [0, M)
//   (2) the running product
//
//       P = prod_{i=0..N-1} 1/(alpha - w_i)
//          * prod_{j=0..M-1} m_j / (alpha - T_j)            (mod p)
//
//       where w_i := T[witness_idx[i]].
//
// Combined fingerprint stream of length N + M:
//   index k < N:   x_k = T[witness_idx[k]],   numerator num_k = 1
//   index k >= N:  x_k = T[k - N],            numerator num_k = m[k - N]
//
// The host issues TWO dispatches in this order, within ONE compute
// command encoder (so kernel B sees kernel A's atomic writes):
//
//   1) logup_count_mult
//        threadsPerGrid       = (N, 1, 1)         rounded up to TG width
//        threadsPerThreadgroup= (min(N, 256), 1, 1)
//        One thread per witness row; atomically increments
//        multiplicities[witness_idx[i]] by 1.
//
//   2) logup_partial_product
//        threadsPerGrid       = (ceil((N+M)/256) * 256, 1, 1)
//        threadsPerThreadgroup= (TG_WIDTH = 256, 1, 1)            // FIXED
//        Each threadgroup owns 256 consecutive indices in [0, N+M).
//        Each thread computes num_k * 1/(alpha - x_k) for its index
//        (or 1 for k >= N+M, the multiplicative identity), then the
//        threadgroup tree-reduces 256 terms into one tile product
//        written to partial[tgid].
//
// The host then reads partial[0..K-1] (K = ceil((N+M)/256)) and
// multiplies them on the CPU to obtain the final running product.
// (This sub-millisecond host-side fold is intentionally not timed.)
//
// Field selection (constant prime_kind):
//   0 = Goldilocks   p = 2^64 - 2^32 + 1
//   1 = BabyBear     p = 2^31 - 2^27 + 1 = 2013265921
// Both reductions are runtime-dispatched on prime_kind; a candidate
// that hardcodes the Goldilocks reduction silently fails the held-out
// BabyBear probe.
//
// Buffer layout (host-fixed, must be preserved by candidate kernels):
//
//   logup_count_mult:
//     buffer 0: device const uint  *witness_idx   (length N)
//     buffer 1: device atomic_uint *multiplicities(length M, zero-initialized by host)
//     buffer 2: constant uint &N
//
//   logup_partial_product:
//     buffer 0: device const ulong *table         (length M)
//     buffer 1: device const uint  *witness_idx   (length N)
//     buffer 2: device const uint  *multiplicities(length M)
//     buffer 3: device       ulong *partial       (length K = ceil((N+M)/256))
//     buffer 4: constant uint &N
//     buffer 5: constant uint &M
//     buffer 6: constant uint &prime_kind         (0 = Goldilocks, 1 = BabyBear)
//     buffer 7: constant ulong &alpha             (canonical, < p)
//
// All field elements (table, alpha, partial[]) are canonical uint64 in
// [0, p); a non-canonical output element is treated as a correctness
// failure even if its residue class matches the reference.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;       // 2^64 - 2^32 + 1
constant ulong EPSILON = 0x00000000FFFFFFFFul;       // 2^32 - 1
constant ulong P_BB    = 2013265921ul;               // 2^31 - 2^27 + 1
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;  // P_GOLD - 2
constant ulong BB_INV_EXP   = 2013265919ul;          // P_BB - 2

constant uint  TG_WIDTH = 256u;

// ---------------------- Goldilocks helpers ----------------------------

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;       // borrow -> fix with -= (2^32 - 1) mod p
    return t;
}

inline ulong2 umul128(ulong a, ulong b) {
    uint a0 = (uint)(a);
    uint a1 = (uint)(a >> 32);
    uint b0 = (uint)(b);
    uint b1 = (uint)(b >> 32);
    ulong p00 = (ulong)a0 * (ulong)b0;
    ulong p01 = (ulong)a0 * (ulong)b1;
    ulong p10 = (ulong)a1 * (ulong)b0;
    ulong p11 = (ulong)a1 * (ulong)b1;
    ulong mid = (p00 >> 32) + (p01 & EPSILON) + (p10 & EPSILON);
    ulong lo  = (p00 & EPSILON) | (mid << 32);
    ulong hi  = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return ulong2(lo, hi);
}

inline ulong gold_reduce128(ulong x_lo, ulong x_hi) {
    ulong x_hi_lo = x_hi & EPSILON;
    ulong x_hi_hi = x_hi >> 32;
    ulong t0 = x_lo - x_hi_hi;
    if (t0 > x_lo) t0 -= EPSILON;
    ulong t1 = x_hi_lo * EPSILON;
    ulong t2 = t0 + t1;
    if (t2 < t0) t2 += EPSILON;
    return gold_canonical(t2);
}

inline ulong gold_mul(ulong a, ulong b) {
    ulong2 prod = umul128(a, b);
    return gold_reduce128(prod.x, prod.y);
}

// ---------------------- BabyBear helpers ------------------------------
// All BabyBear elements fit in 31 bits, so a*b fits in 62 bits and the
// % operator is well-defined on uint64.

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

// ---------------------- Generic dispatch ------------------------------

inline ulong mod_sub(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_sub(a, b) : bb_sub(a, b);
}

inline ulong mod_mul(ulong a, ulong b, uint prime_kind) {
    return (prime_kind == 0u) ? gold_mul(a, b) : bb_mul(a, b);
}

inline ulong mod_pow(ulong base, ulong exp, uint prime_kind) {
    ulong r = 1ul;
    while (exp != 0ul) {
        if ((exp & 1ul) != 0ul) {
            r = mod_mul(r, base, prime_kind);
        }
        base = mod_mul(base, base, prime_kind);
        exp >>= 1;
    }
    return r;
}

inline ulong mod_inv(ulong x, uint prime_kind) {
    ulong exp = (prime_kind == 0u) ? GOLD_INV_EXP : BB_INV_EXP;
    return mod_pow(x, exp, prime_kind);
}

// ----------------------------------------------------------------------
// Kernel A: count multiplicities (atomic add).
// ----------------------------------------------------------------------

kernel void logup_count_mult(
    device const uint  *witness_idx    [[buffer(0)]],
    device atomic_uint *multiplicities [[buffer(1)]],
    constant uint      &N              [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= N) return;
    uint j = witness_idx[i];
    atomic_fetch_add_explicit(&multiplicities[j], 1u, memory_order_relaxed);
}

// ----------------------------------------------------------------------
// Kernel B: per-tile fingerprint inversion + threadgroup product
// reduction.
//
// Per-thread: term = num_k * inv(alpha - x_k) when k < N + M, else 1.
// Per-threadgroup: log2(TG_WIDTH)-step in-place tree reduction over
// scratch[]; thread 0 writes the tile product to partial[tgid].
// ----------------------------------------------------------------------

kernel void logup_partial_product(
    device const ulong *table         [[buffer(0)]],
    device const uint  *witness_idx   [[buffer(1)]],
    device const uint  *multiplicities[[buffer(2)]],
    device       ulong *partial       [[buffer(3)]],
    constant uint      &N             [[buffer(4)]],
    constant uint      &M             [[buffer(5)]],
    constant uint      &prime_kind    [[buffer(6)]],
    constant ulong     &alpha         [[buffer(7)]],
    uint gid  [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup ulong scratch[TG_WIDTH];

    uint total = N + M;
    ulong term;
    if (gid < total) {
        ulong x, num;
        if (gid < N) {
            x   = table[witness_idx[gid]];
            num = 1ul;
        } else {
            x   = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        ulong denom = mod_sub(alpha, x, prime_kind);
        ulong inv   = mod_inv(denom, prime_kind);
        term = mod_mul(num, inv, prime_kind);
    } else {
        term = 1ul;                  // multiplicative identity for OOB threads
    }

    scratch[tid] = term;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Power-of-two tree reduction (TG_WIDTH = 256 = 2^8).
    for (uint stride = TG_WIDTH >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] = mod_mul(scratch[tid], scratch[tid + stride], prime_kind);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        partial[tgid] = scratch[0];
    }
}
