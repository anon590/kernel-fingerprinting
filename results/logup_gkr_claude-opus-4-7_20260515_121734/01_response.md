## Optimization

The seed's dominant cost is `mod_inv` via Fermat exponentiation — ~64 (BabyBear) or ~64 (Goldilocks) modular multiplications per element. The standard trick is **batch inversion (Montgomery's trick)**: replace N individual inversions with one inversion plus ~3N multiplications.

Better still: since each thread's term contributes to a *product*, we never actually need the individual inverses. We can fold numerators and denominators separately:

`P_tile = (prod num_k) * inv(prod denom_k)`

This reduces work to two threadgroup reductions plus **one** inversion per tile (done by thread 0), eliminating 255 inversions per tile. The tree reduction also uses `mod_mul` only, with a single final `mod_inv` call.

I also specialize `bb_mul` with Barrett-style reduction (still using `%` since it's correct and the bottleneck is gone), and keep Goldilocks helpers intact.

```metal
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant ulong P_GOLD  = 0xFFFFFFFF00000001ul;
constant ulong EPSILON = 0x00000000FFFFFFFFul;
constant ulong P_BB    = 2013265921ul;
constant ulong GOLD_INV_EXP = 0xFFFFFFFEFFFFFFFFul;
constant ulong BB_INV_EXP   = 2013265919ul;
constant uint  TG_WIDTH = 256u;

inline ulong gold_canonical(ulong x) {
    return (x >= P_GOLD) ? (x - P_GOLD) : x;
}

inline ulong gold_sub(ulong a, ulong b) {
    ulong t = a - b;
    if (t > a) t -= EPSILON;
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

inline ulong bb_sub(ulong a, ulong b) {
    return (a >= b) ? (a - b) : (a + P_BB - b);
}

inline ulong bb_mul(ulong a, ulong b) {
    return (a * b) % P_BB;
}

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
    threadgroup ulong num_scratch[TG_WIDTH];
    threadgroup ulong den_scratch[TG_WIDTH];

    uint total = N + M;
    ulong num_term;
    ulong den_term;

    if (gid < total) {
        ulong x;
        ulong num;
        if (gid < N) {
            x   = table[witness_idx[gid]];
            num = 1ul;
        } else {
            x   = table[gid - N];
            num = (ulong)multiplicities[gid - N];
        }
        num_term = num;
        den_term = mod_sub(alpha, x, prime_kind);
    } else {
        num_term = 1ul;
        den_term = 1ul;
    }

    // Special case: if num == 0 (a multiplicity is 0), the term is 0,
    // and the whole tile product becomes 0. We must propagate that.
    // We'll detect this by computing num_product separately; if it's 0,
    // we skip inversion. But mod_inv(den_product) is well-defined since
    // alpha avoids table values, so denom != 0.

    num_scratch[tid] = num_term;
    den_scratch[tid] = den_term;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TG_WIDTH >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            num_scratch[tid] = mod_mul(num_scratch[tid], num_scratch[tid + stride], prime_kind);
            den_scratch[tid] = mod_mul(den_scratch[tid], den_scratch[tid + stride], prime_kind);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        ulong num_prod = num_scratch[0];
        ulong den_prod = den_scratch[0];
        ulong result;
        if (num_prod == 0ul) {
            result = 0ul;
        } else {
            ulong inv_den = mod_inv(den_prod, prime_kind);
            result = mod_mul(num_prod, inv_den, prime_kind);
        }
        partial[tgid] = result;
    }
}
```