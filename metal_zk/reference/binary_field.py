"""Binary-field carry-less multiplication reference (CPU).

The reference implements two parameter sets exercised by the Z11
``binius_clmul`` task:

- **GF(2^128)** (in-distribution): each element is a polynomial of
  degree < 128 over ``GF(2)``, multiplied modulo the AES-GCM
  irreducible polynomial ``R(x) = x^128 + x^7 + x^2 + x + 1``. Bit
  ``i`` of the lower ``u64`` limb represents ``x^i``; bit ``i`` of
  the upper limb represents ``x^(64 + i)``. The product of two
  128-bit operands is reduced via the standard 2-stage GCM
  reduction (the top 128 bits are folded into the bottom 128 bits
  using the low-7-bit pattern of ``R``; any residual overflow ---
  at most 7 bits --- is folded a second time).

- **GF(2^256) Fan-Hasan tower** (held-out probe): the tower
  ``GF(2^256) = GF(2^128)[v] / (v^2 + v + alpha)`` for a fixed
  ``alpha in GF(2^128)``. A product
  ``(a_0 + a_1 v) (b_0 + b_1 v) = c_0 + c_1 v`` decomposes as

      c_0 = a_0 b_0 + alpha * a_1 b_1
      c_1 = a_0 b_1 + a_1 b_0 + a_1 b_1

  (the ``a_1 b_1`` term in ``c_1`` comes from ``v^2 = v + alpha`` in
  characteristic 2). The reference computes it via four schoolbook
  GF(2^128) muls plus the ``alpha`` scaling; a Karatsuba candidate
  can reach the same result with three sub-muls + the alpha scaling.

  We pick ``alpha`` such that ``Tr_{GF(2^128) / GF(2)}(alpha) = 1``
  --- the Artin-Schreier criterion for ``v^2 + v + alpha`` to be
  irreducible over ``GF(2^128)``. The fixed choice
  ``alpha = x^127`` (a monomial; ``alpha_lo = 0``,
  ``alpha_hi = 0x8000_0000_0000_0000``) has trace 1 under the
  AES-GCM polynomial and is cached in :data:`DEFAULT_TOWER_ALPHA`.

The held-out probe is structurally distinct from the in-distribution
sizes along two axes simultaneously:

  - element width doubles (4 u64s per element vs 2);
  - multiplication shape changes (one polynomial mul + reduction vs
    three to five polynomial muls + alpha scaling + cross-coeff
    sum).

A candidate that hardcodes ``field_words = 2``, that ignores the
``tower`` flag, or that bakes the 128-bit irreducible polynomial as
the entire reduction (no alpha scaling) silently produces garbage
on the held-out probe.

Buffer layout convention
========================
For a batch of ``N`` elements with ``field_words`` u64 limbs per
element, the storage is a flat ``(N * field_words,)`` ``np.uint64``
array. Element ``i`` occupies limbs
``[i * field_words .. i * field_words + field_words)``; within each
element, limb 0 is the least significant 64 bits, limb 1 the next
significant 64 bits, etc. For the tower (field_words = 4), limbs 0,
1 form the ``v^0`` coefficient ``a_0 in GF(2^128)`` and limbs 2, 3
form the ``v^1`` coefficient ``a_1``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# R(x) = x^128 + x^7 + x^2 + x + 1 (AES-GCM convention). Stored as the
# low-127-bit pattern only --- the leading x^128 is implicit.
IRR_LOW_BITS: int = (1 << 7) | (1 << 2) | (1 << 1) | 1     # 0x87
MASK128: int = (1 << 128) - 1
MASK64: int = (1 << 64) - 1


# ----------------------------------------------------------------------
# Single-element GF(2^128) operations (slow, used only for cross-checks)
# ----------------------------------------------------------------------

def gf128_clmul(a: int, b: int) -> int:
    """Carry-less product of two 128-bit polynomials, unreduced (256 bits)."""
    p = 0
    while b:
        if b & 1:
            p ^= a
        a <<= 1
        b >>= 1
    return p


def gf128_reduce(p: int) -> int:
    """Reduce a polynomial of degree < 256 modulo ``R(x)`` to degree < 128."""
    while p >> 128:
        hi = p >> 128
        p &= MASK128
        p ^= hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7)
    return p


def gf128_mul(a: int, b: int) -> int:
    """GF(2^128) field multiplication modulo ``R(x)``."""
    return gf128_reduce(gf128_clmul(a & MASK128, b & MASK128))


def gf128_square(a: int) -> int:
    return gf128_mul(a, a)


def gf128_trace(a: int) -> int:
    """Absolute trace ``Sum_{i=0}^{127} a^(2^i)`` in ``GF(2)``."""
    s = 0
    t = a & MASK128
    for _ in range(128):
        s ^= t
        t = gf128_square(t)
    # Trace lives in GF(2); the 128-bit accumulator should have at
    # most one nonzero bit, in position 0.
    return s & 1


# Locked-in tower alpha. The Artin-Schreier criterion for
# ``v^2 + v + alpha`` to be irreducible over GF(2^128) is
# ``Tr_{GF(2^128) / GF(2)}(alpha) = 1``. Every element of the
# proper subfields ``GF(2^d)`` for ``d | 128, d < 128`` has trace 0
# (the trace from GF(2^128) to GF(2) factors through ``(128 / d) *
# Tr_d``, which vanishes for ``d <= 64``); a candidate must therefore
# lie outside ``GF(2^64)``. The monomial ``alpha = x^127`` is the
# simplest such element and is verified trace-1 by
# :func:`gf128_trace`.
DEFAULT_TOWER_ALPHA: int = 1 << 127


# ----------------------------------------------------------------------
# Single-element GF(2^256) Fan-Hasan tower mul
# ----------------------------------------------------------------------

def gf256_tower_mul(a: int, b: int, alpha: int = DEFAULT_TOWER_ALPHA) -> int:
    """Multiply two 256-bit elements over ``GF(2^128)[v] / (v^2 + v + alpha)``.

    The element ``a`` is encoded as ``a_0 + a_1 * 2^128`` where ``a_0``,
    ``a_1`` are 128-bit ``GF(2^128)`` elements (the ``v^0`` and ``v^1``
    coefficients respectively).
    """
    a0 = a & MASK128
    a1 = (a >> 128) & MASK128
    b0 = b & MASK128
    b1 = (b >> 128) & MASK128
    m00 = gf128_mul(a0, b0)
    m01 = gf128_mul(a0, b1)
    m10 = gf128_mul(a1, b0)
    m11 = gf128_mul(a1, b1)
    c0 = m00 ^ gf128_mul(alpha, m11)
    c1 = m01 ^ m10 ^ m11
    return (c0 & MASK128) | ((c1 & MASK128) << 128)


# ----------------------------------------------------------------------
# Vectorised GF(2^128) mul over numpy uint64 arrays
# ----------------------------------------------------------------------

def _split128(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 2)`` packed limb view -> ``(lo, hi)`` 1-d uint64 arrays."""
    flat = np.ascontiguousarray(elements, dtype=np.uint64).reshape(-1, 2)
    return flat[:, 0].copy(), flat[:, 1].copy()


def gf128_mul_vec(
    a_lo: np.ndarray, a_hi: np.ndarray,
    b_lo: np.ndarray, b_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched GF(2^128) multiply on uint64 lo/hi limb pairs.

    Operates as a bit-by-bit shift-XOR sweep across 128 bits of ``b``,
    vectorised across the batch axis via numpy. 128 numpy passes ---
    each pass touches ``O(N)`` u64s --- so total cost is ``O(128 N)``
    u64 ops, which is roughly the same per-element work as the GPU
    kernel does. For ``N = 2^20`` this completes in 1-2 seconds on a
    laptop CPU; the disk cache amortises that across reps.
    """
    if a_lo.shape != a_hi.shape or a_lo.shape != b_lo.shape \
            or a_lo.shape != b_hi.shape:
        raise ValueError("a_lo / a_hi / b_lo / b_hi must share shape")
    a_lo = a_lo.astype(np.uint64, copy=False)
    a_hi = a_hi.astype(np.uint64, copy=False)
    b_lo = b_lo.astype(np.uint64, copy=False)
    b_hi = b_hi.astype(np.uint64, copy=False)

    zero = np.zeros_like(a_lo)
    t0 = zero.copy()
    t1 = zero.copy()
    t2 = zero.copy()
    t3 = zero.copy()
    one = np.uint64(1)

    for i in range(128):
        if i < 64:
            bit = (b_lo >> np.uint64(i)) & one
        else:
            bit = (b_hi >> np.uint64(i - 64)) & one
        mask = np.uint64(0) - bit                  # 0 or 0xFFFFFFFFFFFFFFFF

        # (a_lo, a_hi) << i across the 256-bit destination (t0, t1, t2, t3).
        if i == 0:
            sh0, sh1, sh2 = a_lo, a_hi, zero
            sh3 = zero
            t0 ^= sh0 & mask
            t1 ^= sh1 & mask
            t2 ^= sh2 & mask
        elif i < 64:
            k = np.uint64(i)
            nk = np.uint64(64 - i)
            sh0 = (a_lo << k)
            sh1 = (a_hi << k) | (a_lo >> nk)
            sh2 = (a_hi >> nk)
            t0 ^= sh0 & mask
            t1 ^= sh1 & mask
            t2 ^= sh2 & mask
        elif i == 64:
            t1 ^= a_lo & mask
            t2 ^= a_hi & mask
        else:
            j = i - 64
            k = np.uint64(j)
            nk = np.uint64(64 - j)
            sh1 = (a_lo << k)
            sh2 = (a_hi << k) | (a_lo >> nk)
            sh3 = (a_hi >> nk)
            t1 ^= sh1 & mask
            t2 ^= sh2 & mask
            t3 ^= sh3 & mask

    # Stage 1: fold (t2, t3) into (t0, t1) via R(x)'s low pattern
    # x^7 + x^2 + x + 1.
    one64 = np.uint64(1)
    two   = np.uint64(2)
    seven = np.uint64(7)
    s63 = np.uint64(63)
    s62 = np.uint64(62)
    s57 = np.uint64(57)

    d_lo0 = t2 ^ (t2 << one64) ^ (t2 << two) ^ (t2 << seven)
    d_lo1 = (t3
             ^ ((t3 << one64) | (t2 >> s63))
             ^ ((t3 << two)   | (t2 >> s62))
             ^ ((t3 << seven) | (t2 >> s57)))
    d_hi  = (t3 >> s63) ^ (t3 >> s62) ^ (t3 >> s57)
    t0 ^= d_lo0
    t1 ^= d_lo1

    # Stage 2: residual ``d_hi`` lives in bits 128..134; fold it once
    # more. (d_hi << 7) ^ (d_hi << 2) ^ (d_hi << 1) ^ d_hi sits within
    # bits 0..13, well inside t0.
    t0 ^= d_hi ^ (d_hi << one64) ^ (d_hi << two) ^ (d_hi << seven)
    return t0, t1


# ----------------------------------------------------------------------
# Vectorised GF(2^256) tower mul
# ----------------------------------------------------------------------

def gf256_tower_mul_vec(
    a: np.ndarray, b: np.ndarray, alpha_lo: int, alpha_hi: int,
) -> np.ndarray:
    """Batched GF(2^256) Fan-Hasan tower multiplication.

    ``a`` and ``b`` are ``(N, 4)`` uint64 arrays (or equivalently flat
    ``(N * 4,)`` arrays reshaped on entry). Limbs 0, 1 of each row
    form the ``v^0`` coefficient; limbs 2, 3 form ``v^1``.
    Returns a contiguous ``(N, 4)`` uint64 array.
    """
    a = np.ascontiguousarray(a, dtype=np.uint64).reshape(-1, 4)
    b = np.ascontiguousarray(b, dtype=np.uint64).reshape(-1, 4)
    if a.shape != b.shape:
        raise ValueError(f"a / b shape mismatch: {a.shape} vs {b.shape}")
    n = a.shape[0]

    a0_lo, a0_hi = a[:, 0].copy(), a[:, 1].copy()
    a1_lo, a1_hi = a[:, 2].copy(), a[:, 3].copy()
    b0_lo, b0_hi = b[:, 0].copy(), b[:, 1].copy()
    b1_lo, b1_hi = b[:, 2].copy(), b[:, 3].copy()

    m00_lo, m00_hi = gf128_mul_vec(a0_lo, a0_hi, b0_lo, b0_hi)
    m01_lo, m01_hi = gf128_mul_vec(a0_lo, a0_hi, b1_lo, b1_hi)
    m10_lo, m10_hi = gf128_mul_vec(a1_lo, a1_hi, b0_lo, b0_hi)
    m11_lo, m11_hi = gf128_mul_vec(a1_lo, a1_hi, b1_lo, b1_hi)

    alpha_lo_arr = np.full(n, np.uint64(alpha_lo & MASK64), dtype=np.uint64)
    alpha_hi_arr = np.full(n, np.uint64(alpha_hi & MASK64), dtype=np.uint64)
    am_lo, am_hi = gf128_mul_vec(alpha_lo_arr, alpha_hi_arr, m11_lo, m11_hi)

    c0_lo = m00_lo ^ am_lo
    c0_hi = m00_hi ^ am_hi
    c1_lo = m01_lo ^ m10_lo ^ m11_lo
    c1_hi = m01_hi ^ m10_hi ^ m11_hi

    out = np.empty((n, 4), dtype=np.uint64)
    out[:, 0] = c0_lo
    out[:, 1] = c0_hi
    out[:, 2] = c1_lo
    out[:, 3] = c1_hi
    return out


# ----------------------------------------------------------------------
# Parameter spec + input generator
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FieldParams:
    name: str               # "gf128" or "gf256_tower"
    field_words: int        # u64 limbs per element (2 or 4)
    tower: int              # 0 or 1, matches the kernel's tower flag
    alpha_lo: int           # tower alpha (low 64 bits); 0 for non-tower
    alpha_hi: int           # tower alpha (high 64 bits); 0 for non-tower


GF128 = FieldParams(
    name="gf128", field_words=2, tower=0,
    alpha_lo=0, alpha_hi=0,
)
GF256_TOWER = FieldParams(
    name="gf256_tower", field_words=4, tower=1,
    alpha_lo=DEFAULT_TOWER_ALPHA & MASK64,
    alpha_hi=(DEFAULT_TOWER_ALPHA >> 64) & MASK64,
)


def params_for(variant: str) -> FieldParams:
    if variant == "gf128":
        return GF128
    if variant == "gf256_tower":
        return GF256_TOWER
    raise ValueError(f"unknown binius_clmul variant: {variant!r}")


def random_inputs(n: int, params: FieldParams, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, b)`` as contiguous ``(n * field_words,)`` uint64 buffers."""
    rng = np.random.default_rng(seed)
    a = rng.integers(
        0, 1 << 64, size=(n * params.field_words,),
        dtype=np.uint64, endpoint=False,
    )
    b = rng.integers(
        0, 1 << 64, size=(n * params.field_words,),
        dtype=np.uint64, endpoint=False,
    )
    return a, b


# ----------------------------------------------------------------------
# Top-level reference + disk cache
# ----------------------------------------------------------------------

def multiply_reference(
    a: np.ndarray, b: np.ndarray, params: FieldParams,
) -> np.ndarray:
    """Bit-exact reference output for one (a, b, params) triple.

    Returns a flat ``(n * field_words,)`` uint64 array matching the
    kernel's output buffer layout.
    """
    fw = params.field_words
    a = np.ascontiguousarray(a, dtype=np.uint64).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.uint64).reshape(-1)
    if a.size % fw or b.size % fw:
        raise ValueError(
            f"input size {a.size} / {b.size} not a multiple of "
            f"field_words = {fw}"
        )
    if a.size != b.size:
        raise ValueError(f"a / b size mismatch: {a.size} vs {b.size}")

    if params.tower == 0:
        a_r = a.reshape(-1, 2)
        b_r = b.reshape(-1, 2)
        c_lo, c_hi = gf128_mul_vec(
            a_r[:, 0], a_r[:, 1], b_r[:, 0], b_r[:, 1],
        )
        out = np.empty_like(a_r)
        out[:, 0] = c_lo
        out[:, 1] = c_hi
        return out.reshape(-1)

    # Tower
    return gf256_tower_mul_vec(
        a, b, params.alpha_lo, params.alpha_hi,
    ).reshape(-1)


_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(
        os.environ.get("METAL_ZK_CACHE", "~/.cache/metal-zk")
    ).expanduser()
    d = root / "binius_clmul"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(a: np.ndarray, b: np.ndarray, params: FieldParams) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(params.name.encode())
    h.update(np.uint32(params.field_words).tobytes())
    h.update(np.uint32(params.tower).tobytes())
    h.update(np.uint64(params.alpha_lo).tobytes())
    h.update(np.uint64(params.alpha_hi).tobytes())
    h.update(np.uint64(a.size).tobytes())
    h.update(np.ascontiguousarray(a, dtype=np.uint64).tobytes())
    h.update(np.ascontiguousarray(b, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def multiply_cached(
    a: np.ndarray, b: np.ndarray, params: FieldParams,
) -> np.ndarray:
    """``multiply_reference`` with an in-memory + on-disk cache.

    Keys on params + input bytes, so an in-dist size and the held-out
    tower size land in distinct cache files even if their element
    counts coincide.
    """
    a_u = np.ascontiguousarray(a, dtype=np.uint64).reshape(-1)
    b_u = np.ascontiguousarray(b, dtype=np.uint64).reshape(-1)
    key = _cache_key(a_u, b_u, params)
    mem_key = f"{params.name}_n{a_u.size // params.field_words}_{key}"
    cached = _MEM_CACHE.get(mem_key)
    if cached is not None:
        return cached
    path = _cache_dir() / (
        f"{params.name}_n{a_u.size // params.field_words}_{key}.u64"
    )
    if path.exists():
        out = np.fromfile(path, dtype=np.uint64)
        if out.size == a_u.size:
            _MEM_CACHE[mem_key] = out
            return out
    out = multiply_reference(a_u, b_u, params)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = out
    return out


__all__ = [
    "IRR_LOW_BITS", "MASK128", "MASK64",
    "DEFAULT_TOWER_ALPHA",
    "FieldParams", "GF128", "GF256_TOWER", "params_for",
    "gf128_clmul", "gf128_reduce", "gf128_mul", "gf128_square",
    "gf128_trace",
    "gf256_tower_mul",
    "gf128_mul_vec", "gf256_tower_mul_vec",
    "random_inputs",
    "multiply_reference", "multiply_cached",
]
