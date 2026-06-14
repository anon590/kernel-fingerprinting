"""ML-KEM (Kyber) and ML-DSA (Dilithium) negacyclic NTT reference (CPU).

Both schemes operate on degree-256 polynomials over a small prime field
and use a forward Cooley-Tukey NTT with bit-reversed twiddle indexing
(matches the pqclean / FIPS 203 / FIPS 204 reference C implementations).

Two parameter sets ship here:

- ``KYBER`` (Z6 in-distribution): ``q = 3329`` (12-bit), ``n = 256``,
  ``n_levels = 7``. The polynomial ring ``Z_q[X] / (X^256 + 1)`` factors
  into 128 quadratic residues over ``F_q`` because ``q - 1 = 2^8 * 13``
  is divisible by ``2n / 2 = 256`` but **not** by ``2n = 512``. The
  primitive 256-th root of unity is ``zeta = 17``, with ``zeta^128 = -1
  (mod q)``. Output is 128 length-2 polynomials in pqclean's permuted
  order (the natural output of bit-reversed-twiddle Cooley-Tukey).

- ``DILITHIUM`` (Z6 held-out probe): ``q = 8380417 = 2^23 - 2^13 + 1``
  (23-bit), ``n = 256``, ``n_levels = 8``. Here ``q - 1 = 2^13 * 1023``
  is divisible by ``2n = 512``, so ``X^256 + 1`` fully factors into 256
  linear residues. The primitive 512-th root of unity is ``zeta = 1753``.
  Output is 256 scalars in the same bit-reversed order.

The held-out probe flips on candidate kernels that:
  * hardcode ``q = 3329`` (Dilithium's modulus is 23-bit, not 12-bit);
  * hardcode 16-bit ``ushort`` coefficient storage (Dilithium needs
    full 32-bit storage and 32-bit-aware Barrett / Montgomery
    constants);
  * hardcode 7 NTT levels (Dilithium does 8);
  * hardcode the zetas-table length (Kyber: 128 entries; Dilithium:
    256 entries).

Zetas convention
================

``zetas[k] = zeta^bit_reverse(k, n_levels) mod q`` for
``k in [0, 2^n_levels)``. The forward NTT consumes them starting at
``k = 1`` (entry ``zetas[0]`` is the identity, never read by the
butterfly loop):

  k = 1
  for len = n/2, n/4, ..., n / 2^n_levels:
      for start = 0, 2*len, 4*len, ..., n - 2*len:
          z = zetas[k]; k += 1
          for j in [start, start + len):
              t          = z * a[j + len] mod q
              a[j + len] = (a[j] - t)      mod q
              a[j]       = (a[j] + t)      mod q

After all levels, ``k`` equals ``2^n_levels`` (127 reads for Kyber
because we start at 1 and stop before the final ``len = 1`` level;
255 reads for Dilithium because it includes that level).

Reference outputs are bit-exactly comparable to pqclean's
``Kyber_ntt`` and ``Dilithium_ntt`` (modulo pqclean storing values in
Montgomery form, which we don't — we keep canonical ``[0, q)``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


KYBER_Q: int = 3329
KYBER_N: int = 256
KYBER_N_LEVELS: int = 7
KYBER_ZETA: int = 17            # primitive 256-th root of unity in F_q

DILITHIUM_Q: int = 8380417
DILITHIUM_N: int = 256
DILITHIUM_N_LEVELS: int = 8
DILITHIUM_ZETA: int = 1753      # primitive 512-th root of unity in F_q


@dataclass(frozen=True)
class NttParams:
    """One (q, n, n_levels, zeta) parameter quadruple.

    ``zeta`` is a primitive ``2^(n_levels + 1)``-th root of unity in
    ``F_q`` for Dilithium-class parameters where the polynomial fully
    factors; for Kyber-class parameters (where ``ord(zeta) = 2 * n /
    2 = n_zetas``) the same convention holds because the NTT stops one
    level short of full factorisation.
    """
    q: int
    n: int
    n_levels: int
    zeta: int
    name: str

    @property
    def n_zetas(self) -> int:
        """Length of the precomputed zetas table = 1 << n_levels."""
        return 1 << self.n_levels


KYBER = NttParams(KYBER_Q, KYBER_N, KYBER_N_LEVELS, KYBER_ZETA, "kyber")
DILITHIUM = NttParams(
    DILITHIUM_Q, DILITHIUM_N, DILITHIUM_N_LEVELS, DILITHIUM_ZETA, "dilithium",
)


def _bit_reverse(k: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


@lru_cache(maxsize=None)
def make_zetas(params: NttParams) -> np.ndarray:
    """Precomputed twiddle table: ``zetas[k] = zeta^br(k) mod q``.

    Length is ``params.n_zetas`` (128 for Kyber, 256 for Dilithium);
    entry 0 holds ``zeta^0 = 1`` and is never read by the NTT loop
    (which starts at ``k = 1``). Cached because every test rep at the
    same size re-derives the same table.
    """
    n_zetas = params.n_zetas
    bits = params.n_levels
    out = np.empty(n_zetas, dtype=np.uint32)
    for k in range(n_zetas):
        out[k] = pow(params.zeta, _bit_reverse(k, bits), params.q)
    return out


def random_inputs(batch: int, params: NttParams, seed: int) -> np.ndarray:
    """``(batch, n)`` random canonical-form coefficients in ``[0, q)``."""
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, params.q, size=(batch, params.n), dtype=np.uint32, endpoint=False,
    )


# ----------------------------------------------------------------------
# Forward NTT (reference)
# ----------------------------------------------------------------------

def ntt_forward_reference(coeffs: np.ndarray, params: NttParams) -> np.ndarray:
    """Forward negacyclic NTT (batched), bit-exact ``uint32`` output.

    ``coeffs`` is ``(batch, n)`` or ``(n,)`` of canonical ``[0, q)``
    values; the returned array has the same shape and dtype
    ``uint32``. All intermediate arithmetic is done in ``uint64`` to
    avoid overflow (Dilithium products span 46 bits).
    """
    coeffs = np.asarray(coeffs, dtype=np.uint64).copy()
    squeezed = (coeffs.ndim == 1)
    if squeezed:
        coeffs = coeffs.reshape(1, -1)
    if coeffs.shape[1] != params.n:
        raise ValueError(
            f"polynomial length {coeffs.shape[1]} != n = {params.n}"
        )
    if not np.all(coeffs < params.q):
        raise ValueError("input contains non-canonical coefficients (>= q)")

    zetas = make_zetas(params).astype(np.uint64)
    q = np.uint64(params.q)

    k = 1
    length = params.n // 2
    for _level in range(params.n_levels):
        start = 0
        while start < params.n:
            zeta = zetas[k]
            k += 1
            low  = coeffs[:, start:start + length]
            high = coeffs[:, start + length:start + 2 * length]
            t = (zeta * high) % q
            # a[j + len] = (a[j] - t) mod q,  a[j] = (a[j] + t) mod q
            new_high = (low + (q - t)) % q
            new_low  = (low + t)        % q
            coeffs[:, start:start + length]              = new_low
            coeffs[:, start + length:start + 2 * length] = new_high
            start += 2 * length
        length >>= 1
    assert k == params.n_zetas, f"used {k - 1} zetas, expected {params.n_zetas - 1}"

    out = coeffs.astype(np.uint32)
    if squeezed:
        out = out.reshape(-1)
    return out


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(os.environ.get("METAL_ZK_CACHE",
                               "~/.cache/metal-zk")).expanduser()
    d = root / "kyber_ntt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(params: NttParams, coeffs: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(params.name.encode())
    h.update(np.uint32(params.q).tobytes())
    h.update(np.uint32(params.n).tobytes())
    h.update(np.uint32(params.n_levels).tobytes())
    h.update(np.uint32(params.zeta).tobytes())
    h.update(np.ascontiguousarray(coeffs, dtype=np.uint32).tobytes())
    return h.hexdigest()[:16]


def ntt_forward_cached(coeffs: np.ndarray, params: NttParams) -> np.ndarray:
    """``ntt_forward_reference`` with disk + memory caching.

    The reference is fast (a few hundred ms at batch=256) so caching is
    a nice-to-have, but it adds up across reps * iterations during a
    long evolution run. Keys on params + input bytes so different
    in-dist / held-out probes land in different cache files.
    """
    coeffs32 = np.ascontiguousarray(coeffs, dtype=np.uint32)
    if coeffs32.ndim == 1:
        coeffs32 = coeffs32.reshape(1, -1)
    batch, n = coeffs32.shape
    key = _cache_key(params, coeffs32)
    mem_key = f"{params.name}_B{batch}_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    path = _cache_dir() / f"{params.name}_B{batch}_{key}.u32"
    if path.exists():
        out = np.fromfile(path, dtype=np.uint32)
        if out.size == batch * n:
            out = out.reshape(batch, n)
            _MEM_CACHE[mem_key] = out
            return out
    out = ntt_forward_reference(coeffs32, params)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = out
    return out


__all__ = [
    "NttParams",
    "KYBER", "DILITHIUM",
    "KYBER_Q", "KYBER_N", "KYBER_N_LEVELS", "KYBER_ZETA",
    "DILITHIUM_Q", "DILITHIUM_N", "DILITHIUM_N_LEVELS", "DILITHIUM_ZETA",
    "make_zetas",
    "random_inputs",
    "ntt_forward_reference",
    "ntt_forward_cached",
]
