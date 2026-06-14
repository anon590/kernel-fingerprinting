"""LogUp lookup-argument reference (CPU).

The LogUp identity (Haebock 2022) for a witness column ``w[0..N)`` drawn
from a table ``T[0..M)`` says::

    sum_i 1 / (alpha - w_i)   ==   sum_j m_j / (alpha - T_j)

where ``alpha`` is a verifier challenge and ``m_j`` is the multiplicity
of ``T_j`` in ``w``. The PLAN.md Z7 sketch asks for the *running product*
of the same fingerprint terms::

    P = prod_i 1/(alpha - w_i) * prod_j m_j / (alpha - T_j)   (mod p)

This module is the bit-exact CPU oracle for the Metal-ZK Z7 task. It

  1. generates a deterministic ``(table, witness_idx, alpha)`` triple for
     a given ``(M, N, prime_kind)``, with the property that ``alpha``
     never collides with a table element (no zero denominators) and
     every table row receives multiplicity at least 1 (a permutation
     prefix in ``witness_idx``);
  2. computes the multiplicities ``m[M]``;
  3. inverts all ``N + M`` denominators via Montgomery's trick in pure
     Python ``int`` arithmetic (so the reference is unambiguously bit
     exact across primes -- Goldilocks ``p = 2^64 - 2^32 + 1`` or
     BabyBear ``p = 2^31 - 2^27 + 1``);
  4. returns the running product as a single canonical field element.

Results are disk-cached under ``~/.cache/metal-zk/logup/`` keyed by
``sha256(table | witness_idx | alpha | prime_kind)``. The largest
in-distribution size (M = 2^20) takes ~10 s the first time and is
served from disk thereafter.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


P_GOLD: int = (1 << 64) - (1 << 32) + 1            # Goldilocks
P_BB:   int = (1 << 31) - (1 << 27) + 1            # BabyBear = 2013265921

PRIMES: dict[int, int] = {0: P_GOLD, 1: P_BB}
PRIME_NAMES: dict[int, str] = {0: "Goldilocks", 1: "BabyBear"}


def prime_of(prime_kind: int) -> int:
    if prime_kind not in PRIMES:
        raise ValueError(
            f"unknown prime_kind={prime_kind}; must be 0 (Goldilocks) or 1 (BabyBear)"
        )
    return PRIMES[prime_kind]


# ----------------------------------------------------------------------
# Input generation
# ----------------------------------------------------------------------

def _random_field_elements(n: int, prime: int, seed: int) -> np.ndarray:
    """``n`` canonical field elements in ``[0, prime)`` as ``uint64``.

    Rejection sampling on a uint64 stream; rejection rate is
    ``(2^64 - prime) / 2^64`` which is ``~2^-32`` for Goldilocks and
    ``~1 - 2^-33`` for BabyBear. To keep BabyBear cheap we instead
    sample modulo ``2^32`` and reject, giving a ~2x oversample.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint64)
    filled = 0
    if prime <= (1 << 32):
        # 32-bit prime path (BabyBear).
        while filled < n:
            chunk = max(1, (n - filled) * 3)
            raw32 = rng.integers(
                0, 1 << 32, size=chunk, dtype=np.uint32, endpoint=False,
            )
            accepted = raw32[raw32 < np.uint32(prime)]
            take = min(len(accepted), n - filled)
            out[filled:filled + take] = accepted[:take].astype(np.uint64)
            filled += take
    else:
        while filled < n:
            chunk = max(1, (n - filled) * 2)
            raw = rng.integers(
                0, 1 << 64, size=chunk, dtype=np.uint64, endpoint=False,
            )
            accepted = raw[raw < np.uint64(prime)]
            take = min(len(accepted), n - filled)
            out[filled:filled + take] = accepted[:take]
            filled += take
    return out


def generate_inputs(
    M: int, N: int, prime_kind: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate deterministic ``(table, witness_idx, alpha)``.

    - ``table``: ``uint64[M]``, canonical field elements in ``[0, p)``.
    - ``witness_idx``: ``uint32[N]``, indices into ``table``. The first
      ``M`` positions are a random permutation of ``[0, M)`` (guarantees
      every multiplicity is at least 1); the remaining ``N - M``
      positions are uniform in ``[0, M)``.
    - ``alpha``: Python ``int`` in ``[0, p)`` and outside the set of
      table values (so no denominator ``alpha - T_j`` is zero).

    Raises ``ValueError`` if ``N < M``.
    """
    if N < M:
        raise ValueError(f"N={N} must be >= M={M}")
    prime = prime_of(prime_kind)

    table = _random_field_elements(M, prime, seed=seed)

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(M).astype(np.uint32)
    rest = rng.integers(0, M, size=N - M, dtype=np.uint32, endpoint=False)
    witness_idx = np.concatenate([perm, rest]).astype(np.uint32)

    h = hashlib.sha256()
    h.update(b"metal-zk:logup:alpha")
    h.update(np.uint64(N).tobytes())
    h.update(np.uint64(M).tobytes())
    h.update(np.uint32(prime_kind).tobytes())
    h.update(np.uint64(seed & ((1 << 64) - 1)).tobytes())
    alpha = int.from_bytes(h.digest()[:8], "big") % prime
    table_set = set(int(v) for v in table.tolist())
    while alpha in table_set:
        alpha = (alpha + 1) % prime

    return table, witness_idx, int(alpha)


# ----------------------------------------------------------------------
# Reference computation
# ----------------------------------------------------------------------

def compute_reference(
    table: np.ndarray,
    witness_idx: np.ndarray,
    alpha: int,
    prime_kind: int,
) -> tuple[np.ndarray, int]:
    """Returns ``(multiplicities, running_product)``.

    - ``multiplicities``: ``uint32[M]`` counts of how many witness rows
      map to each table index.
    - ``running_product``: canonical Python int in ``[0, p)``; the
      product of ``num_k * inv(alpha - x_k)`` over the combined
      fingerprint stream of length ``N + M``, where ``num_k = 1`` for
      witness rows (``k < N``) and ``num_k = m[k - N]`` for table rows.

    Batched inversion uses Montgomery's trick (3*(N+M) modmuls + one
    full Fermat inverse) on Python ints.
    """
    prime = prime_of(prime_kind)
    M = int(table.shape[0])
    N = int(witness_idx.shape[0])

    table_list = [int(v) for v in table.tolist()]
    widx_list = [int(v) for v in witness_idx.tolist()]

    # Multiplicities via numpy.add.at for speed.
    multiplicities = np.zeros(M, dtype=np.uint32)
    np.add.at(multiplicities, witness_idx.astype(np.int64), np.uint32(1))

    # Denominators d_k = (alpha - x_k) mod p.
    nlen = N + M
    denoms = [0] * nlen
    for i in range(N):
        denoms[i] = (alpha - table_list[widx_list[i]]) % prime
    for j in range(M):
        denoms[N + j] = (alpha - table_list[j]) % prime

    # Sanity: any zero denom would break Montgomery's trick + leak the
    # bug into the GPU oracle. The input generator excludes alpha == T_j
    # so this should never fire, but assert in case the cache key drifts.
    for d in denoms:
        if d == 0:
            raise RuntimeError(
                "zero denominator in LogUp reference -- alpha collides "
                "with some table element; check generate_inputs."
            )

    # Montgomery's trick: prefix[i] = prod_{j < i} denoms[j]; then one
    # full inverse, then sweep back.
    prefix = [1] * (nlen + 1)
    for i in range(nlen):
        prefix[i + 1] = (prefix[i] * denoms[i]) % prime
    q = pow(prefix[nlen], prime - 2, prime)         # = 1 / prod d
    inverses = [0] * nlen
    for i in range(nlen - 1, -1, -1):
        inverses[i] = (q * prefix[i]) % prime       # = 1 / d_i
        q = (q * denoms[i]) % prime                 # = 1 / prod_{j<i} d_j

    # Running product: prod_k num_k * inv_k.
    mult_list = [int(v) for v in multiplicities.tolist()]
    rp = 1
    for k in range(N):
        rp = (rp * inverses[k]) % prime             # num = 1
    for j in range(M):
        rp = (rp * mult_list[j] % prime * inverses[N + j]) % prime

    return multiplicities, int(rp)


# ----------------------------------------------------------------------
# Disk cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"


def _cache_dir() -> Path:
    root = Path(os.environ.get("METAL_ZK_CACHE",
                               "~/.cache/metal-zk")).expanduser()
    d = root / "logup"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    table: np.ndarray, witness_idx: np.ndarray, alpha: int, prime_kind: int,
) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint32(prime_kind).tobytes())
    h.update(np.uint64(alpha).tobytes())
    h.update(np.ascontiguousarray(table, dtype=np.uint64).tobytes())
    h.update(np.ascontiguousarray(witness_idx, dtype=np.uint32).tobytes())
    return h.hexdigest()[:16]


def compute_reference_cached(
    table: np.ndarray,
    witness_idx: np.ndarray,
    alpha: int,
    prime_kind: int,
) -> tuple[np.ndarray, int]:
    """``compute_reference`` with disk-backed caching."""
    M = int(table.shape[0])
    N = int(witness_idx.shape[0])
    key = _cache_key(table, witness_idx, alpha, prime_kind)
    path = _cache_dir() / f"M{M}_N{N}_pk{prime_kind}_{key}.npz"
    if path.exists():
        try:
            z = np.load(path)
            mult = z["mult"]
            prod = int(z["prod"].item())
            if mult.shape == (M,) and mult.dtype == np.uint32:
                return mult, prod
        except Exception:
            # Corrupted file -- fall through and recompute.
            pass
    mult, prod = compute_reference(table, witness_idx, alpha, prime_kind)
    # np.savez auto-appends ".npz" if missing; use a tmp that already
    # ends in .npz so the rename target is what we wrote.
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez(
        tmp,
        mult=mult.astype(np.uint32),
        prod=np.array(prod, dtype=np.uint64),
    )
    os.replace(tmp, path)
    return mult, prod


__all__ = [
    "P_GOLD",
    "P_BB",
    "PRIMES",
    "PRIME_NAMES",
    "prime_of",
    "generate_inputs",
    "compute_reference",
    "compute_reference_cached",
]
