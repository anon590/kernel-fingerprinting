"""Multilinear sumcheck-round reference (CPU).

One round of a degree-``d`` sumcheck on a product polynomial

    g(x) = f_0(x) * f_1(x) * ... * f_{d-1}(x)

where each ``f_i: {0,1}^k -> F_p`` is multilinear, stored as a flat
length-``2^k`` table of evaluations on the Boolean hypercube. The
round folds the *first* variable: the prover sends a univariate

    h(X) = sum_{x' in {0,1}^{k-1}} prod_i f_i(X, x')

of degree at most ``d`` (product of ``d`` polynomials that are affine
in ``X``), represented by its evaluations at the ``d+1`` interpolation
points ``X in {0, 1, ..., d}``. The verifier picks a random challenge
``r`` and both sides fold each factor::

    f_i_new[j] = f_i(r, j) = f_i(0, j) + r * (f_i(1, j) - f_i(0, j))

producing tables of length ``2^{k-1}`` for the next round.

This module is the bit-exact CPU oracle for the Metal-ZK Z13 task. It

  1. generates a deterministic ``(f_tables, r)`` pair for a given
     ``(k_log, d, prime_kind, seed)``. The challenge ``r`` is
     hash-derived in ``[2, p)``; excluding 0 / 1 keeps the linear
     extension non-degenerate (``r=0`` collapses the fold to the low
     half, ``r=1`` to the high half -- either silently masks a
     fold-direction bug).
  2. computes the pre-round claim ``S = sum_x prod_i f_i(x)`` in pure
     Python ``int`` arithmetic so the reference is unambiguously bit
     exact across primes (Goldilocks ``p = 2^64 - 2^32 + 1`` or
     BabyBear ``p = 2^31 - 2^27 + 1``).
  3. returns ``h_evals[0..d]``, ``f_out`` (shape ``(d, 2^(k-1))``),
     ``h(r)`` via Lagrange interpolation, and the post-round claim
     ``post_claim = sum_y prod_i f_i_new[y]``. The identities

         h_evals[0] + h_evals[1] == claim         (sumcheck identity)
         h_at_r                  == post_claim    (round closure)

     are *sumcheck consistency invariants*; the task surfaces both so a
     candidate whose ``h_evals`` matches a same-buggy reference still
     gets caught by the cross-check against the folded table.

Results are disk-cached under ``~/.cache/metal-zk/sumcheck/`` keyed by
``sha256(f_tables | r | prime_kind | d)``. The largest in-distribution
size (k_log = 18, d = 2) takes a few seconds the first time and is
served from disk thereafter.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .logup import P_BB, P_GOLD, PRIME_NAMES, PRIMES, prime_of  # noqa: F401


# ----------------------------------------------------------------------
# Input generation
# ----------------------------------------------------------------------

def _random_field_elements(n: int, prime: int, seed: int) -> np.ndarray:
    """``n`` canonical field elements in ``[0, prime)`` as ``uint64``.

    Same rejection sampler as :mod:`metal_zk.reference.logup`; we keep
    a private copy rather than re-export to avoid coupling the two
    references at the API boundary.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.uint64)
    filled = 0
    if prime <= (1 << 32):
        while filled < n:
            chunk = max(1, (n - filled) * 3)
            raw = rng.integers(0, 1 << 32, size=chunk,
                               dtype=np.uint32, endpoint=False)
            accepted = raw[raw < np.uint32(prime)]
            take = min(len(accepted), n - filled)
            out[filled:filled + take] = accepted[:take].astype(np.uint64)
            filled += take
    else:
        while filled < n:
            chunk = max(1, (n - filled) * 2)
            raw = rng.integers(0, 1 << 64, size=chunk,
                               dtype=np.uint64, endpoint=False)
            accepted = raw[raw < np.uint64(prime)]
            take = min(len(accepted), n - filled)
            out[filled:filled + take] = accepted[:take]
            filled += take
    return out


def _challenge_for(k_log: int, d: int, prime_kind: int, seed: int) -> int:
    """Hash-derived round challenge in ``[2, p)``."""
    prime = prime_of(prime_kind)
    h = hashlib.sha256()
    h.update(b"metal-zk:sumcheck:r")
    h.update(np.uint32(k_log).tobytes())
    h.update(np.uint32(d).tobytes())
    h.update(np.uint32(prime_kind).tobytes())
    h.update(np.uint64(seed & ((1 << 64) - 1)).tobytes())
    v = int.from_bytes(h.digest()[:8], "big") % prime
    if v < 2:
        v = (v + 2) % prime
        if v < 2:                        # only possible for tiny test primes
            v = 2
    return int(v)


def generate_inputs(
    k_log: int, d: int, prime_kind: int, seed: int,
) -> tuple[np.ndarray, int]:
    """Generate ``(f_tables, r)`` for one sumcheck round.

    - ``f_tables``: ``uint64[d, 2^k_log]``, canonical evals of the
      ``d`` multilinear factors on the Boolean hypercube ``{0,1}^k``.
    - ``r``: Python ``int`` in ``[2, p)`` -- the round challenge.
    """
    if d < 1:
        raise ValueError(f"d={d} must be >= 1")
    if k_log < 1:
        raise ValueError(f"k_log={k_log} must be >= 1")
    n = 1 << k_log
    prime = prime_of(prime_kind)
    f = np.empty((d, n), dtype=np.uint64)
    for i in range(d):
        sub_seed = (seed + i * 0x9E37_79B9_7F4A_7C15) & ((1 << 64) - 1)
        f[i] = _random_field_elements(n, prime, seed=sub_seed)
    r = _challenge_for(k_log, d, prime_kind, seed)
    return f, int(r)


# ----------------------------------------------------------------------
# Reference computation
# ----------------------------------------------------------------------

def _lagrange_eval_at(values: list[int], x: int, prime: int) -> int:
    """Evaluate at ``x`` the unique poly of degree ``< len(values)``
    that takes ``values[t]`` at ``t = 0, 1, ..., len(values) - 1``.
    Used only for the ``h(r) == post_claim`` consistency check.
    """
    deg_plus_1 = len(values)
    total = 0
    for t in range(deg_plus_1):
        num = 1
        den = 1
        for s in range(deg_plus_1):
            if s == t:
                continue
            num = (num * ((x - s) % prime)) % prime
            den = (den * ((t - s) % prime)) % prime
        total = (total + values[t] * num % prime * pow(den, prime - 2, prime)) % prime
    return int(total)


def compute_reference(
    f_tables: np.ndarray, r: int, prime_kind: int,
) -> dict:
    """Bit-exact reference for one degree-``d`` sumcheck round.

    Returns a dict with keys:
      ``claim``     : pre-round ``S = sum_x prod_i f_i(x)``.
      ``h_evals``   : ``uint64[d+1]``, evaluations ``h(0)..h(d)``.
      ``f_out``     : ``uint64[d, 2^(k-1)]``, folded factor tables.
      ``post_claim``: ``sum_y prod_i f_i_new[y]`` (= ``h(r)`` if all is
                     well; surfaced separately so the host can cross
                     check both halves).
      ``h_at_r``    : ``h(r)`` via Lagrange interpolation; matches
                     ``post_claim`` iff the round is internally
                     consistent.
    """
    prime = prime_of(prime_kind)
    d, n = f_tables.shape
    if n & (n - 1) != 0:
        raise ValueError(f"per-factor table length {n} must be a power of two")
    half = n >> 1

    F = [[int(v) for v in f_tables[i].tolist()] for i in range(d)]

    # Pre-round claim S = sum_x prod_i f_i(x).
    claim = 0
    for j in range(n):
        prod = 1
        for i in range(d):
            prod = (prod * F[i][j]) % prime
        claim = (claim + prod) % prime

    # h(t) = sum_{j in [0, half)} prod_i f_i(t, j) for t in {0,..,d}.
    h_evals = [0] * (d + 1)
    for j in range(half):
        deltas = [(F[i][j + half] - F[i][j]) % prime for i in range(d)]
        for t in range(d + 1):
            prod = 1
            for i in range(d):
                if t == 0:
                    ft = F[i][j]
                elif t == 1:
                    ft = F[i][j + half]
                else:
                    ft = (F[i][j] + t * deltas[i]) % prime
                prod = (prod * ft) % prime
            h_evals[t] = (h_evals[t] + prod) % prime

    # f_i_new[j] = f_i^(0)[j] + r * (f_i^(1)[j] - f_i^(0)[j])  (mod p).
    f_out = np.zeros((d, half), dtype=np.uint64)
    post_claim = 0
    for j in range(half):
        prod = 1
        for i in range(d):
            f0 = F[i][j]
            delta = (F[i][j + half] - f0) % prime
            fn = (f0 + r * delta) % prime
            f_out[i, j] = np.uint64(fn)
            prod = (prod * fn) % prime
        post_claim = (post_claim + prod) % prime

    h_at_r = _lagrange_eval_at(h_evals, int(r) % prime, prime)

    return {
        "claim": int(claim),
        "h_evals": np.array([int(v) for v in h_evals], dtype=np.uint64),
        "f_out": f_out,
        "post_claim": int(post_claim),
        "h_at_r": int(h_at_r),
    }


# ----------------------------------------------------------------------
# Disk cache
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"


def _cache_dir() -> Path:
    root = Path(os.environ.get("METAL_ZK_CACHE",
                               "~/.cache/metal-zk")).expanduser()
    d = root / "sumcheck"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(f_tables: np.ndarray, r: int, prime_kind: int, d: int) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint32(prime_kind).tobytes())
    h.update(np.uint32(d).tobytes())
    h.update(np.uint64(int(r)).tobytes())
    h.update(np.ascontiguousarray(f_tables, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def compute_reference_cached(
    f_tables: np.ndarray, r: int, prime_kind: int,
) -> dict:
    """``compute_reference`` with disk-backed caching."""
    d, n = f_tables.shape
    key = _cache_key(f_tables, r, prime_kind, d)
    path = _cache_dir() / f"d{d}_N{n}_pk{prime_kind}_{key}.npz"
    if path.exists():
        try:
            z = np.load(path)
            return {
                "claim": int(z["claim"].item()),
                "h_evals": z["h_evals"].astype(np.uint64),
                "f_out": z["f_out"].astype(np.uint64),
                "post_claim": int(z["post_claim"].item()),
                "h_at_r": int(z["h_at_r"].item()),
            }
        except Exception:
            pass
    ref = compute_reference(f_tables, r, prime_kind)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez(
        tmp,
        claim=np.array(ref["claim"], dtype=np.uint64),
        h_evals=ref["h_evals"].astype(np.uint64),
        f_out=ref["f_out"].astype(np.uint64),
        post_claim=np.array(ref["post_claim"], dtype=np.uint64),
        h_at_r=np.array(ref["h_at_r"], dtype=np.uint64),
    )
    os.replace(tmp, path)
    return ref


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
