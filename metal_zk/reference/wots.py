"""WOTS+ / SPHINCS+ chain reference (CPU).

Each chain takes an ``n_bytes``-byte seed, applies the Keccak-256 inner
hash (FIPS 202: rate=136, capacity=64, domain=0x06) ``w`` times in
sequence, truncating each digest to its first ``n_bytes`` bytes before
feeding it into the next iteration, and emits the resulting tip. The
``L * d`` chains in a WOTS+ public-key derivation are independent, so
the batch axis is embarrassingly parallel; the ``w``-step iteration
along each chain is strictly sequential.

For ``n_bytes < rate_bytes`` (always true at the test sizes -- the
held-out probe uses ``n_bytes=32`` and the in-distribution sizes use
``n_bytes=16``, both well under the 136-byte SHA3 rate) the sponge
collapses to a one-block absorb + one-block squeeze per chain step:

    state                          := 0
    state[lane 0..n_lanes-1]       := previous_chunk
    state[lane n_lanes, byte 0]    ^= 0x06        # SHA3 domain pad
    state[lane 16, byte 7]         ^= 0x80        # FIPS 202 final pad
    state                          := Keccak-f1600(state)
    next_chunk                     := state[lane 0..n_lanes-1]

The held-out twist swaps the chunk size (``n_bytes`` 16 -> 32) and the
chain count, but keeps the same Keccak-256 padding. A candidate that
hardcodes ``n_bytes / 8`` lanes for the absorb / squeeze or bakes the
in-distribution chain count into its dispatch loop silently fails.

The vectorised numpy reference reuses :func:`keccak_f1600_vec` from
``metal_zk.reference.keccak`` and loops the chain step ``w`` times
across the full batch. Disk + memory cache is keyed by
``sha256(n_bytes, w, batch, input bytes)`` so the same input is
re-evaluated at most once across an evolve sweep.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .keccak import (
    SHA3_256_RATE_BYTES, SHA3_256_DOMAIN,
)


# Per-permutation bitop count, mirrored from
# ``metal_zk.tasks.keccak_f1600_batch._BITOPS_PER_PERMUTATION``: theta
# (20 XORs + 5 rotates + 5 D-XORs) + chi (5 NOTs + 25 ANDs + 75 XORs)
# + iota (1 XOR), summed over 24 rounds.
BITOPS_PER_PERMUTATION: int = 24 * (20 + 5 + 5 + 25 + 24 + 75 + 1)


# ----------------------------------------------------------------------
# Chain reference
# ----------------------------------------------------------------------

def chain_batch(
    seeds_u64: np.ndarray,
    n_bytes: int,
    w: int,
    rate_bytes: int = SHA3_256_RATE_BYTES,
    domain: int = SHA3_256_DOMAIN,
) -> np.ndarray:
    """Apply the Keccak-256 chain step ``w`` times across a batch.

    Parameters
    ----------
    seeds_u64
        Shape ``(B, n_bytes // 8)`` uint64.
    n_bytes
        Chunk size in bytes (must satisfy ``n_bytes % 8 == 0`` and
        ``0 < n_bytes < rate_bytes``).
    w
        Number of chain iterations. ``w >= 0``; ``w == 0`` returns the
        seed unchanged.
    rate_bytes
        Sponge rate in bytes (default SHA3-256 = 136). Only checked for
        validity; the actual permutation is delegated to
        ``hashlib.sha3_256``, which is the SHA3-256 mode hard-coded into
        the standard library. The held-out probe shares the same mode
        (it just truncates the digest at a different boundary), so we
        do not need to re-implement non-standard sponge framings here.
    domain
        Padding domain byte (default SHA3 = 0x06). Same caveat as
        ``rate_bytes``.

    Returns
    -------
    np.ndarray
        Shape ``(B, n_bytes // 8)`` uint64 tip array.

    Notes
    -----
    The chain step is identical to ``hashlib.sha3_256(prev).digest()
    [:n_bytes]``: the SHA3-256 sponge framing (rate=136, domain=0x06)
    is what ``hashlib`` implements, and at our test sizes ``n_bytes``
    is always strictly less than both the rate (136) and the digest
    length (32) so the squeeze never needs more than one permutation.
    We therefore reuse the C-accelerated stdlib hash rather than
    re-running the numpy-vectorised Keccak from
    ``metal_zk.reference.keccak`` -- at w=256, n_chains=2^16 the latter
    takes minutes per first-time cache build, while this path takes
    seconds.
    """
    if n_bytes % 8:
        raise ValueError(f"n_bytes must be a multiple of 8; got {n_bytes}")
    if not 0 < n_bytes < rate_bytes:
        raise ValueError(
            f"need 0 < n_bytes < rate_bytes; got n_bytes={n_bytes}, "
            f"rate_bytes={rate_bytes}"
        )
    if rate_bytes != SHA3_256_RATE_BYTES or domain != SHA3_256_DOMAIN:
        raise NotImplementedError(
            "chain_batch currently delegates to hashlib.sha3_256 and "
            f"only supports the standard SHA3-256 sponge framing "
            f"(rate=136, domain=0x06); got rate={rate_bytes}, "
            f"domain=0x{domain:02x}"
        )
    if w < 0:
        raise ValueError(f"w must be non-negative; got {w}")

    n_lanes = n_bytes // 8

    if seeds_u64.dtype != np.uint64:
        raise TypeError(f"seeds dtype must be uint64; got {seeds_u64.dtype}")
    if seeds_u64.ndim != 2 or seeds_u64.shape[1] != n_lanes:
        raise ValueError(
            f"seeds shape must be (B, {n_lanes}); got {seeds_u64.shape}"
        )

    B_total = int(seeds_u64.shape[0])

    if w == 0:
        return seeds_u64.copy()

    # hashlib runs over byte buffers; the numpy view of seeds_u64 is
    # already little-endian uint64-packed, so a single tobytes() gives
    # the per-chain seed bytes back to back.
    seeds_bytes = np.ascontiguousarray(seeds_u64, dtype=np.uint64).tobytes()
    out_bytes = bytearray(B_total * n_bytes)

    sha3_256 = hashlib.sha3_256
    for i in range(B_total):
        cur = seeds_bytes[i * n_bytes:(i + 1) * n_bytes]
        for _ in range(w):
            cur = sha3_256(cur).digest()[:n_bytes]
        out_bytes[i * n_bytes:(i + 1) * n_bytes] = cur

    return np.frombuffer(bytes(out_bytes), dtype=np.uint64).reshape(
        B_total, n_lanes
    )


def random_seeds(batch: int, n_bytes: int, seed: int) -> np.ndarray:
    """Random ``(batch, n_bytes // 8)`` uint64 seed buffer."""
    if n_bytes % 8:
        raise ValueError(f"n_bytes must be a multiple of 8; got {n_bytes}")
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, 1 << 64, size=(batch, n_bytes // 8),
        dtype=np.uint64, endpoint=False,
    )


# ----------------------------------------------------------------------
# Disk cache (mirrors metal_zk.reference.keccak)
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(
        os.environ.get("METAL_ZK_CACHE", "~/.cache/metal-zk")
    ).expanduser()
    d = root / "wots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    seeds_u64: np.ndarray,
    n_bytes: int,
    w: int,
    rate_bytes: int,
    domain: int,
) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint32(n_bytes).tobytes())
    h.update(np.uint32(w).tobytes())
    h.update(np.uint32(rate_bytes).tobytes())
    h.update(np.uint32(domain).tobytes())
    h.update(np.uint64(seeds_u64.shape[0]).tobytes())
    h.update(np.ascontiguousarray(seeds_u64, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def chain_batch_cached(
    seeds_u64: np.ndarray,
    n_bytes: int,
    w: int,
    rate_bytes: int = SHA3_256_RATE_BYTES,
    domain: int = SHA3_256_DOMAIN,
) -> np.ndarray:
    """`chain_batch` with an in-memory + disk cache keyed by
    ``sha256(params, batch, input bytes)``.

    Necessary because the largest in-distribution size (w=256,
    n_chains=2^16) takes several hundred milliseconds in the
    vectorised reference, and the same input is re-evaluated once per
    timing rep across the evolve loop.
    """
    key = _cache_key(seeds_u64, n_bytes, w, rate_bytes, domain)
    mem_key = f"w_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    B = int(seeds_u64.shape[0])
    path = _cache_dir() / (
        f"B{B}_n{n_bytes}_w{w}_r{rate_bytes}_d{domain:02x}_{key}.u64"
    )
    if path.exists():
        flat = np.fromfile(path, dtype=np.uint64)
        n_lanes = n_bytes // 8
        if flat.shape[0] == B * n_lanes:
            out = flat.reshape(B, n_lanes)
            _MEM_CACHE[mem_key] = out
            return out
    out = chain_batch(seeds_u64, n_bytes, w, rate_bytes, domain)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = out
    return out


__all__ = [
    "BITOPS_PER_PERMUTATION",
    "chain_batch",
    "chain_batch_cached",
    "random_seeds",
]
