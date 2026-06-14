"""Keccak-f[1600] sponge reference (CPU).

The reference implements the Keccak-f[1600] permutation specified in
FIPS 202 (24 rounds of theta, rho, pi, chi, iota over a 5x5 array of
64-bit lanes). Two sponge modes are exercised by the Z8 task:

- **SHA3-256**: rate = 136 bytes, capacity = 64 bytes, domain byte =
  0x06, output = 32 bytes. Matches ``hashlib.sha3_256(msg).digest()``.
- **SHAKE128**: rate = 168 bytes, capacity = 32 bytes, domain byte =
  0x1F, output = configurable. Matches
  ``hashlib.shake_128(msg).digest(out_bytes)``.

All test inputs satisfy ``msg_bytes < rate_bytes`` (single absorb
block) and ``msg_bytes``, ``rate_bytes``, ``out_bytes`` are multiples
of 8, so the host packs everything as ``np.uint64`` arrays.

Two permutation flavours are exposed:

- ``keccak_f1600_single(state)``: per-instance reference, mainly for
  cross-validating against ``hashlib`` on a few small inputs.
- ``keccak_f1600_vec(states)``: numpy-vectorised over the batch axis;
  the only reference fast enough for the 2^22-instance test size.

The full sponge pipeline (``hash_batch``) does absorb + pad + permute
+ squeeze, looping the squeeze over multiple permutations if the
requested output exceeds the rate. ``hash_batch_cached`` adds a
sha256-keyed disk cache so the pure-numpy reference runs at most once
per ``(mode, batch, seed)`` triple.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Constants from FIPS 202
# ----------------------------------------------------------------------

KECCAK_RC: np.ndarray = np.array(
    [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
        0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
        0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
        0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
        0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
        0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
        0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ],
    dtype=np.uint64,
)


# Rho offsets: KECCAK_RHO[x][y] is the left-rotate amount for lane (x, y).
KECCAK_RHO: list[list[int]] = [
    [ 0, 36,  3, 41, 18],
    [ 1, 44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14],
]


# Standard mode parameters (rate, capacity, domain).
SHA3_256_RATE_BYTES: int = 136
SHA3_256_DOMAIN: int = 0x06
SHAKE128_RATE_BYTES: int = 168
SHAKE128_DOMAIN: int = 0x1F


# ----------------------------------------------------------------------
# Permutation (vectorised over a leading batch axis)
# ----------------------------------------------------------------------

def _rotl64(x: np.ndarray, k: int) -> np.ndarray:
    """Left-rotate a uint64 numpy array by ``k`` bits (0 <= k < 64)."""
    if k == 0:
        return x
    k64 = np.uint64(k)
    nk = np.uint64(64 - k)
    return ((x << k64) | (x >> nk)).astype(np.uint64)


def keccak_f1600_vec(state: np.ndarray) -> np.ndarray:
    """Apply 24 rounds of Keccak-f[1600] to ``state``.

    ``state`` is a ``(B, 5, 5)`` ``uint64`` array where ``state[b, x, y]``
    holds lane ``(x, y)`` of instance ``b``. The state is updated
    in-place and also returned for convenience.
    """
    if state.dtype != np.uint64:
        raise TypeError(f"state.dtype must be uint64; got {state.dtype}")
    if state.ndim != 3 or state.shape[-2:] != (5, 5):
        raise ValueError(
            f"state shape must be (B, 5, 5); got {state.shape}"
        )

    for rnd in range(24):
        # Theta: column XOR + 1-bit-rotated lateral mix.
        C = np.bitwise_xor.reduce(state, axis=2)            # (B, 5)
        D = np.empty_like(C)
        for x in range(5):
            D[:, x] = C[:, (x - 1) % 5] ^ _rotl64(C[:, (x + 1) % 5], 1)
        state ^= D[:, :, None]

        # Rho + Pi: rotate each lane by r[x][y], then scatter into a new grid.
        new_grid = np.empty_like(state)
        for x in range(5):
            for y in range(5):
                x_new = y
                y_new = (2 * x + 3 * y) % 5
                new_grid[:, x_new, y_new] = _rotl64(
                    state[:, x, y], KECCAK_RHO[x][y]
                )

        # Chi: nonlinear row mix on the post-rho/pi grid.
        for y in range(5):
            row0 = new_grid[:, 0, y]
            row1 = new_grid[:, 1, y]
            row2 = new_grid[:, 2, y]
            row3 = new_grid[:, 3, y]
            row4 = new_grid[:, 4, y]
            state[:, 0, y] = row0 ^ ((~row1) & row2)
            state[:, 1, y] = row1 ^ ((~row2) & row3)
            state[:, 2, y] = row2 ^ ((~row3) & row4)
            state[:, 3, y] = row3 ^ ((~row4) & row0)
            state[:, 4, y] = row4 ^ ((~row0) & row1)

        # Iota.
        state[:, 0, 0] ^= KECCAK_RC[rnd]

    return state


def keccak_f1600_single(state: np.ndarray) -> np.ndarray:
    """Single-instance Keccak-f[1600] for cross-validation. ``state`` is
    a ``(5, 5)`` ``uint64`` array."""
    if state.shape != (5, 5):
        raise ValueError(f"state shape must be (5, 5); got {state.shape}")
    return keccak_f1600_vec(state[None, ...].copy())[0]


# ----------------------------------------------------------------------
# Sponge (absorb + pad + permute + squeeze, batched)
# ----------------------------------------------------------------------

def _lane_xy(k: int) -> tuple[int, int]:
    """Lane index k -> (x, y) where k = x + 5 * y."""
    return k % 5, k // 5


def hash_batch(
    messages_u64: np.ndarray,
    msg_bytes: int,
    rate_bytes: int,
    out_bytes: int,
    domain: int,
    chunk: int = 1 << 12,
) -> np.ndarray:
    """Batched Keccak sponge hash.

    Parameters
    ----------
    messages_u64
        Shape ``(B, msg_bytes // 8)`` uint64. Lane ``i`` holds bytes
        ``8*i .. 8*i + 7`` of instance's message in little-endian.
    msg_bytes, rate_bytes, out_bytes
        Sponge parameters in bytes. All must be multiples of 8.
        ``0 <= msg_bytes < rate_bytes`` (single absorb block).
    domain
        Padding domain separator (0x06 for SHA3, 0x1F for SHAKE).
    chunk
        Process the batch in slices of this many instances to bound
        peak memory; the result is identical to ``chunk = B``.

    Returns
    -------
    np.ndarray
        Shape ``(B, out_bytes // 8)`` uint64.
    """
    if msg_bytes % 8 or rate_bytes % 8 or out_bytes % 8:
        raise ValueError(
            "msg_bytes / rate_bytes / out_bytes must all be multiples of 8; "
            f"got {msg_bytes}, {rate_bytes}, {out_bytes}"
        )
    if not 0 <= msg_bytes < rate_bytes:
        raise ValueError(
            f"need 0 <= msg_bytes < rate_bytes; got "
            f"msg_bytes={msg_bytes}, rate_bytes={rate_bytes}"
        )
    if out_bytes <= 0:
        raise ValueError(f"out_bytes must be positive; got {out_bytes}")

    msg_lanes = msg_bytes // 8
    rate_lanes = rate_bytes // 8
    out_lanes = out_bytes // 8

    if messages_u64.dtype != np.uint64:
        raise TypeError(
            f"messages dtype must be uint64; got {messages_u64.dtype}"
        )
    if messages_u64.ndim != 2 or messages_u64.shape[1] != msg_lanes:
        raise ValueError(
            f"messages shape must be (B, {msg_lanes}); got "
            f"{messages_u64.shape}"
        )

    B_total = int(messages_u64.shape[0])
    out = np.empty((B_total, out_lanes), dtype=np.uint64)

    pad_lane_idx = msg_lanes
    pad_lane_x, pad_lane_y = _lane_xy(pad_lane_idx)
    last_lane_idx = rate_lanes - 1
    last_lane_x, last_lane_y = _lane_xy(last_lane_idx)
    domain_word = np.uint64(domain & 0xFF)
    last_word = np.uint64(0x80) << np.uint64(56)

    for start in range(0, B_total, chunk):
        end = min(start + chunk, B_total)
        b = end - start
        state = np.zeros((b, 5, 5), dtype=np.uint64)

        # Absorb single block, XORing message lanes into state lanes 0..msg_lanes-1.
        for k in range(msg_lanes):
            x, y = _lane_xy(k)
            state[:, x, y] ^= messages_u64[start:end, k]

        # Pad domain at byte position msg_bytes (lane msg_lanes, byte 0).
        state[:, pad_lane_x, pad_lane_y] ^= domain_word
        # Pad 0x80 at byte position rate_bytes - 1 (last byte of last rate lane).
        state[:, last_lane_x, last_lane_y] ^= last_word

        # Permute then squeeze.
        written = 0
        while written < out_lanes:
            keccak_f1600_vec(state)
            remaining = out_lanes - written
            take = min(rate_lanes, remaining)
            for j in range(take):
                x, y = _lane_xy(j)
                out[start:end, written + j] = state[:, x, y]
            written += take

    return out


# ----------------------------------------------------------------------
# Random input
# ----------------------------------------------------------------------

def random_messages(batch: int, msg_bytes: int, seed: int) -> np.ndarray:
    """Random ``(batch, msg_bytes // 8)`` uint64 message buffer."""
    if msg_bytes % 8:
        raise ValueError(f"msg_bytes must be a multiple of 8; got {msg_bytes}")
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, 1 << 64, size=(batch, msg_bytes // 8),
        dtype=np.uint64, endpoint=False,
    )


# ----------------------------------------------------------------------
# Disk cache (mirrors metal_zk.reference.poseidon2)
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(
        os.environ.get("METAL_ZK_CACHE", "~/.cache/metal-zk")
    ).expanduser()
    d = root / "keccak"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    messages_u64: np.ndarray,
    msg_bytes: int,
    rate_bytes: int,
    out_bytes: int,
    domain: int,
) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(np.uint32(msg_bytes).tobytes())
    h.update(np.uint32(rate_bytes).tobytes())
    h.update(np.uint32(out_bytes).tobytes())
    h.update(np.uint32(domain).tobytes())
    h.update(np.uint64(messages_u64.shape[0]).tobytes())
    h.update(np.ascontiguousarray(messages_u64, dtype=np.uint64).tobytes())
    return h.hexdigest()[:16]


def hash_batch_cached(
    messages_u64: np.ndarray,
    msg_bytes: int,
    rate_bytes: int,
    out_bytes: int,
    domain: int,
) -> np.ndarray:
    """`hash_batch` with an in-memory + disk cache keyed by
    ``sha256(mode params, batch, input bytes)``.

    Necessary at the 2^22-instance size where even the vectorised
    reference takes tens of seconds to run; the same input is
    re-evaluated once per timing rep across the evolve loop and
    re-running every time would dominate wall time.
    """
    key = _cache_key(messages_u64, msg_bytes, rate_bytes, out_bytes, domain)
    mem_key = f"k_{key}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]
    B = int(messages_u64.shape[0])
    path = _cache_dir() / (
        f"B{B}_r{rate_bytes}_o{out_bytes}_d{domain:02x}_{key}.u64"
    )
    if path.exists():
        flat = np.fromfile(path, dtype=np.uint64)
        if flat.shape[0] == B * (out_bytes // 8):
            out = flat.reshape(B, out_bytes // 8)
            _MEM_CACHE[mem_key] = out
            return out
    out = hash_batch(messages_u64, msg_bytes, rate_bytes, out_bytes, domain)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_CACHE[mem_key] = out
    return out


__all__ = [
    "KECCAK_RC",
    "KECCAK_RHO",
    "SHA3_256_RATE_BYTES",
    "SHA3_256_DOMAIN",
    "SHAKE128_RATE_BYTES",
    "SHAKE128_DOMAIN",
    "keccak_f1600_single",
    "keccak_f1600_vec",
    "hash_batch",
    "hash_batch_cached",
    "random_messages",
]
