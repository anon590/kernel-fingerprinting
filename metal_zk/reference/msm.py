"""Multi-scalar multiplication reference (Z1).

Two short-Weierstrass curves over their respective prime base fields:

  - BLS12-381 G1   (in-distribution)   y^2 = x^3 + 4 ; q ~ 381 bits
  - BN254 G1       (held-out)          y^2 = x^3 + 3 ; q ~ 254 bits

Both are stored uniformly with ``n_limbs = 6`` (6 * 64 = 384 bits >= q
for either curve), so the GPU kernel can run unmodified across the
two curves -- only the runtime modulus ``q``, Montgomery constant
``q' = -q^-1 mod 2^64``, and curve order ``r`` differ. PLAN.md's
"4x64-bit limbs" phrasing is precise for BN254 only; BLS12-381 G1's
base field is 381-bit and *requires* 6 limbs. We pick the uniform
6-limb design so a single kernel handles both: BN254 is supplied
with its top two limbs of ``q`` (and all coordinates) zero.

Montgomery form: every field element ``x`` is stored as ``x * R mod q``
with ``R = 2^384``. So R^2 mod q and -q^-1 mod 2^64 are curve-dependent
constants supplied by the host.

Coordinate convention: Jacobian ``(X, Y, Z)`` over the base field;
the affine point is ``(X / Z^2, Y / Z^3)``. ``Z = 0`` represents the
point at infinity.

Test-input generation (algebraic shortcut)
-----------------------------------------
At N up to 2^18 we cannot afford N independent CPU scalar-mults --
each is ~10 ms in Python bigint. Instead we use a determined
**linear** structure for the base points:

  k_i = (a + b * i) mod r,  P_i = k_i * G

with seed-derived a, b. The recurrence ``P_{i+1} = P_i + B`` (where
``B = b * G``) costs **one EC addition per point** instead of one
scalar-mul, so we can precompute the entire ``N`` base points in O(N)
EC adds (~half a second per 2^14 batch, cached on disk).

The MSM result is then

  R = sum_i s_i * P_i
    = sum_i s_i * (a + b * i) * G
    = (a * sum_i s_i + b * sum_i i * s_i) * G       (mod r in the scalar)
    = alpha * G

which costs **one** EC scalar-mul on the CPU (~10 ms) plus O(N)
integer arithmetic. The shortcut never leaks to the GPU -- the
kernel just sees N points and N scalars and must compute the full MSM
without exploiting the linear structure.

Held-out twist: the curve switches between BLS12-381 G1 and BN254 G1.
A candidate that hardcodes BLS12-381's modulus (or its Montgomery
constants) silently produces wrong output on the BN254 size; a
candidate that hardcodes "first 4 limbs only" -- skipping the top 2
zero limbs that BLS12-381 needs -- silently fails BLS12-381.

Bit-exactness
-------------
Jacobian representations are non-unique: (X, Y, Z) and
(X*u^2, Y*u^3, Z*u) represent the same affine point for any u. To
bit-exact compare the GPU output against the algebraic reference we
normalize *both sides* to affine Montgomery form (X_aff_mont,
Y_aff_mont) before the equality test. Point at infinity is detected
by Z == 0 on either side; agreement on Z == 0 is a match.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


N_LIMBS: int = 6                    # uniform across curves
LIMB_BITS: int = 64
R_SHIFT: int = N_LIMBS * LIMB_BITS   # 384; Montgomery factor R = 2^384
R: int = 1 << R_SHIFT
LIMB_MASK: int = (1 << LIMB_BITS) - 1


# ----------------------------------------------------------------------
# Curve parameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CurveParams:
    """Short-Weierstrass curve y^2 = x^3 + b over F_q with G of order r."""
    name: str
    q: int                      # base field modulus
    r: int                      # scalar field modulus (curve subgroup order)
    b: int                      # curve constant
    g_x: int                    # generator, affine x
    g_y: int                    # generator, affine y


# BLS12-381 G1 (Plonky2 / arkworks-bls12_381 standard params).
# Parameters from the IETF pairing-friendly-curves draft / arkworks.
BLS12_381_G1: CurveParams = CurveParams(
    name="bls12_381_g1",
    q=4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787,
    r=52435875175126190479447740508185965837690552500527637822603658699938581184513,
    b=4,
    g_x=3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507,
    g_y=1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569,
)

BN254_G1: CurveParams = CurveParams(
    name="bn254_g1",
    q=21888242871839275222246405745257275088696311157297823662689037894645226208583,
    r=21888242871839275222246405745257275088548364400416034343698204186575808495617,
    b=3,
    g_x=1,
    g_y=2,
)


def get_curve(name: str) -> CurveParams:
    name = name.lower()
    if name in ("bls12_381_g1", "bls12_381", "bls"):
        return BLS12_381_G1
    if name in ("bn254_g1", "bn254", "bn"):
        return BN254_G1
    raise KeyError(f"unknown curve: {name!r}")


# ----------------------------------------------------------------------
# Bigint <-> limb packing
# ----------------------------------------------------------------------

def int_to_limbs(x: int, n_limbs: int = N_LIMBS) -> np.ndarray:
    """Little-endian uint64 limbs of a non-negative ``x``."""
    if x < 0:
        raise ValueError("int_to_limbs requires non-negative input")
    out = np.zeros(n_limbs, dtype=np.uint64)
    for i in range(n_limbs):
        out[i] = np.uint64(x & LIMB_MASK)
        x >>= LIMB_BITS
    if x != 0:
        raise ValueError(
            f"value does not fit in {n_limbs} 64-bit limbs"
        )
    return out


def limbs_to_int(limbs: np.ndarray) -> int:
    """Inverse of ``int_to_limbs``. ``limbs`` must be ``uint64``."""
    x = 0
    for i in range(len(limbs) - 1, -1, -1):
        x = (x << LIMB_BITS) | int(limbs[i])
    return x


# ----------------------------------------------------------------------
# Montgomery arithmetic (CPU, bigint backend)
# ----------------------------------------------------------------------

class Montgomery:
    """Montgomery arithmetic in F_q with R = 2^(64 * n_limbs).

    The CPU implementation is a thin wrapper around Python ``int``: it
    never materialises the CIOS step structure (that's the GPU's job),
    just exposes the mathematical operation so the reference can be
    written naturally.
    """

    def __init__(self, q: int, n_limbs: int = N_LIMBS):
        if q & 1 == 0:
            raise ValueError("Montgomery requires odd modulus")
        self.q = q
        self.n_limbs = n_limbs
        self.R = 1 << (n_limbs * LIMB_BITS)
        # q' = -q^-1 mod 2^64 (the CIOS scalar). Python's pow with -1
        # accepts negative bases since 3.8.
        q_inv_64 = pow(q, -1, 1 << 64)
        self.q_inv_neg_64 = ((1 << 64) - q_inv_64) & LIMB_MASK
        self.R_mod_q = self.R % q
        self.R2_mod_q = (self.R * self.R) % q
        self.R_inv = pow(self.R, -1, q)

    def to_mont(self, x: int) -> int:
        return (x % self.q) * self.R % self.q

    def from_mont(self, xm: int) -> int:
        return (xm * self.R_inv) % self.q

    def mul(self, a_m: int, b_m: int) -> int:
        """Montgomery multiplication: (a*b*R^-1) mod q."""
        return (a_m * b_m * self.R_inv) % self.q

    def add(self, a_m: int, b_m: int) -> int:
        return (a_m + b_m) % self.q

    def sub(self, a_m: int, b_m: int) -> int:
        return (a_m - b_m) % self.q

    def neg(self, a_m: int) -> int:
        return (-a_m) % self.q

    def inv(self, a_m: int) -> int:
        """Mod-inverse in Montgomery form."""
        if a_m == 0:
            raise ZeroDivisionError("inverse of zero")
        std = self.from_mont(a_m)
        std_inv = pow(std, -1, self.q)
        return self.to_mont(std_inv)


# ----------------------------------------------------------------------
# Jacobian point arithmetic in Montgomery form (CPU reference)
# ----------------------------------------------------------------------

@dataclass
class JacMont:
    """Jacobian point with all three coords in Montgomery form."""
    X: int
    Y: int
    Z: int

    def is_infinity(self) -> bool:
        return self.Z == 0


def jac_infinity() -> JacMont:
    return JacMont(X=0, Y=0, Z=0)


def jac_from_affine(x_aff: int, y_aff: int, m: Montgomery) -> JacMont:
    """Affine standard form -> Jacobian Montgomery (Z = R mod q)."""
    return JacMont(
        X=m.to_mont(x_aff),
        Y=m.to_mont(y_aff),
        Z=m.R_mod_q,           # 1 in Montgomery form
    )


def jac_to_affine_mont(p: JacMont, m: Montgomery) -> tuple[int, int] | None:
    """Returns (x_aff_mont, y_aff_mont) or None if infinity."""
    if p.Z == 0:
        return None
    z_inv = m.inv(p.Z)
    z_inv_sq = m.mul(z_inv, z_inv)
    z_inv_cu = m.mul(z_inv_sq, z_inv)
    x_aff_m = m.mul(p.X, z_inv_sq)
    y_aff_m = m.mul(p.Y, z_inv_cu)
    return (x_aff_m, y_aff_m)


def jac_double(p: JacMont, m: Montgomery) -> JacMont:
    """Double a Jacobian point on a curve with a = 0 (short Weierstrass
    ``y^2 = x^3 + b``)."""
    if p.Z == 0 or p.Y == 0:
        return jac_infinity()
    X1, Y1, Z1 = p.X, p.Y, p.Z
    A = m.mul(X1, X1)                # X1^2
    B = m.mul(Y1, Y1)                # Y1^2
    C = m.mul(B, B)                  # B^2
    xb = m.add(X1, B)
    D = m.sub(m.sub(m.mul(xb, xb), A), C)
    D = m.add(D, D)                  # D = 2*((X1+B)^2 - A - C)
    E = m.add(m.add(A, A), A)        # 3A
    F = m.mul(E, E)                  # E^2
    X3 = m.sub(F, m.add(D, D))       # F - 2D
    Y3 = m.sub(m.mul(E, m.sub(D, X3)),
               m.add(m.add(m.add(C, C), m.add(C, C)),
                     m.add(m.add(C, C), m.add(C, C))))   # 8C via three doublings
    Z3 = m.add(m.mul(Y1, Z1), m.mul(Y1, Z1))             # 2*Y1*Z1
    return JacMont(X=X3, Y=Y3, Z=Z3)


def jac_add(p: JacMont, q: JacMont, m: Montgomery) -> JacMont:
    """Jacobian addition. Handles infinity inputs and ``P == Q`` /
    ``P == -Q`` exceptional cases.

    Uses the standard formulas (Bernstein-Lange "add-2007-bl"):
      Z1Z1 = Z1^2 ; Z2Z2 = Z2^2
      U1 = X1*Z2Z2 ; U2 = X2*Z1Z1
      S1 = Y1*Z2*Z2Z2 ; S2 = Y2*Z1*Z1Z1
      if U1 == U2:
          if S1 == S2: return double(P)
          else:        return infinity
      H = U2 - U1 ; r = S2 - S1
      HH = H^2 ; HHH = H*HH ; V = U1*HH
      X3 = r^2 - HHH - 2V
      Y3 = r*(V - X3) - S1*HHH
      Z3 = Z1*Z2*H
    """
    if p.Z == 0:
        return JacMont(X=q.X, Y=q.Y, Z=q.Z)
    if q.Z == 0:
        return JacMont(X=p.X, Y=p.Y, Z=p.Z)
    X1, Y1, Z1 = p.X, p.Y, p.Z
    X2, Y2, Z2 = q.X, q.Y, q.Z
    Z1Z1 = m.mul(Z1, Z1)
    Z2Z2 = m.mul(Z2, Z2)
    U1 = m.mul(X1, Z2Z2)
    U2 = m.mul(X2, Z1Z1)
    S1 = m.mul(m.mul(Y1, Z2), Z2Z2)
    S2 = m.mul(m.mul(Y2, Z1), Z1Z1)
    if U1 == U2:
        if S1 == S2:
            return jac_double(p, m)
        return jac_infinity()
    H = m.sub(U2, U1)
    R_ = m.sub(S2, S1)
    HH = m.mul(H, H)
    HHH = m.mul(H, HH)
    V = m.mul(U1, HH)
    X3 = m.sub(m.sub(m.mul(R_, R_), HHH), m.add(V, V))
    Y3 = m.sub(m.mul(R_, m.sub(V, X3)), m.mul(S1, HHH))
    Z3 = m.mul(m.mul(Z1, Z2), H)
    return JacMont(X=X3, Y=Y3, Z=Z3)


def jac_scalar_mul(p: JacMont, k: int, m: Montgomery) -> JacMont:
    """Compute ``k * p`` via MSB-to-LSB double-and-add."""
    if k == 0 or p.Z == 0:
        return jac_infinity()
    if k < 0:
        raise ValueError("negative scalar not supported")
    acc = jac_infinity()
    for i in range(k.bit_length() - 1, -1, -1):
        acc = jac_double(acc, m)
        if (k >> i) & 1:
            acc = jac_add(acc, p, m)
    return acc


# ----------------------------------------------------------------------
# Test-input + reference generation
# ----------------------------------------------------------------------

@dataclass
class MsmInputs:
    """One ready-to-dispatch MSM test case.

    - ``scalars_u64`` shape ``(N, 4)`` uint64 (little-endian limbs of s_i)
    - ``points_in_u64`` shape ``(N, 18)`` uint64 (X, Y, Z each 6 limbs,
      little-endian, in Montgomery form)
    - ``expected_jac_mont_u64`` shape ``(18,)`` uint64: the algebraic
      reference output in Jacobian Montgomery
    - ``expected_aff_mont_u64`` shape ``(12,)`` uint64: the same point in
      affine Montgomery form (X_aff_mont, Y_aff_mont); ``None`` if
      infinity. Used for the bit-exact comparison after the GPU output is
      itself normalized to affine.
    """
    curve: CurveParams
    scalars_u64: np.ndarray
    points_in_u64: np.ndarray
    expected_jac_mont: JacMont
    expected_aff_mont_u64: np.ndarray | None        # (12,) or None
    a: int
    b: int
    seed: int


def _seed_derived_ab(curve: CurveParams, seed: int) -> tuple[int, int]:
    """Two deterministic non-zero seed-derived scalars (a, b) modulo r.

    Both are guaranteed in [1, r). Using SHA-256 of a domain-tagged
    seed keeps the values stable across Python versions."""
    base = f"metal-zk:msm:{curve.name}:seed:{seed}".encode()
    h1 = int.from_bytes(hashlib.sha256(base + b":a").digest(), "big")
    h2 = int.from_bytes(hashlib.sha256(base + b":b").digest(), "big")
    a = (h1 % (curve.r - 1)) + 1
    b = (h2 % (curve.r - 1)) + 1
    return a, b


def _pack_jac_mont(p: JacMont, n_limbs: int = N_LIMBS) -> np.ndarray:
    """(X, Y, Z) Montgomery -> 18-uint64 little-endian limb array."""
    out = np.empty(3 * n_limbs, dtype=np.uint64)
    out[0:n_limbs] = int_to_limbs(p.X, n_limbs)
    out[n_limbs:2 * n_limbs] = int_to_limbs(p.Y, n_limbs)
    out[2 * n_limbs:3 * n_limbs] = int_to_limbs(p.Z, n_limbs)
    return out


def _pack_aff_mont(x_aff_m: int, y_aff_m: int, n_limbs: int = N_LIMBS) -> np.ndarray:
    out = np.empty(2 * n_limbs, dtype=np.uint64)
    out[0:n_limbs] = int_to_limbs(x_aff_m, n_limbs)
    out[n_limbs:2 * n_limbs] = int_to_limbs(y_aff_m, n_limbs)
    return out


def _random_scalars(curve: CurveParams, n: int, seed: int) -> np.ndarray:
    """N scalars in ``[0, r)`` packed as (N, 4) little-endian uint64.

    Uses numpy's default RNG; rejection-samples to discard the small
    bias from a uniform draw out of ``[0, 2^256)``."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 4), dtype=np.uint64)
    filled = 0
    r = curve.r
    while filled < n:
        chunk = max(1, (n - filled) * 2)
        raw = rng.integers(
            0, 1 << 64, size=(chunk, 4), dtype=np.uint64, endpoint=False,
        )
        # Reassemble into Python ints to test ``< r``; accept those that
        # pass. This is the slow path but rejection rate is at most
        # (2^256 - r) / 2^256, which for both curves is < 1/4 -- we lose
        # at most a quarter of the draws.
        for row in raw:
            if filled >= n:
                break
            v = (int(row[3]) << 192) | (int(row[2]) << 128) | (int(row[1]) << 64) | int(row[0])
            if v < r:
                out[filled] = row
                filled += 1
    return out


def _precompute_points_uncached(
    curve: CurveParams,
    n: int,
    a: int,
    b: int,
    m: Montgomery,
) -> np.ndarray:
    """Compute ``P_i = (a + b * i) * G`` for ``i in [0, n)``, Jacobian
    Montgomery, packed as (N, 18) uint64.

    Strategy: P_0 = a*G, B = b*G, then P_{i+1} = P_i + B. One EC scalar
    mul + N EC adds total.
    """
    g_mont = jac_from_affine(curve.g_x, curve.g_y, m)
    a_mod = a % curve.r
    b_mod = b % curve.r
    A_pt = jac_scalar_mul(g_mont, a_mod, m)
    B_pt = jac_scalar_mul(g_mont, b_mod, m)

    out = np.empty((n, 3 * N_LIMBS), dtype=np.uint64)
    cur = A_pt
    for i in range(n):
        out[i] = _pack_jac_mont(cur)
        cur = jac_add(cur, B_pt, m)
    return out


# ----------------------------------------------------------------------
# Disk + memory cache for the point buffer
# ----------------------------------------------------------------------

_CACHE_VERSION = "v1"
_MEM_POINT_CACHE: dict[str, np.ndarray] = {}


def _cache_dir() -> Path:
    root = Path(os.environ.get(
        "METAL_ZK_CACHE", "~/.cache/metal-zk",
    )).expanduser()
    d = root / "msm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(curve_name: str, n: int, a: int, b: int) -> str:
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(curve_name.encode())
    h.update(np.uint64(n).tobytes())
    h.update(a.to_bytes(64, "little"))
    h.update(b.to_bytes(64, "little"))
    return h.hexdigest()[:16]


def precompute_points_cached(
    curve: CurveParams,
    n: int,
    a: int,
    b: int,
    m: Montgomery,
) -> np.ndarray:
    """``_precompute_points_uncached`` with an in-memory + disk cache.

    First-time build at N=2^18 takes a few minutes (one EC add per
    point in pure Python); cache hits are ~milliseconds.
    """
    key = _cache_key(curve.name, n, a, b)
    mem_key = f"pts_{key}"
    if mem_key in _MEM_POINT_CACHE:
        return _MEM_POINT_CACHE[mem_key]
    path = _cache_dir() / f"{curve.name}_N{n}_{key}.u64"
    if path.exists():
        flat = np.fromfile(path, dtype=np.uint64)
        if flat.shape[0] == n * 3 * N_LIMBS:
            out = flat.reshape(n, 3 * N_LIMBS)
            _MEM_POINT_CACHE[mem_key] = out
            return out
    out = _precompute_points_uncached(curve, n, a, b, m)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)
    _MEM_POINT_CACHE[mem_key] = out
    return out


def _scalars_int_view(scalars_u64: np.ndarray) -> list[int]:
    """Convert (N, 4) uint64 little-endian to a list of Python ints.

    Used in the reference shortcut; ``int()`` calls dominate the cost,
    so prefer ``int.from_bytes`` over four shift-or assemblies.
    """
    raw = np.ascontiguousarray(scalars_u64, dtype=np.uint64).tobytes()
    return [int.from_bytes(raw[i:i + 32], "little")
            for i in range(0, len(raw), 32)]


def gen_inputs(
    curve: CurveParams,
    n: int,
    seed: int,
) -> MsmInputs:
    """Build a full ready-to-dispatch MSM test case.

    Deterministic on (curve, n, seed). The expensive part (per-pair
    point precompute) is cached on disk.
    """
    m = Montgomery(curve.q, N_LIMBS)
    a, b = _seed_derived_ab(curve, seed)
    scalars_u64 = _random_scalars(curve, n, seed)
    points_in_u64 = precompute_points_cached(curve, n, a, b, m)

    # Algebraic shortcut: R = (a*S0 + b*S1) * G where S0 = sum s_i,
    # S1 = sum i*s_i, both mod r.
    s_list = _scalars_int_view(scalars_u64)
    S0 = 0
    S1 = 0
    r = curve.r
    for i, si in enumerate(s_list):
        S0 = (S0 + si) % r
        S1 = (S1 + i * si) % r
    alpha = (a * S0 + b * S1) % r
    g_mont = jac_from_affine(curve.g_x, curve.g_y, m)
    expected = jac_scalar_mul(g_mont, alpha, m) if alpha != 0 else jac_infinity()

    aff = jac_to_affine_mont(expected, m)
    expected_aff_u64: np.ndarray | None
    if aff is None:
        expected_aff_u64 = None
    else:
        expected_aff_u64 = _pack_aff_mont(aff[0], aff[1])

    return MsmInputs(
        curve=curve,
        scalars_u64=scalars_u64,
        points_in_u64=points_in_u64,
        expected_jac_mont=expected,
        expected_aff_mont_u64=expected_aff_u64,
        a=a,
        b=b,
        seed=seed,
    )


# ----------------------------------------------------------------------
# Counting model (for the roofline)
# ----------------------------------------------------------------------

MODMULS_PER_DOUBLE: int = 10        # 3 squarings + 4 muls + spread additions
MODMULS_PER_ADD: int = 16           # 4 squarings + 11 muls + spread additions
SCALAR_BITS_SCANNED: int = 256      # we scan a fixed 256-bit window per pair


def modmuls_per_pair(scalar_bits: int = SCALAR_BITS_SCANNED) -> int:
    """Naive double-and-add: ``scalar_bits`` doublings + half-density
    adds. Counts only base-field multiplications (the Montgomery cost
    inside the kernel)."""
    n_doubles = scalar_bits
    n_adds = scalar_bits // 2
    return n_doubles * MODMULS_PER_DOUBLE + n_adds * MODMULS_PER_ADD


def modmuls_per_reduce_pair() -> int:
    """One Jacobian add at each tree-reduce step."""
    return MODMULS_PER_ADD


__all__ = [
    "N_LIMBS",
    "LIMB_BITS",
    "R_SHIFT",
    "R",
    "CurveParams",
    "BLS12_381_G1",
    "BN254_G1",
    "get_curve",
    "Montgomery",
    "JacMont",
    "jac_infinity",
    "jac_from_affine",
    "jac_to_affine_mont",
    "jac_double",
    "jac_add",
    "jac_scalar_mul",
    "MsmInputs",
    "gen_inputs",
    "int_to_limbs",
    "limbs_to_int",
    "precompute_points_cached",
    "MODMULS_PER_DOUBLE",
    "MODMULS_PER_ADD",
    "SCALAR_BITS_SCANNED",
    "modmuls_per_pair",
    "modmuls_per_reduce_pair",
]
