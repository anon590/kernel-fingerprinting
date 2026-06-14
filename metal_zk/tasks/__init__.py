"""Concrete benchmark tasks.

Importing this package registers every task in the global task registry.
Each `try / except ImportError` lets us start with a partial regime set
during development without breaking the CLI.
"""

try:
    from . import goldilocks_ntt   # noqa: F401  (Z2)
except ImportError:
    pass
try:
    from . import poseidon2_hash   # noqa: F401  (Z3)
except ImportError:
    pass
try:
    from . import keccak_f1600_batch   # noqa: F401  (Z8)
except ImportError:
    pass
try:
    from . import merkle_build   # noqa: F401  (Z4)
except ImportError:
    pass
try:
    from . import logup_gkr   # noqa: F401  (Z7)
except ImportError:
    pass
try:
    from . import wots_chain   # noqa: F401  (Z10)
except ImportError:
    pass
try:
    from . import montgomery_msm   # noqa: F401  (Z1)
except ImportError:
    pass
try:
    from . import pippenger_buckets   # noqa: F401  (Z9)
except ImportError:
    pass
try:
    from . import fri_round   # noqa: F401  (Z5)
except ImportError:
    pass
try:
    from . import kyber_ntt   # noqa: F401  (Z6)
except ImportError:
    pass
try:
    from . import binius_clmul   # noqa: F401  (Z11)
except ImportError:
    pass
try:
    from . import multilinear_sumcheck_round   # noqa: F401  (Z13)
except ImportError:
    pass
