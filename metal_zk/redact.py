"""Held-out disclosure redaction for the controlled grade-C experiment.

Three Metal-ZK task specifications inadvertently disclose the held-out
configuration to the model (keccak_f1600_batch names SHAKE128; kyber_ntt's
seed comment names the Dilithium modulus; wots_chain names n_bytes=32).
Section 3 (grade C) shows the models then *enumerate* the disclosed
configuration and pass the gate by transcription rather than
generalization.

That evidence is observational: the disclosure was authored, not assigned.
This module supports the controlled version --- re-running the same three
tasks with the held-out identity *redacted* from everything the model sees,
to causally separate disclosure-driven enumeration (which should vanish
under redaction) from knowledge-driven enumeration (a model that infers the
held-out from public standards, which should survive).

Design invariant: redaction touches ONLY prompt-facing text --- the task
description, the kernel-signature block, and the seed comments shown in the
initial prompt. The in-distribution and held-out *evaluations* use the
original, unmodified task and seed source. Disclosure is thus the single
manipulated variable; everything downstream (sizes, scoring, correctness,
the held-out probe) is identical across the two arms.

Every redaction must match at least once (a spec edit that breaks a
``find`` string raises rather than silently no-opping), and after the
prompt is assembled the per-task denylist is asserted absent --- so a
``redacted`` run that still leaks the held-out identity fails loudly
instead of contaminating the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Redaction:
    """Replace an exact disclosing substring with a neutral one.

    ``find`` is matched verbatim against the description, the
    kernel-signature block, and the seed source; ``replace`` is the
    redacted text. ``note`` documents what is being removed and why, for
    the run's redaction manifest.
    """
    find: str
    replace: str
    note: str = ""


def redact_text(text: str, redactions: list[Redaction]) -> tuple[str, dict[str, int]]:
    """Apply ``redactions`` to ``text``; return (new_text, {find: n_hits})."""
    counts: dict[str, int] = {}
    for r in redactions:
        counts[r.find] = text.count(r.find)
        if counts[r.find]:
            text = text.replace(r.find, r.replace)
    return text, counts


def apply_redactions(
    description: str,
    kernel_signatures: str,
    seed_src: str,
    redactions: list[Redaction],
) -> tuple[str, str, str, list[dict]]:
    """Redact all three prompt-facing texts.

    Returns ``(description', signatures', seed', manifest)``. Raises
    ``ValueError`` if any redaction matches nowhere across the three texts
    (i.e. the spec drifted and the ``find`` string no longer exists), so
    the experiment can never run with a stale, ineffective redaction.
    """
    desc2, c_desc = redact_text(description, redactions)
    sig2, c_sig = redact_text(kernel_signatures, redactions)
    seed2, c_seed = redact_text(seed_src, redactions)

    manifest: list[dict] = []
    missed: list[str] = []
    for r in redactions:
        total = c_desc[r.find] + c_sig[r.find] + c_seed[r.find]
        where = [w for w, c in (("description", c_desc[r.find]),
                                ("signatures", c_sig[r.find]),
                                ("seed", c_seed[r.find])) if c]
        manifest.append({
            "find": r.find, "replace": r.replace, "note": r.note,
            "n_matches": total, "where": where,
        })
        if total == 0:
            missed.append(r.find)
    if missed:
        raise ValueError(
            "redaction find-string(s) not present in any prompt text "
            f"(spec drift?): {missed!r}"
        )
    return desc2, sig2, seed2, manifest


def assert_no_disclosure(authored_text: str, denylist: list[str]) -> None:
    """Raise if any held-out-identifying phrase survives in ``authored_text``.

    The integrity gate runs against the text the EXPERIMENT authors --- the
    redacted description, kernel signature, and displayed seed --- to verify
    our redaction actually removed the disclosure. It must NOT be run against
    model-generated content: a candidate that itself names the held-out
    (e.g. the model recalling Kyber's Dilithium sibling, or SHAKE128) is the
    knowledge-driven enumeration the experiment measures, and is recorded per
    iteration rather than suppressed. Feeding such a candidate back into the
    next prompt (as the (1+1) loop does) must not abort the run.
    """
    leaked = [p for p in denylist if p in authored_text]
    if leaked:
        raise ValueError(
            "redaction left held-out identity in authored prompt text "
            f"(denylist hits): {leaked!r} --- spec/seed redaction incomplete"
        )
