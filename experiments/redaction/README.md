# Controlled grade-C experiment: disclosure vs. knowledge

**Question (paper §6, weakness 2 / Limitations).** Three Metal-ZK task
specifications disclosed the held-out configuration, and the models
enumerated it and *passed* the gate by transcription rather than
generalization (grade C, §3.1). That evidence is **observational**: the
disclosure was authored, not randomly assigned. This experiment supplies
the **controlled** version, re-running the same tasks with the held-out
identity *redacted* from everything the model sees.

It causally separates two explanations for the enumeration:

| | prediction under redaction |
|---|---|
| **disclosure-driven** (the spec told it) | the held-out-config predicate disappears |
| **knowledge-driven** (it knows NIST PQC standards) | the predicate persists |

## Design

The single manipulated variable is **what the model sees**. Under
`--redact-held-out`, the held-out identity is stripped from the task
description, the kernel-signature block, and the seed comments
(`metal_zk/redact.py`; per-task `redactions` in the three task files). The
runtime-parameterization contract ("read the params from buffers,
hardcoding produces wrong/slow output") is **kept** — only the *identity*
of the unseen probe is removed.

Everything downstream is identical across arms: the in-distribution sizes,
the scoring function `S_T`, the bit-exact correctness gate, and the
held-out probe `Φ_T` itself. So any difference in enumeration or transfer
is attributable to disclosure alone.

Two integrity guarantees, both enforced at run time (fail loudly, never
silently degrade):
- **drift guard** — every declared redaction must match at least once, or
  `apply_redactions` raises (a future spec edit can't silently no-op it).
- **denylist** — once at setup, `assert_no_disclosure` asserts no
  held-out-identifying phrase survived in the text *we* author (redacted
  description + signature + displayed seed); a leaking redaction aborts.

**The denylist guards our authored text only — it does not police the
model.** A candidate that *itself* names the held-out (Claude recalling
that Kyber's NTT sibling is Dilithium; a model writing SHAKE128 or n_bytes
== 32) is the **knowledge-driven enumeration this experiment measures**, so
it is *recorded* (`model_named_held_out` per iteration and in
`summary.json`), never suppressed. The (1+1) loop feeds each candidate back
into the next prompt, so a per-prompt denylist would abort precisely on the
cells where the knowledge-driven effect is strongest — biasing the result
and discarding the signal. (This is why an earlier per-iteration gate was
removed.)

What changed (per `--dry-run-prompt`):
- **keccak_f1600_batch** — removes "the held-out size uses SHAKE128
  (rate=168, domain=0x1F, out=256…)"; keeps "scored on several
  (rate,out,domain) sets including configs not listed; out_bytes may
  exceed rate_bytes."
- **kyber_ntt** — removes the seed comment "(modulus; 3329 or 8380417)"
  → "(modulus; bound at runtime, fits in 32 bits)". (The description never
  named Dilithium; the seed comment was the only leak.)
- **wots_chain** — removes "held-out n_bytes=32" / "n_bytes 16 → 32" from
  the description (2 spots) and the seed comments (2 lines).

## Running it

Nothing here runs automatically. From the repo root:

```bash
# 0. Inspect exactly what redaction removes (no LLM, no GPU):
python run_benchmark.py --task keccak_f1600_batch --dry-run-prompt
python run_benchmark.py --task kyber_ntt          --dry-run-prompt
python run_benchmark.py --task wots_chain         --dry-run-prompt

# 1. The sweep: 3 tasks × 3 models × {disclosed, redacted} = 18 runs.
experiments/redaction/run_redaction_sweep.sh            # dry-run: prints commands
experiments/redaction/run_redaction_sweep.sh --go       # actually launch

# Re-running the disclosed arm is optional — the paper's existing runs can
# be reused. To run only the new control:
ARMS=redacted experiments/redaction/run_redaction_sweep.sh --go

# 2. Held-out evaluation of both arms (reuses the production evaluator):
python experiments/redaction/eval_held_out_redaction.py        # dry list
python experiments/redaction/eval_held_out_redaction.py --go   # evaluate on GPU

# 3. Analysis: per-cell table + disclosed-vs-redacted predicate contingency.
python experiments/redaction/analyze_redaction.py
```

Single-cell smoke test before committing to the full sweep:

```bash
python run_benchmark.py --task kyber_ntt --model gpt-5.5 \
    --iterations 12 --redact-held-out --output-dir results/redaction_experiment
```

Knobs (env vars on the sweep script): `ITER` (default 12), `TASKS`,
`MODELS`, `ARMS`, `OUT`.

## Outputs

Runs land in `results/redaction_experiment/<task>_<model>_<ts>[_redacted]/`,
each with the usual artifacts plus, for redacted runs, a
`redaction_manifest.json` recording exactly what was swapped (auditable;
`summary.json` carries `"redacted": true`). The analysis writes
`held_out_redaction.json` and `analyze_redaction.md`.

## Reading the result

Three readouts per cell:
- **`Φ_T` transfer** (`ho_speedup`, `held_correct`) — the objective outcome.
- **predicate screen** — does `best.metal` branch on the held-out identity?
  A cheap static pass à la Guidance #4; it flags candidates but a
  differentially-tuned enumeration (grade C) vs a generic correctness branch
  (e.g. wots/Opus's shared `n_lanes>=4` arm, adjudicated as genuine transfer
  in Appendix B) still needs the mechanism audit.
- **`model_named`** — did the model surface the held-out identity unprompted
  during the search (any iteration, code or prose)? The stronger
  process-level knowledge-driven signal.

The headline contrast is disclosed-vs-redacted: enumeration that **vanishes**
under redaction is disclosure-driven; enumeration (or `model_named` events)
that **persists** is knowledge-driven — the model inferred the held-out from
public standards. A partial result (even one task × three models) already
sharpens the grade-C claim from "observed" to "controlled".
