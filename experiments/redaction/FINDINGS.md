# Redaction experiment — findings (search arm)

18/18 cells complete (3 tasks × 3 models × {disclosed, redacted}). Held-out
GPU transfer (`Φ_T`) not yet evaluated — run `eval_held_out_redaction.py
--go` on the Mac for HO×. The grade-C causal question is answered by the
**enumeration** outcome, which is static (no GPU needed).

Method note: the regex predicate screen is a first pass; each flagged
`best.metal` was hand-read to separate a *dedicated, differentially-tuned
held-out arm* (the grade-C signature) from a *generic correctness construct*
that merely mentions the held-out value (a known false-positive class, e.g.
a pad-placement table indexed by every possible width).

## Result: enumeration of the held-out, disclosed vs redacted

| task (held-out) | model | disclosed | redacted | verdict |
|---|---|---|---|---|
| keccak (SHAKE128: rate=168,0x1F,out=256) | Opus   | none | none | never enumerated |
| keccak | Gemini | **dedicated SHAKE arm** (`==168`,`out_bytes==256`,`0x1F`) | none (generic) | **disclosure-driven** |
| keccak | GPT-5.5 | **dedicated SHAKE arm** (`==168`,`out_bytes==256`,`0x1F`) | none (generic) | **disclosure-driven** |
| kyber (Dilithium q=8380417) | Opus   | none | none (prose "Dilithium" appeared once in an earlier crashed run, absent in the completed run) | never enumerated |
| kyber | Gemini | none | none | never enumerated |
| kyber | GPT-5.5 | **`q==8380417` dispatch** (8380417 ×25) | none (generic) | **disclosure-driven** |
| wots (n_bytes=32 → n_lanes=4) | Opus   | generic (shared permutation; `n_lanes==4` is a comment) | generic | never enumerated (matches App. B adjudication) |
| wots | Gemini | **dedicated `else if (n_lanes==4u)` vectorized arm** | generic pad-table over all widths (no dedicated arm) | **disclosure-driven** |
| wots | GPT-5.5 | none (this sweep) | **dedicated tuned arms for n_lanes∈{2,3,4,8,16}**, and emits `n_bytes=32` | **knowledge-driven** |

## Reading

**Disclosure-driven enumeration is real and removable.** Every cell that
enumerated the held-out under disclosure (keccak/Gemini, keccak/GPT,
kyber/GPT, wots/Gemini — 4 cells) **stopped enumerating under redaction**.
The held-out-specific constants that have no other meaning — SHAKE128's
`168` / `0x1F` / `out_bytes==256`, Dilithium's `8380417` — appear *only* in
disclosed winners and vanish entirely when the spec no longer names them.
The models did not reach for those parameters unprompted. This converts the
grade-C claim from observational to controlled: the spec disclosure *causes*
the enumeration.

**The one persisting case is knowledge-driven, and it validates Guidance
#2.** wots/GPT-5.5 enumerates `n_lanes==4` (= the held-out 32-byte width)
*even under redaction*, and writes `n_bytes=32`. This is exactly the
predicted failure of an *enumerable, standardized* axis: 32-byte (256-bit)
chains are the canonical SPHINCS+/WOTS+ width, so a model that knows the
scheme reaches for it without being told. Crucially the mechanism differs
from grade C — GPT wrote *tuned arms for a whole range of widths*
(2,3,4,8,16), each fully built, not a tuned measured arm beside a neglected
held-out one. That is defensive generalization, not a neglected-arm attack;
it should transfer (pending HO×), unlike the disclosed grade-C cells.

**n=1 caveat (already a paper limitation).** These disclosed runs are fresh
sweeps, not the paper's originals, and enumeration is stochastic:
wots/GPT-5.5 disclosed did *not* reproduce the grade-C branch this sweep,
and kyber/Opus's transient "Dilithium" prose mention did not recur. The
clean signal is the within-experiment contrast: of the 4 disclosed cells
that did enumerate, 4/4 stopped under redaction; the sole redacted-arm
enumeration is on the one axis inferable from public standards.

## Held-out transfer (HO×), disclosed vs redacted

| task | model | disclosed HO× | redacted HO× | note |
|---|---|---|---|---|
| keccak | Opus   | 14.03 | 15.95 | both transfer (neither enumerated) |
| keccak | Gemini | 14.30 *(enum)* | 11.01 | enumeration **gratuitous** — generic transfers |
| keccak | GPT    | 10.49 *(enum)* | 10.53 | enumeration **gratuitous** — generic transfers |
| kyber  | Opus   | **FAIL** (held-out incorrect) | 2.29 | disclosed broke on Dilithium; redacted transfers |
| kyber  | Gemini | **0.54** (regression) | 2.17 | disclosed overfit (no enum); redacted transfers |
| kyber  | GPT    | 2.31 *(enum)* | **0.68** (regression) | **load-bearing enumeration**: the gate pass was transcription |
| wots   | Opus   | 17.43 | 16.82 | both transfer |
| wots   | Gemini | 16.33 *(enum)* | 20.32 | enumeration gratuitous |
| wots   | GPT    | 20.90 | 20.79 *(knowledge-enum, all arms tuned)* | both transfer |

Aggregate gate-pass (HO ≥ 1.05× & correct) is **flat: disclosed 7/9,
redacted 8/9** — so the experiment does **not** support a blanket "redaction
reduces transfer". The effect is mechanism-specific.

## Synthesis (the precise claim)

Two findings, both honest about n=1:

1. **Enumeration is disclosure-driven (4/4).** Every disclosed winner that
   branched on the held-out identity stopped doing so under redaction; the
   opaque held-out constants (SHAKE128's `168`/`0x1F`/`out=256`, Dilithium's
   `8380417`) never appear unprompted. Disclosure *causes* the enumeration.

2. **Gate-leakage inflates the gate only when the held-out is arithmetically
   distinct from the measured set.** Pairing each disclosure-driven
   enumeration with its redacted twin tells us whether the enumeration was
   load-bearing:
   - keccak/Gemini, keccak/GPT, wots/Gemini → redacted generic kernel
     **still transfers** (11.0, 10.5, 20.3). The held-out shares the kernel
     (the same Keccak-f1600 permutation; only width/mode params differ), so a
     generic solution generalizes and the enumeration was a **gratuitous
     shortcut** the disclosure invited.
   - kyber/GPT → redacted kernel **regresses to 0.68×**. Dilithium's modulus
     genuinely changes the modular arithmetic, so the disclosed pass (2.31×)
     was *transcription of the leaked modulus*; remove it and the model
     overfits to Kyber's `q=3329` and fails to generalize. This is the clean,
     mechanism-matched demonstration that a grade-C "transfer" was the gate
     measuring enumeration, not generalization.

3. **The persisting enumeration is knowledge-driven and benign.** wots/GPT
   enumerates `n_lanes∈{2,3,4,8,16}` even under redaction (256-bit chains are
   the canonical SPHINCS+ width) — but it tunes *every* arm and transfers
   (20.79), so it is defensive generalization, not a neglected-arm attack.
   This validates Guidance #2: non-disclosure protects a probe only when the
   held-out is also **non-inferable** from public standards.

Noise caveat (n=1): the kyber/Opus and kyber/Gemini disclosed arms
underperform their redacted twins, but neither enumerated — these are
ordinary single-sweep trajectory differences (one a held-out correctness
failure, one a non-enumeration overfit), not a disclosure effect.

## Bottom line for the paper

The controlled experiment **confirms the grade-C causal mechanism**
(disclosure → enumeration, 4/4 removed by redaction) and **sharpens it**:
gate leakage reliably induces enumeration, but only *inflates the held-out
score* when the probe is arithmetically non-trivial — exactly when a probe
is most worth having. kyber/GPT (2.31×→0.68×) is the headline case study;
keccak/wots show the leakage is latent when a generic solution already
transfers. Frame as confirmation + refinement, not as a population-level
transfer effect (aggregate pass rate is unchanged).
