/-
# Lean 4 verification of Appendix B ("Evaluation reuse with instance
# fingerprinting: a self-contained account") of
# "Gaming Without an Attacker: Benchmark Fingerprinting in LLM-Driven
#  Search Under Selection Pressure"  (workshop_paper/main.tex)

Module map (paper item → Lean file):

* Definition B.2 (`def:adaptive`) node count `M ≤ 2^k`,
  tree count `M + 1 ≤ 2^{k+1}`                      → `AppB.NodeCount`
* union-bound / tower-property assembly steps       → `AppB.UnionBound`
* Theorem B.3 (`thm:upper`) validity + radius       → `AppB.TheoremUpper`
* Remark (bit budgets) vacuousness instance         → `AppB.TheoremUpper`
* Lemma B.5 (`lem:mad`) anti-concentration (full)   → `AppB.LemmaMad`
* Chernoff lower tail for Bin(N,p) (full)           → `AppB.Chernoff`
* Theorem B.6 (`thm:lowerapp`) attack, (a),(b),(c),
  certificates                                      → `AppB.TheoremLower`
* Theorem B.6(b) on the pool space, bridge-free     → `AppB.PairMoments`
* §B.7 (`app:enum`) weighted pool space: product
  laws, centered cell moments, Cauchy–Schwarz       → `AppB.WeightedPool`
* Theorem B.8 (`thm:dict-app`) enumerable ⇒ richness
  ⇒ attack fires; Theorem B.9 (`thm:nowitness`) no
  identity witness on a diffuse law; Theorem B.10
  (`thm:starve`) starvation bound + two-sided       → `AppB.Enumerability`

(Assumption B.4 (`ass:richness`) is not a claim; it is *modeled* in
`AppB.TheoremLower` / `AppB.PairMoments` as the payoff
`Y(c_v, ξ) = μ₀ + σ·v(φ(ξ))` with uniform fingerprint cells.  The
enumerability layer of `AppB.Enumerability` adds no new classical input:
its only inequality is Cauchy–Schwarz, proved from first principles in
`AppB.WeightedPool`.)
-/
import AppB.NodeCount
import AppB.UnionBound
import AppB.TheoremUpper
import AppB.LemmaMad
import AppB.Chernoff
import AppB.TheoremLower
import AppB.PairMoments
import AppB.WeightedPool
import AppB.Enumerability
