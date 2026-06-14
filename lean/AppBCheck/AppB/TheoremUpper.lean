/-
# Appendix B, Theorem B.3 (`thm:upper`): validity under bounded-leakage reuse

Paper proof structure (after conditioning on the proposer's private
randomness ω, under which the pool `D` is still i.i.d.):

1. the feedback tree has at most `M + 1 ≤ 2^{k+1}` candidates (including the
   initial incumbent — `AppB.tree_count_le_total`), each node `v`
   determining a *fixed* candidate `c_v` — a fixed function of `v`, fixed
   before `D` is drawn;
2. per node, one-sided Hoeffding for the fixed `c_v` on the i.i.d. pool
   gives `P(J(c_v) < Ĵ_D(c_v) − r) ≤ β/2^{k+1}` — here a hypothesis `hEB`
   (Hoeffding's inequality is the cited classical primitive); the radius
   arithmetic `exp(−2Nr²/B²) = β/2^{k+1}` at the displayed
   `r = B√(((k+1)ln2 + ln(1/β))/(2N))` is verified *exactly* in
   `hoeffding_level`;
3. union bound over the `≤ 2^{k+1}` nodes;
4. pointwise: in any realized run, every queried candidate (and the output
   `ĉ`) equals the node candidate at the realized feedback prefix, so the
   realized failure event is contained in the union of node failure events;
5. integrate over ω (the tower-property step `AppB.prob_le_of_condExp_le`).

`thm_upper` formalizes steps 3–4 at a generic per-node level `β/M`:
`Bad v` is the node-`v` failure event `{D : J(c_v) < Ĵ_D(c_v) − r}`, and the
realized failure event at round `j` is `{ω | ω ∈ Bad (node j ω)}` — the
candidate actually scored at round `j` is the one sitting at the realized
feedback prefix.  The number of rounds may be data-dependent (any `node`
function is allowed, and rounds beyond the horizon can repeat a node: unions
are insensitive to that).  `thm_upper_fixed_level` specializes to the
paper's convention: every node scored at the *fixed* level `β/2^{k+1}`.

`remark_bit_budgets_vacuous` verifies the Remark (bit budgets): with
full-precision float feedback, `k` is of order 64 × (configurations) ×
(rounds) — already at 3 configurations × 10 rounds (`k = 1920`) and a pool
of `N = 3` configurations the certified radius exceeds the payoff range `B`
for every confidence level, so the bound is vacuous.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Real.Sqrt
import AppB.UnionBound

open MeasureTheory
open scoped ENNReal

namespace AppB

/-- **Theorem B.3, validity (generic level).**  Nodes `V` (the feedback
prefixes), per-node failure events `Bad v` of probability ≤ `β/M`
(one-sided Hoeffding applied to the fixed candidate `c_v` — hypothesis
`hEB`), and an arbitrary realized-prefix map `node j : Ω → V`.  The
probability that *some* realized round fails is at most `β`, for every
proposer (the proposer only influences `node`). -/
theorem thm_upper {Ω V : Type*} [MeasurableSpace Ω] [Fintype V] [Nonempty V]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (Bad : V → Set Ω) (β : ℝ≥0∞)
    (hEB : ∀ v, P (Bad v) ≤ β / (Fintype.card V : ℝ≥0∞))
    (J : ℕ) (node : ℕ → Ω → V) :
    P (⋃ j ∈ Finset.range J, {ω | ω ∈ Bad (node j ω)}) ≤ β := by
  have hsub : (⋃ j ∈ Finset.range J, {ω | ω ∈ Bad (node j ω)}) ⊆ ⋃ v, Bad v := by
    intro ω hω
    simp only [Set.mem_iUnion, Set.mem_setOf_eq] at hω
    obtain ⟨j, _, hj⟩ := hω
    exact Set.mem_iUnion.mpr ⟨node j ω, hj⟩
  exact le_trans (measure_mono hsub) (union_bound_uniform P Bad β hEB)

/-- **Theorem B.3, validity at the fixed level `β/2^{k+1}`** (the paper's
convention): if every node fails with probability at most `β/2^{k+1}` and
there are `card V ≤ 2^{k+1}` nodes (`tree_count_le_total`), the realized
failure probability is at most `β`.  Reduces to `thm_upper` since
`β/2^{k+1} ≤ β/M`. -/
theorem thm_upper_fixed_level {Ω V : Type*} [MeasurableSpace Ω] [Fintype V] [Nonempty V]
    (P : Measure Ω) [IsProbabilityMeasure P]
    (Bad : V → Set Ω) (β : ℝ≥0∞) (k : ℕ)
    (hcard : (Fintype.card V : ℝ≥0∞) ≤ 2 ^ (k + 1))
    (hEB : ∀ v, P (Bad v) ≤ β / 2 ^ (k + 1))
    (J : ℕ) (node : ℕ → Ω → V) :
    P (⋃ j ∈ Finset.range J, {ω | ω ∈ Bad (node j ω)}) ≤ β :=
  thm_upper P Bad β
    (fun v => le_trans (hEB v) (ENNReal.div_le_div_left hcard β)) J node

/-- **Theorem B.3, radius arithmetic.**  At the radius
`r := B√(((k+1)ln2 + ln(1/β))/(2N))` of the theorem statement, the one-sided
Hoeffding tail `exp(−2Nr²/B²)` for an `[a,b]`-valued payoff with range `B`
equals the per-node level `β/2^{k+1}` *exactly*, so the union bound over the
`≤ 2^{k+1}` tree candidates (`thm_upper_fixed_level`) yields overall failure
probability `≤ β`. -/
theorem hoeffding_level (B β : ℝ) (N k : ℕ) (hB : 0 < B) (hN : 1 ≤ N)
    (hβ0 : 0 < β) (hβ1 : β ≤ 1) :
    Real.exp (-(2 * N * (B * Real.sqrt ((((k : ℝ) + 1) * Real.log 2
        + Real.log (1 / β)) / (2 * N))) ^ 2 / B ^ 2))
      = β / 2 ^ (k + 1) := by
  have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
  have hlogβ : 0 ≤ Real.log (1 / β) :=
    Real.log_nonneg (by rw [le_div_iff₀ hβ0]; linarith)
  have hl2 : 0 ≤ Real.log 2 := Real.log_nonneg one_le_two
  have harg : 0 ≤ (((k : ℝ) + 1) * Real.log 2 + Real.log (1 / β)) / (2 * N) := by
    positivity
  have hsq : Real.sqrt ((((k : ℝ) + 1) * Real.log 2 + Real.log (1 / β)) / (2 * N)) ^ 2
      = (((k : ℝ) + 1) * Real.log 2 + Real.log (1 / β)) / (2 * N) :=
    Real.sq_sqrt harg
  have hexp : 2 * N * (B * Real.sqrt ((((k : ℝ) + 1) * Real.log 2
        + Real.log (1 / β)) / (2 * N))) ^ 2 / B ^ 2
      = ((k : ℝ) + 1) * Real.log 2 + Real.log (1 / β) := by
    rw [mul_pow, hsq]
    field_simp
  have hrhs : Real.log (β / 2 ^ (k + 1)) = Real.log β - ((k : ℝ) + 1) * Real.log 2 := by
    rw [Real.log_div hβ0.ne' (by positivity), Real.log_pow]
    push_cast
    ring
  have hkey : -(((k : ℝ) + 1) * Real.log 2 + Real.log (1 / β))
      = Real.log (β / 2 ^ (k + 1)) := by
    rw [one_div, Real.log_inv, hrhs]
    ring
  rw [hexp, hkey, Real.exp_log (by positivity)]

/-- **Remark (bit budgets).**  Full-precision feedback makes the certificate
vacuous: at `k = 64 × 3 × 10 = 1920` leaked bits (three configurations of
float feedback over ten rounds) and a pool of `N = 3` configurations, the
certified radius `r = B√(((k+1)ln2 + ln(1/β))/(2N))` exceeds the payoff
range `B` for every confidence level `β ≤ 1` (numerically `r ≈ 14.9·B` at
`β = 0.05`). -/
theorem remark_bit_budgets_vacuous (B β : ℝ) (hB : 0 < B)
    (hβ0 : 0 < β) (hβ1 : β ≤ 1) :
    B < B * Real.sqrt ((((1920 : ℝ) + 1) * Real.log 2 + Real.log (1 / β))
      / (2 * 3)) := by
  have hlogβ : 0 ≤ Real.log (1 / β) :=
    Real.log_nonneg (by rw [le_div_iff₀ hβ0]; linarith)
  have hl2 : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have harg : (1 : ℝ) < (((1920 : ℝ) + 1) * Real.log 2 + Real.log (1 / β))
      / (2 * 3) := by
    rw [lt_div_iff₀ (by norm_num : (0 : ℝ) < 2 * 3)]
    nlinarith
  have h1 : (1 : ℝ) < Real.sqrt ((((1920 : ℝ) + 1) * Real.log 2
      + Real.log (1 / β)) / (2 * 3)) := by
    have h := Real.sqrt_lt_sqrt zero_le_one harg
    rwa [Real.sqrt_one] at h
  calc B = B * 1 := by ring
    _ < B * Real.sqrt ((((1920 : ℝ) + 1) * Real.log 2 + Real.log (1 / β))
        / (2 * 3)) := mul_lt_mul_of_pos_left h1 hB

end AppB
