/-
# Probabilistic assembly steps for Theorem B.3 (`thm:upper`)

Two generic facts the proof of Theorem B.3 invokes:

* `union_bound_uniform` — union bound over the finite family of feedback-tree
  nodes at uniform per-node level `β / M`;
* `prob_le_of_condExp_le` — the "condition on the proposer's private
  randomness ω … integrate over ω" step: an a.s. conditional probability
  bound integrates to an unconditional one (the tower property, with
  conditional probability rendered as the conditional expectation of the
  event's indicator).
-/
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic

open MeasureTheory
open scoped ENNReal

namespace AppB

variable {Ω : Type*}

/-- Union bound over a finite family of `M` nodes at uniform level `β / M`.
No measurability needed (outer-measure subadditivity). -/
theorem union_bound_uniform [MeasurableSpace Ω] (P : Measure Ω)
    {V : Type*} [Fintype V] [Nonempty V] (Bad : V → Set Ω) (β : ℝ≥0∞)
    (h : ∀ v, P (Bad v) ≤ β / (Fintype.card V : ℝ≥0∞)) :
    P (⋃ v, Bad v) ≤ β := by
  refine le_trans (measure_iUnion_le _) ?_
  rw [tsum_fintype]
  calc ∑ v, P (Bad v)
      ≤ ∑ _v : V, β / (Fintype.card V : ℝ≥0∞) := Finset.sum_le_sum fun v _ => h v
    _ = (Fintype.card V : ℝ≥0∞) * (β / (Fintype.card V : ℝ≥0∞)) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ ≤ β := ENNReal.mul_div_le

/-- The "integrate over ω" step of Theorem B.3's proof: if the conditional
probability of the event `A` given a sub-σ-algebra `G` is a.s. at most `β`,
then `P(A) ≤ β`. -/
theorem prob_le_of_condExp_le {G m0 : MeasurableSpace Ω} (hG : G ≤ m0)
    (P : Measure[m0] Ω) [IsProbabilityMeasure P] {A : Set Ω}
    (hA : MeasurableSet[m0] A) {β : ℝ}
    (h : ∀ᵐ ω ∂P, (P[A.indicator (fun _ => (1 : ℝ)) | G]) ω ≤ β) :
    P.real A ≤ β := by
  have hint : Integrable (A.indicator fun _ => (1 : ℝ)) P :=
    (integrable_const (1 : ℝ)).indicator hA
  calc P.real A
      = ∫ ω, A.indicator (fun _ => (1 : ℝ)) ω ∂P := by
        simpa using (integral_indicator_one hA).symm
    _ = ∫ ω, (P[A.indicator (fun _ => (1 : ℝ)) | G]) ω ∂P := by
        rw [integral_condExp (m₀ := m0) (μ := P) (f := A.indicator fun _ => (1 : ℝ)) hG]
    _ ≤ ∫ _ω, β ∂P := integral_mono_ae integrable_condExp (integrable_const β) h
    _ = β := by simp
end AppB
