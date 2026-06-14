/-
# Appendix B, Definition B.2 (`def:adaptive`): leakage and node count

Paper claim: with feedback alphabets `𝔽_1, …, 𝔽_T` and leakage
`k := ∑_{t ≤ T} log₂|𝔽_t|`, the number of distinct feedback prefixes at
which a query can be issued,
  `M := ∑_{t ≤ T} ∏_{i<t} |𝔽_i|`,
satisfies `M ≤ 2^k` whenever every `|𝔽_i| ≥ 2`; the proof of Theorem B.3
additionally counts the initial incumbent: `M + 1 ≤ 2^{k+1}`.

We index the alphabets as `a 0, …, a (J-1)` with `J = T` rounds, so `M`
reads `∑_{j ∈ range J} ∏_{i ∈ range j} a i` and `2^k = ∏_{i ∈ range J} a i`
(definition of `k`).  The engine is the geometric-sum bound
`node_count_le`: the prefix count up to and including depth `J` is at most
twice the product of the first `J` alphabet sizes.
-/
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

open Finset

namespace AppB

/-- Geometric-sum engine: with every feedback alphabet of size `≥ 2`, the
number of feedback-prefix nodes up to depth `J` (inclusive) is at most twice
the product of the first `J` alphabet sizes. -/
theorem node_count_le (J : ℕ) (a : ℕ → ℕ) (ha : ∀ i < J, 2 ≤ a i) :
    ∑ j ∈ range (J + 1), ∏ i ∈ range j, a i ≤ 2 * ∏ i ∈ range J, a i := by
  induction J with
  | zero => simp
  | succ J ih =>
    have hJ : 2 ≤ a J := ha J (by omega)
    have ih' : ∑ j ∈ range (J + 1), ∏ i ∈ range j, a i ≤ 2 * ∏ i ∈ range J, a i :=
      ih fun i hi => ha i (by omega)
    have hstep : 2 * ∏ i ∈ range J, a i ≤ ∏ i ∈ range (J + 1), a i := by
      rw [prod_range_succ, mul_comm]
      exact Nat.mul_le_mul_left _ hJ
    calc ∑ j ∈ range (J + 2), ∏ i ∈ range j, a i
        = (∑ j ∈ range (J + 1), ∏ i ∈ range j, a i) + ∏ i ∈ range (J + 1), a i :=
          sum_range_succ _ _
      _ ≤ 2 * ∏ i ∈ range J, a i + ∏ i ∈ range (J + 1), a i := by omega
      _ ≤ ∏ i ∈ range (J + 1), a i + ∏ i ∈ range (J + 1), a i := by omega
      _ = 2 * ∏ i ∈ range (J + 1), a i := by ring

/-- **Definition B.2's node-count bound** (product form): with every feedback
alphabet of size `≥ 2`, the number of feedback-prefix nodes
`M = ∑_{t ≤ T} ∏_{i<t} |𝔽_i|` is at most the product of *all* `J = T`
alphabet sizes. -/
theorem node_count_le_total (J : ℕ) (a : ℕ → ℕ) (ha : ∀ i < J, 2 ≤ a i) :
    ∑ j ∈ range J, ∏ i ∈ range j, a i ≤ ∏ i ∈ range J, a i := by
  cases J with
  | zero => simp
  | succ J' =>
    have hJ : 2 ≤ a J' := ha J' (by omega)
    calc ∑ j ∈ range (J' + 1), ∏ i ∈ range j, a i
        ≤ 2 * ∏ i ∈ range J', a i := node_count_le J' a fun i hi => ha i (by omega)
      _ ≤ a J' * ∏ i ∈ range J', a i := Nat.mul_le_mul_right _ hJ
      _ = ∏ i ∈ range (J' + 1), a i := by rw [prod_range_succ, mul_comm]

/-- **Definition B.2, `2^k` form**: if the product of *all* alphabet sizes is
`2^k` (i.e. `k = ∑_{t ≤ T} log₂|𝔽_t|` is the total transcript length), then
`M ≤ 2^k`. -/
theorem node_count_le_two_pow_total (J k : ℕ) (a : ℕ → ℕ)
    (ha : ∀ i < J, 2 ≤ a i) (hk : ∏ i ∈ range J, a i = 2 ^ k) :
    ∑ j ∈ range J, ∏ i ∈ range j, a i ≤ 2 ^ k :=
  hk ▸ node_count_le_total J a ha

/-- **Theorem B.3, proof step**: "including the initial incumbent, the tree
contains at most `M + 1 ≤ 2^{k+1}` candidates". -/
theorem tree_count_le_total (J k : ℕ) (a : ℕ → ℕ)
    (ha : ∀ i < J, 2 ≤ a i) (hk : ∏ i ∈ range J, a i = 2 ^ k) :
    (∑ j ∈ range J, ∏ i ∈ range j, a i) + 1 ≤ 2 ^ (k + 1) := by
  have hM := node_count_le_two_pow_total J k a ha hk
  have h1 : 1 ≤ 2 ^ k := Nat.one_le_two_pow
  calc (∑ j ∈ range J, ∏ i ∈ range j, a i) + 1
      ≤ 2 ^ k + 2 ^ k := by omega
    _ = 2 ^ (k + 1) := by rw [pow_succ]; ring

end AppB
