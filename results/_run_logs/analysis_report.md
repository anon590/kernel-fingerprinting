# metal-zk benchmark summary

Tasks:  binius_clmul, fri_round, goldilocks_ntt, keccak_f1600_batch, kyber_ntt, logup_gkr, merkle_build, montgomery_msm, multilinear_sumcheck_round, pippenger_buckets, poseidon2_hash, wots_chain
Models: claude-opus-4-7, gemini-3.1-pro-preview, gpt-5.5

_Scope: latest usable run (has `summary.json`) per (task, model), restricted to claude-opus-4-7, gemini-3.1-pro-preview, gpt-5.5. Other models on disk — e.g. the exploratory gemini-3.5-flash sweeps — are excluded via `_common.MODELS`._

_No completed run (omitted from every section): `pippenger_buckets/gpt-5.5`._

## In-distribution scores (gmean fraction-of-ceiling over in-dist sizes)

| Task | Model | iters | Seed | Best | Speedup | Best iter | Compile fails | Correctness fails | Wall time | Run dir |
|---|---|---|---|---|---|---|---|---|---|---|
| binius_clmul | claude-opus-4-7 | 10 | 0.0937 | 0.1968 | **2.10×** | 5 | 0/10 | 1/10 | 7.2 min | binius_clmul_claude-opus-4-7_20260517_172042 |
| binius_clmul | gemini-3.1-pro-preview | 10 | 0.0852 | 0.2928 | **3.44×** | 4 | 0/10 | 0/10 | 45.4 min | binius_clmul_gemini-3.1-pro-preview_20260518_121813 |
| binius_clmul | gpt-5.5 | 10 | 0.1041 | 0.4425 | **4.25×** | 10 | 0/10 | 0/10 | 69.2 min | binius_clmul_gpt-5.5_20260518_145759 |
| fri_round | claude-opus-4-7 | 10 | 0.2736 | 0.3527 | **1.29×** | 10 | 0/10 | 0/10 | 8.5 min | fri_round_claude-opus-4-7_20260517_095429 |
| fri_round | gemini-3.1-pro-preview | 10 | 0.2530 | 0.2599 | **1.03×** | 3 | 0/10 | 2/10 | 38.8 min | fri_round_gemini-3.1-pro-preview_20260518_130712 |
| fri_round | gpt-5.5 | 10 | 0.2523 | 0.3372 | **1.34×** | 10 | 0/10 | 0/10 | 60.4 min | fri_round_gpt-5.5_20260518_134705 |
| goldilocks_ntt | claude-opus-4-7 | 10 | 0.3798 | 0.3798 | **1.00×** | 0 | 0/9 | 0/9 | 3.9 min | goldilocks_ntt_claude-opus-4-7_20260515_110752 |
| goldilocks_ntt | gemini-3.1-pro-preview | 10 | 0.2846 | 0.3990 | **1.40×** | 4 | 2/10 | 1/10 | 35.5 min | goldilocks_ntt_gemini-3.1-pro-preview_20260514_225332 |
| goldilocks_ntt | gpt-5.5 | 10 | 0.4050 | 0.4050 | **1.00×** | 0 | 0/10 | 0/10 | 64.1 min | goldilocks_ntt_gpt-5.5_20260516_181241 |
| keccak_f1600_batch | claude-opus-4-7 | 15 | 0.0758 | 0.9638 | **12.72×** | 5 | 0/15 | 1/15 | 12.5 min | keccak_f1600_batch_claude-opus-4-7_20260515_100640 |
| keccak_f1600_batch | gemini-3.1-pro-preview | 15 | 0.0754 | 0.7844 | **10.41×** | 11 | 0/15 | 1/15 | 65.6 min | keccak_f1600_batch_gemini-3.1-pro-preview_20260516_135900 |
| keccak_f1600_batch | gpt-5.5 | 15 | 0.0764 | 0.7010 | **9.18×** | 11 | 0/15 | 0/15 | 82.1 min | keccak_f1600_batch_gpt-5.5_20260516_191646 |
| kyber_ntt | claude-opus-4-7 | 10 | 0.0037 | 0.0120 | **3.29×** | 9 | 0/10 | 4/10 | 5.5 min | kyber_ntt_claude-opus-4-7_20260517_140020 |
| kyber_ntt | gemini-3.1-pro-preview | 10 | 0.0037 | 0.0072 | **1.96×** | 6 | 0/10 | 0/10 | 36.6 min | kyber_ntt_gemini-3.1-pro-preview_20260519_100611 |
| kyber_ntt | gpt-5.5 | 10 | 0.0037 | 0.0144 | **3.92×** | 8 | 1/10 | 0/10 | 108.1 min | kyber_ntt_gpt-5.5_20260517_140621 |
| logup_gkr | claude-opus-4-7 | 10 | 0.0010 | 0.0446 | **43.98×** | 6 | 0/10 | 0/10 | 7.5 min | logup_gkr_claude-opus-4-7_20260515_121734 |
| logup_gkr | gemini-3.1-pro-preview | 10 | 0.0011 | 0.0390 | **36.52×** | 3 | 1/10 | 0/10 | 40.7 min | logup_gkr_gemini-3.1-pro-preview_20260515_133024 |
| logup_gkr | gpt-5.5 | 10 | 0.0010 | 0.0466 | **45.97×** | 3 | 0/10 | 0/10 | 62.7 min | logup_gkr_gpt-5.5_20260515_122603 |
| merkle_build | claude-opus-4-7 | 10 | 0.3079 | 0.3730 | **1.21×** | 5 | 0/10 | 1/10 | 10.9 min | merkle_build_claude-opus-4-7_20260515_112851 |
| merkle_build | gemini-3.1-pro-preview | 10 | 0.3042 | 0.4097 | **1.35×** | 2 | 0/10 | 0/10 | 43.9 min | merkle_build_gemini-3.1-pro-preview_20260516_150439 |
| merkle_build | gpt-5.5 | 10 | 0.3113 | 0.4397 | **1.41×** | 9 | 0/10 | 0/10 | 59.1 min | merkle_build_gpt-5.5_20260516_203852 |
| montgomery_msm | claude-opus-4-7 | 10 | 0.0039 | 0.0106 | **2.70×** | 3 | 0/10 | 0/10 | 26.7 min | montgomery_msm_claude-opus-4-7_20260516_114712 |
| montgomery_msm | gemini-3.1-pro-preview | 10 | 0.0039 | 0.0070 | **1.77×** | 9 | 0/10 | 0/10 | 69.6 min | montgomery_msm_gemini-3.1-pro-preview_20260516_161611 |
| montgomery_msm | gpt-5.5 | 10 | 0.0040 | 0.0068 | **1.72×** | 2 | 1/10 | 0/10 | 82.4 min | montgomery_msm_gpt-5.5_20260516_121529 |
| multilinear_sumcheck_round | claude-opus-4-7 | 10 | 0.0152 | 0.1239 | **8.14×** | 5 | 1/10 | 0/10 | 8.1 min | multilinear_sumcheck_round_claude-opus-4-7_20260517_173349 |
| multilinear_sumcheck_round | gemini-3.1-pro-preview | 10 | 0.0146 | 0.1066 | **7.27×** | 2 | 1/9 | 0/9 | 37.1 min | multilinear_sumcheck_round_gemini-3.1-pro-preview_20260518_182527 |
| multilinear_sumcheck_round | gpt-5.5 | 10 | 0.0126 | 0.1272 | **10.10×** | 2 | 0/10 | 0/10 | 55.0 min | multilinear_sumcheck_round_gpt-5.5_20260518_165514 |
| pippenger_buckets | claude-opus-4-7 | 10 | 0.0004 | 0.0034 | **8.36×** | 7 | 0/10 | 0/10 | 24.4 min | pippenger_buckets_claude-opus-4-7_20260516_155047 |
| pippenger_buckets | gemini-3.1-pro-preview | 10 | 0.0004 | 0.0029 | **6.87×** | 8 | 1/10 | 1/10 | 62.2 min | pippenger_buckets_gemini-3.1-pro-preview_20260516_223832 |
| poseidon2_hash | claude-opus-4-7 | 10 | 0.2231 | 0.3635 | **1.63×** | 5 | 0/10 | 0/10 | 11.3 min | poseidon2_hash_claude-opus-4-7_20260515_104542 |
| poseidon2_hash | gemini-3.1-pro-preview | 10 | 0.0289 | 0.0314 | **1.09×** | 7 | 0/10 | 1/10 | 52.8 min | poseidon2_hash_gemini-3.1-pro-preview_20260515_075818 |
| poseidon2_hash | gpt-5.5 | 10 | 0.2456 | 0.3076 | **1.25×** | 7 | 0/10 | 0/10 | 58.5 min | poseidon2_hash_gpt-5.5_20260516_213756 |
| wots_chain | claude-opus-4-7 | 10 | 0.0772 | 1.3370 | **17.31×** | 2 | 0/10 | 1/10 | 10.7 min | wots_chain_claude-opus-4-7_20260515_180734 |
| wots_chain | gemini-3.1-pro-preview | 10 | 0.0757 | 1.2440 | **16.43×** | 10 | 1/10 | 0/10 | 46.9 min | wots_chain_gemini-3.1-pro-preview_20260516_172545 |
| wots_chain | gpt-5.5 | 10 | 0.0773 | 1.2108 | **15.66×** | 6 | 1/10 | 0/10 | 48.0 min | wots_chain_gpt-5.5_20260515_182422 |

## Best-candidate per-size breakdown (in-dist, fraction of ceiling)

- **binius_clmul / claude-opus-4-7** — gf128_N64K: 17.6% | gf128_N256K: 20.5% | gf128_N1M: 21.2%
- **binius_clmul / gemini-3.1-pro-preview** — gf128_N64K: 12.0% | gf128_N256K: 44.8% | gf128_N1M: 46.6%
- **binius_clmul / gpt-5.5** — gf128_N64K: 21.4% | gf128_N256K: 78.4% | gf128_N1M: 51.7%
- **fri_round / claude-opus-4-7** — f2_N64K: 17.2% | f2_N256K: 44.5% | f2_N1M: 57.4%
- **fri_round / gemini-3.1-pro-preview** — f2_N64K: 16.2% | f2_N256K: 29.2% | f2_N1M: 37.2%
- **fri_round / gpt-5.5** — f2_N64K: 16.4% | f2_N256K: 38.5% | f2_N1M: 61.0%
- **goldilocks_ntt / claude-opus-4-7** — N2_14: 16.9% | N2_16: 36.9% | N2_18: 88.1%
- **goldilocks_ntt / gemini-3.1-pro-preview** — N2_14: 24.6% | N2_16: 57.8% | N2_18: 44.6%
- **goldilocks_ntt / gpt-5.5** — N2_14: 16.8% | N2_16: 85.2% | N2_18: 46.4%
- **keccak_f1600_batch / claude-opus-4-7** — sha3_256_B16K: 69.2% | sha3_256_B256K: 114.7% | sha3_256_B4M: 112.7%
- **keccak_f1600_batch / gemini-3.1-pro-preview** — sha3_256_B16K: 30.2% | sha3_256_B256K: 124.8% | sha3_256_B4M: 128.2%
- **keccak_f1600_batch / gpt-5.5** — sha3_256_B16K: 33.3% | sha3_256_B256K: 83.4% | sha3_256_B4M: 124.3%
- **kyber_ntt / claude-opus-4-7** — kyb_B1: 0.1% | kyb_B16: 1.8% | kyb_B256: 8.0%
- **kyber_ntt / gemini-3.1-pro-preview** — kyb_B1: 0.1% | kyb_B16: 0.9% | kyb_B256: 7.2%
- **kyber_ntt / gpt-5.5** — kyb_B1: 0.1% | kyb_B16: 1.3% | kyb_B256: 22.6%
- **logup_gkr / claude-opus-4-7** — gold_M4K: 1.5% | gold_M64K: 6.1% | gold_M1M: 9.6%
- **logup_gkr / gemini-3.1-pro-preview** — gold_M4K: 1.3% | gold_M64K: 4.9% | gold_M1M: 9.3%
- **logup_gkr / gpt-5.5** — gold_M4K: 1.4% | gold_M64K: 6.1% | gold_M1M: 11.5%
- **merkle_build / claude-opus-4-7** — a2_N64K: 26.4% | a2_N256K: 40.9% | a2_N1M: 48.1%
- **merkle_build / gemini-3.1-pro-preview** — a2_N64K: 29.4% | a2_N256K: 44.9% | a2_N1M: 52.0%
- **merkle_build / gpt-5.5** — a2_N64K: 31.2% | a2_N256K: 48.1% | a2_N1M: 56.6%
- **montgomery_msm / claude-opus-4-7** — bls_N4K: 1.0% | bls_N16K: 1.0% | bls_N64K: 1.1%
- **montgomery_msm / gemini-3.1-pro-preview** — bls_N4K: 0.5% | bls_N16K: 0.7% | bls_N64K: 0.8%
- **montgomery_msm / gpt-5.5** — bls_N4K: 0.6% | bls_N16K: 0.7% | bls_N64K: 0.7%
- **multilinear_sumcheck_round / claude-opus-4-7** — gold_k14_d2: 7.8% | gold_k16_d2: 14.5% | gold_k18_d2: 16.8%
- **multilinear_sumcheck_round / gemini-3.1-pro-preview** — gold_k14_d2: 4.6% | gold_k16_d2: 9.8% | gold_k18_d2: 26.8%
- **multilinear_sumcheck_round / gpt-5.5** — gold_k14_d2: 10.2% | gold_k16_d2: 14.3% | gold_k18_d2: 14.1%
- **pippenger_buckets / claude-opus-4-7** — uniform_N4K: 0.6% | uniform_N16K: 0.4% | uniform_N64K: 0.2%
- **pippenger_buckets / gemini-3.1-pro-preview** — uniform_N4K: 0.5% | uniform_N16K: 0.3% | uniform_N64K: 0.1%
- **poseidon2_hash / claude-opus-4-7** — t3_B4K: 25.7% | t3_B64K: 36.8% | t3_B1M: 50.9%
- **poseidon2_hash / gemini-3.1-pro-preview** — t3_B4K: 1.2% | t3_B64K: 5.0% | t3_B1M: 5.2%
- **poseidon2_hash / gpt-5.5** — t3_B4K: 13.7% | t3_B64K: 35.1% | t3_B1M: 60.6%
- **wots_chain / claude-opus-4-7** — w16_C64K: 133.5% | w64_C64K: 133.8% | w256_C64K: 133.8%
- **wots_chain / gemini-3.1-pro-preview** — w16_C64K: 123.4% | w64_C64K: 125.2% | w256_C64K: 124.6%
- **wots_chain / gpt-5.5** — w16_C64K: 118.5% | w64_C64K: 121.3% | w256_C64K: 123.5%

## Held-out: seed vs best (one new config per task)

_Each task is evaluated on a single held-out config never seen during the search. `Generalisation` = held-out best frac ÷ in-dist best gmean (≈1.0 means the held-out config matches in-dist quality; >1 means it transferred even better). `Speedup vs seed` = held-out best frac ÷ held-out seed frac._

| Task | Model | Held-out | Seed frac | Best frac | In-dist best (gmean) | Held-out best (abs) | Generalisation | Speedup vs seed | Notes |
|---|---|---|---|---|---|---|---|---|---|
| binius_clmul | claude-opus-4-7 | gf256_tower_N256K | 5.44% | 1.88% | 0.1968 | 21.1 Gbitops/s (u64) | 0.10× | 0.34× |  |
| binius_clmul | gemini-3.1-pro-preview | gf256_tower_N256K | 5.44% | 22.50% | 0.2928 | 253.1 Gbitops/s (u64) | 0.77× | 4.13× |  |
| binius_clmul | gpt-5.5 | gf256_tower_N256K | 5.44% | 24.42% | 0.4425 | 274.7 Gbitops/s (u64) | 0.55× | 4.49× |  |
| fri_round | claude-opus-4-7 | f4_N128K | 1.68% | 2.37% | 0.3527 | 13.3 Gmodmul/s (int64) | 0.07× | 1.41× |  |
| fri_round | gemini-3.1-pro-preview | f4_N128K | 1.68% | 1.58% | 0.2599 | 8.9 Gmodmul/s (int64) | 0.06× | 0.94× |  |
| fri_round | gpt-5.5 | f4_N128K | 1.68% | 1.58% | 0.3372 | 8.9 Gmodmul/s (int64) | 0.05× | 0.94× |  |
| goldilocks_ntt | claude-opus-4-7 | N2_20 | 136.62% | 136.63% | 0.3798 | 273.3 GB/s | 3.60× | 1.00× |  |
| goldilocks_ntt | gemini-3.1-pro-preview | N2_20 | 136.62% | 138.27% | 0.3990 | 276.5 GB/s | 3.47× | 1.01× |  |
| goldilocks_ntt | gpt-5.5 | N2_20 | 136.62% | 137.50% | 0.4050 | 275.0 GB/s | 3.40× | 1.01× |  |
| keccak_f1600_batch | claude-opus-4-7 | shake128_B1M_out256 | 3.92% | 38.60% | 0.9638 | 434.2 Gbitops/s (u64) | 0.40× | 9.85× |  |
| keccak_f1600_batch | gemini-3.1-pro-preview | shake128_B1M_out256 | 3.92% | 61.47% | 0.7844 | 691.5 Gbitops/s (u64) | 0.78× | 15.69× |  |
| keccak_f1600_batch | gpt-5.5 | shake128_B1M_out256 | 3.92% | 43.08% | 0.7010 | 484.6 Gbitops/s (u64) | 0.61× | 11.00× |  |
| kyber_ntt | claude-opus-4-7 | dil_B64 | 0.82% | 1.81% | 0.0120 | 3.6 GB/s | 1.50× | 2.21× |  |
| kyber_ntt | gemini-3.1-pro-preview | dil_B64 | 0.82% | 3.25% | 0.0072 | 6.5 GB/s | 4.53× | 3.96× |  |
| kyber_ntt | gpt-5.5 | dil_B64 | 0.82% | 3.34% | 0.0144 | 6.7 GB/s | 2.32× | 4.08× |  |
| logup_gkr | claude-opus-4-7 | bb_M256K | 0.09% | 0.43% | 0.0446 | 0.9 GB/s | 0.10× | 4.80× |  |
| logup_gkr | gemini-3.1-pro-preview | bb_M256K | 0.09% | — | 0.0390 | — | — | — | best correctness fail |
| logup_gkr | gpt-5.5 | bb_M256K | 0.09% | 2.46% | 0.0466 | 4.9 GB/s | 0.53× | 27.20× |  |
| merkle_build | claude-opus-4-7 | a4_N512K | 3.31% | 3.63% | 0.3730 | 20.4 Gmodmul/s (int64) | 0.10× | 1.10× |  |
| merkle_build | gemini-3.1-pro-preview | a4_N512K | 3.31% | 3.85% | 0.4097 | 21.6 Gmodmul/s (int64) | 0.09× | 1.16× |  |
| merkle_build | gpt-5.5 | a4_N512K | 3.31% | 3.15% | 0.4397 | 17.7 Gmodmul/s (int64) | 0.07× | 0.95× |  |
| montgomery_msm | claude-opus-4-7 | bn254_N8K | 0.03% | 0.09% | 0.0106 | 0.5 Gmodmul/s (int64) | 0.08× | 2.71× |  |
| montgomery_msm | gemini-3.1-pro-preview | bn254_N8K | 0.03% | 0.05% | 0.0070 | 0.3 Gmodmul/s (int64) | 0.08× | 1.68× |  |
| montgomery_msm | gpt-5.5 | bn254_N8K | 0.03% | 0.06% | 0.0068 | 0.3 Gmodmul/s (int64) | 0.08× | 1.74× |  |
| multilinear_sumcheck_round | claude-opus-4-7 | bb_k18_d3 | 4.36% | 3.90% | 0.1239 | 7.8 GB/s | 0.31× | 0.90× |  |
| multilinear_sumcheck_round | gemini-3.1-pro-preview | bb_k18_d3 | 4.36% | 4.03% | 0.1066 | 8.1 GB/s | 0.38× | 0.93× |  |
| multilinear_sumcheck_round | gpt-5.5 | bb_k18_d3 | 4.36% | 18.63% | 0.1272 | 37.3 GB/s | 1.46× | 4.28× |  |
| pippenger_buckets | claude-opus-4-7 | zipf15_N16K | 0.01% | 0.01% | 0.0034 | 0.0 GB/s | 0.02× | 1.18× |  |
| pippenger_buckets | gemini-3.1-pro-preview | zipf15_N16K | 0.01% | 0.01% | 0.0029 | 0.0 GB/s | 0.02× | 1.02× |  |
| poseidon2_hash | claude-opus-4-7 | t4_B256K | 3.99% | 4.25% | 0.3635 | 23.9 Gmodmul/s (int64) | 0.12× | 1.06× |  |
| poseidon2_hash | gemini-3.1-pro-preview | t4_B256K | 3.99% | 4.60% | 0.0314 | 25.9 Gmodmul/s (int64) | 1.46× | 1.15× |  |
| poseidon2_hash | gpt-5.5 | t4_B256K | 3.99% | 3.65% | 0.3076 | 20.6 Gmodmul/s (int64) | 0.12× | 0.92× |  |
| wots_chain | claude-opus-4-7 | sphincs256s_w32_C128K | 3.93% | 68.95% | 1.3370 | 775.7 Gbitops/s (u64) | 0.52× | 17.53× |  |
| wots_chain | gemini-3.1-pro-preview | sphincs256s_w32_C128K | 3.93% | 64.21% | 1.2440 | 722.3 Gbitops/s (u64) | 0.52× | 16.33× |  |
| wots_chain | gpt-5.5 | sphincs256s_w32_C128K | 3.93% | 60.63% | 1.2108 | 682.1 Gbitops/s (u64) | 0.50× | 15.42× |  |

## Cross-model comparison (in-dist best, held-out best)

| Task | claude-opus-4-7 in-dist | claude-opus-4-7 held-out | gemini-3.1-pro-preview in-dist | gemini-3.1-pro-preview held-out | gpt-5.5 in-dist | gpt-5.5 held-out |
|---|---|---|---|---|---|---|
| binius_clmul | 0.1968 | 1.88% | 0.2928 | 22.50% | 0.4425 | 24.42% |
| fri_round | 0.3527 | 2.37% | 0.2599 | 1.58% | 0.3372 | 1.58% |
| goldilocks_ntt | 0.3798 | 136.63% | 0.3990 | 138.27% | 0.4050 | 137.50% |
| keccak_f1600_batch | 0.9638 | 38.60% | 0.7844 | 61.47% | 0.7010 | 43.08% |
| kyber_ntt | 0.0120 | 1.81% | 0.0072 | 3.25% | 0.0144 | 3.34% |
| logup_gkr | 0.0446 | 0.43% | 0.0390 | FAIL | 0.0466 | 2.46% |
| merkle_build | 0.3730 | 3.63% | 0.4097 | 3.85% | 0.4397 | 3.15% |
| montgomery_msm | 0.0106 | 0.09% | 0.0070 | 0.05% | 0.0068 | 0.06% |
| multilinear_sumcheck_round | 0.1239 | 3.90% | 0.1066 | 4.03% | 0.1272 | 18.63% |
| pippenger_buckets | 0.0034 | 0.01% | 0.0029 | 0.01% | — | — |
| poseidon2_hash | 0.3635 | 4.25% | 0.0314 | 4.60% | 0.3076 | 3.65% |
| wots_chain | 1.3370 | 68.95% | 1.2440 | 64.21% | 1.2108 | 60.63% |
