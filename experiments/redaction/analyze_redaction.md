# Redaction experiment — analysis

| task                 | model                  | arm | ID× | HO× | held_ok | predicate? | model_named |
|----------------------|------------------------|-----|-----|-----|---------|------------|-------------|
| keccak_f1600_batch   | claude-opus-4-7        | dis | 9.60 | 14.03 | yes | - | - |
| keccak_f1600_batch   | claude-opus-4-7        | red | 9.74 | 15.95 | yes | - | - |
| keccak_f1600_batch   | gemini-3.1-pro-preview | dis | 11.79 | 14.30 | yes | ==\s*168\b x2; rate_bytes\s*== x3; out_bytes\s*==\s*256 x1 | - |
| keccak_f1600_batch   | gemini-3.1-pro-preview | red | 9.83 | 11.01 | yes | - | - |
| keccak_f1600_batch   | gpt-5.5                | dis | 10.27 | 10.49 | yes | rate_bytes\s*== x2; out_bytes\s*==\s*256 x1 | - |
| keccak_f1600_batch   | gpt-5.5                | red | 13.03 | 10.53 | yes | - | - |
| kyber_ntt            | claude-opus-4-7        | dis | 2.41 | ? | FAIL | - | - |
| kyber_ntt            | claude-opus-4-7        | red | 2.08 | 2.29 | yes | - | - |
| kyber_ntt            | gemini-3.1-pro-preview | dis | 3.17 | 0.54 | yes | - | - |
| kyber_ntt            | gemini-3.1-pro-preview | red | 2.36 | 2.17 | yes | - | - |
| kyber_ntt            | gpt-5.5                | dis | 1.90 | 2.31 | yes | 8380417 x25 | - |
| kyber_ntt            | gpt-5.5                | red | 2.95 | 0.68 | yes | - | - |
| wots_chain           | claude-opus-4-7        | dis | 17.23 | 17.43 | yes | n_lanes\s*==\s*4 x1 | - |
| wots_chain           | claude-opus-4-7        | red | 16.70 | 16.82 | yes | - | - |
| wots_chain           | gemini-3.1-pro-preview | dis | 16.40 | 16.33 | yes | n_lanes\s*==\s*4 x1 | - |
| wots_chain           | gemini-3.1-pro-preview | red | 20.27 | 20.32 | yes | n_lanes\s*==\s*4 x2 | - |
| wots_chain           | gpt-5.5                | dis | 21.33 | 20.90 | yes | - | - |
| wots_chain           | gpt-5.5                | red | 20.66 | 20.79 | yes | n_lanes\s*==\s*4 x1 | n_bytes=32 |

## Held-out-predicate screen (disclosed vs redacted)

- disclosed arm: 5/9 winners branch on the held-out identity
- redacted arm:  2/9 winners branch on the held-out identity
- redacted arm:  1/9 runs where the MODEL named the held-out unprompted during the search (knowledge-driven recall)

Reading: predicates that vanish under redaction are disclosure-driven enumeration; predicates (or model_named events) that persist are knowledge-driven (the model inferred the held-out from public standards — NIST PQC moduli, SHAKE128, SPHINCS+ widths). The predicate is a static screen on the winner; model_named is the stronger process signal (any iteration, code or prose). Differentially-tuned enumeration (grade C) vs a generic correctness branch (e.g. wots/Opus `n_lanes>=4`) is separated by the mechanism audit of Appendix B, not by this regex.
