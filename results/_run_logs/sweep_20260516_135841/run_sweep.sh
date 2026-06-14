#!/usr/bin/env bash
# Sequential model-sweep runner. Each command runs only after the previous
# finishes (regardless of its rc) so a single LLM hiccup doesn't drop the
# remaining tasks on the floor.
set -u
cd /Users/anon/metal-zk

LOG_DIR="$(dirname "$0")"
SWEEP="$LOG_DIR/sweep.log"

run() {
  local tag="$1"; shift
  local log="$LOG_DIR/${tag}.log"
  echo "=== STARTING ${tag} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SWEEP"
  echo "    cmd: $*" >> "$SWEEP"
  local t0=$(date +%s)
  ( "$@" ) >"$log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "=== FINISHED ${tag} rc=${rc} dt=${dt}s $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SWEEP"
}

run 01_gemini_keccak_f1600_batch  uv run run_benchmark.py --task keccak_f1600_batch --model gemini-3.1-pro-preview --iterations 15
run 02_gemini_merkle_build        uv run run_benchmark.py --task merkle_build       --model gemini-3.1-pro-preview --iterations 10
run 03_gemini_montgomery_msm      uv run run_benchmark.py --task montgomery_msm     --model gemini-3.1-pro-preview --iterations 10
run 04_gemini_wots_chain          uv run run_benchmark.py --task wots_chain         --model gemini-3.1-pro-preview --iterations 10
run 05_gpt55_goldilocks_ntt       uv run run_benchmark.py --task goldilocks_ntt     --model gpt-5.5                --iterations 10
run 06_gpt55_keccak_f1600_batch   uv run run_benchmark.py --task keccak_f1600_batch --model gpt-5.5                --iterations 15
run 07_gpt55_merkle_build         uv run run_benchmark.py --task merkle_build       --model gpt-5.5                --iterations 10
run 08_gpt55_poseidon2_hash       uv run run_benchmark.py --task poseidon2_hash     --model gpt-5.5                --iterations 10

echo "=== SWEEP COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SWEEP"
