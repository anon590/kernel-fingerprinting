#!/usr/bin/env bash
# Controlled grade-C experiment: the 3 disclosed tasks x 3 models x 2 arms
# (disclosed / redacted). The disclosed arm reproduces the runs already in
# the paper; the redacted arm is the new causal control. See README.md.
#
# A (task, model, arm) cell is considered FINISHED if a matching run
# directory contains summary.json (evolve writes it only at the very end).
# Finished cells are skipped so an interrupted sweep can be resumed without
# re-spending budget; pass --force to re-run them anyway. Partially-complete
# dirs (no summary.json) do not count as finished and are left in place --- a
# re-run starts a fresh timestamped directory.
#
# Usage:
#   ./run_redaction_sweep.sh            # dry-run: show run/skip plan, run nothing
#   ./run_redaction_sweep.sh --go       # run the unfinished cells
#   ./run_redaction_sweep.sh --go --force   # re-run every cell, finished or not
#   ITER=12 MODELS="gpt-5.5" ./run_redaction_sweep.sh --go   # override scope
#
# Run only the new control arm (disclosed arm is reusable from the paper):
#   ARMS="redacted" ./run_redaction_sweep.sh --go
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$REPO/results/redaction_experiment}"
ITER="${ITER:-12}"
TASKS="${TASKS:-keccak_f1600_batch kyber_ntt wots_chain}"
MODELS="${MODELS:-claude-opus-4-7 gemini-3.1-pro-preview gpt-5.5}"
ARMS="${ARMS:-disclosed redacted}"

GO=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --go)    GO=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown flag: $a (use --go and/or --force)" >&2; exit 2 ;;
  esac
done

# Echo the finished run dir for (task, model, arm), else empty. "Finished"
# = a matching directory (correct arm) containing summary.json.
finished_run() {
  local task="$1" model="$2" arm="$3" d base
  [[ -d "$OUT" ]] || return 0
  for d in "$OUT/${task}_${model}_"*; do
    [[ -d "$d" ]] || continue
    base="$(basename "$d")"
    if [[ "$arm" == "redacted" ]]; then
      [[ "$base" == *_redacted ]] || continue
    else
      [[ "$base" == *_redacted ]] && continue
    fi
    if [[ -f "$d/summary.json" ]]; then
      echo "$base"
      return 0
    fi
  done
}

n=0; planned=0; skipped=0
for task in $TASKS; do
  for model in $MODELS; do
    for arm in $ARMS; do
      n=$((n + 1))
      flag=""
      [[ "$arm" == "redacted" ]] && flag="--redact-held-out"
      done_dir="$(finished_run "$task" "$model" "$arm")"

      if [[ -n "$done_dir" && "$FORCE" == "0" ]]; then
        skipped=$((skipped + 1))
        printf '[%2d] SKIP  %-20s %-24s %-9s (finished: %s)\n' \
          "$n" "$task" "$model" "$arm" "$done_dir"
        continue
      fi

      planned=$((planned + 1))
      tag="RUN  "
      [[ -n "$done_dir" ]] && tag="FORCE"
      cmd=(python "$REPO/run_benchmark.py"
           --task "$task" --model "$model"
           --iterations "$ITER" --output-dir "$OUT" $flag)
      printf '[%2d] %s %-20s %-24s %-9s\n     %s\n' \
        "$n" "$tag" "$task" "$model" "$arm" "${cmd[*]}"
      if [[ "$GO" == "1" ]]; then
        "${cmd[@]}"
      fi
    done
  done
done

echo
echo "Plan: $planned to run, $skipped skipped (finished), $n total cells."
[[ "$FORCE" == "1" ]] && echo "  (--force: finished cells re-run)"
if [[ "$GO" == "0" ]]; then
  echo "Dry run: nothing executed. Re-run with --go to launch. Output dir: $OUT"
else
  echo "Sweep done. Next:"
  echo "  python $REPO/experiments/redaction/eval_held_out_redaction.py --go"
  echo "  python $REPO/experiments/redaction/analyze_redaction.py"
fi
