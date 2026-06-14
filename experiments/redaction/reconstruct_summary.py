#!/usr/bin/env python3
"""Rebuild the terminal artifacts of an evolve run that hung mid-loop.

When a run dies after completing some iterations but before evolve writes
its end-of-run files (summary.json, history.json, best.metal,
best_result.json), the per-iteration artifacts (NN_result.json,
NN_candidate.metal, NN_prompt.md) still fully determine the outcome:
iterations that produced no NN_result.json never entered the (1+1)
comparison, so they cannot have changed the incumbent. This script replays
the promotion rule over the completed iterations and writes the four
terminal files exactly as evolve would have, plus a ``reconstructed`` block
recording how (so it is never mistaken for a clean run).

Caveats it records honestly:
  * seed_score is read from the rounded value in 01_prompt.md (evolve
    persists no 00_result.json), so ``improvement`` carries that rounding.
  * elapsed_s is estimated from file mtimes (result.json - prompt.md).

Usage:
  python reconstruct_summary.py <run_dir>            # dry: print the plan
  python reconstruct_summary.py <run_dir> --write     # write the 4 files
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# allow importing the task registry to recover the per-task denylist
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _seed_score(run: Path) -> float | None:
    m = re.search(r"score \(gmean of fraction\):\s*([0-9.]+)",
                  (run / "01_prompt.md").read_text())
    return float(m.group(1)) if m else None


def _mtime(p: Path) -> float | None:
    return p.stat().st_mtime if p.exists() else None


def _elapsed(run: Path, i: int) -> float | None:
    a, b = _mtime(run / f"{i:02d}_prompt.md"), _mtime(run / f"{i:02d}_result.json")
    return round(b - a, 1) if (a and b and b >= a) else None


def _denylist_for(task: str) -> list[str]:
    try:
        import types
        sys.modules.setdefault("Metal", types.ModuleType("Metal"))
        from metal_zk import tasks  # noqa: F401
        from metal_zk.task import get_task
        return get_task(task).spec.held_out_denylist
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--write", action="store_true",
                    help="Write the files (default: dry-run print only).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing summary.json.")
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    name = run.name
    if (run / "summary.json").exists() and not args.force:
        print(f"refusing: {run}/summary.json already exists (use --force).")
        return 1

    redacted = name.endswith("_redacted")
    base = name[:-9] if redacted else name
    for t in ("keccak_f1600_batch", "kyber_ntt", "wots_chain",
              "goldilocks_ntt", "merkle_build", "fri_round", "poseidon2_hash",
              "logup_gkr", "pippenger_buckets", "montgomery_msm",
              "binius_clmul", "multilinear_sumcheck_round", "wots_chain"):
        if base.startswith(t + "_"):
            task = t
            model = base[len(t) + 1:].rsplit("_", 2)[0]
            break
    else:
        print(f"cannot parse task/model from dir name {name!r}")
        return 1

    denylist = _denylist_for(task) if redacted else []
    seed_score = _seed_score(run)

    # Replay the (1+1) loop over completed iterations.
    history = [{
        "iteration": 0, "role": "seed", "compile_ok": True,
        "correct": seed_score is not None, "score": seed_score,
        "fail_reason": None, "source_path": str(run / "00_seed.metal"),
        "elapsed_s": None, "is_new_best": True, "model_named_held_out": [],
    }]
    best_score, best_iter = seed_score, 0
    n_done = 0
    for i in range(1, 100):
        rj = run / f"{i:02d}_result.json"
        if not rj.exists():
            # No result for i: did the model at least respond? If neither
            # prompt's response nor result exists, we've passed the last
            # completed iteration.
            if not (run / f"{i:02d}_prompt.md").exists():
                break
            continue
        n_done = i
        r = json.loads(rj.read_text())
        sc = r.get("score")
        cand = run / f"{i:02d}_candidate.metal"
        named = []
        if redacted and cand.exists():
            blob = ((run / f"{i:02d}_response.md").read_text()
                    if (run / f"{i:02d}_response.md").exists() else "")
            blob += "\n" + cand.read_text()
            named = sorted({p for p in denylist if p in blob})
        nb = (sc is not None) and (best_score is None or sc > best_score)
        if nb:
            best_score, best_iter = sc, i
        history.append({
            "iteration": i, "role": "candidate",
            "compile_ok": bool(r.get("compile_ok")),
            "correct": sc is not None, "score": sc,
            "fail_reason": r.get("fail_reason"),
            "source_path": str(cand), "elapsed_s": _elapsed(run, i),
            "is_new_best": nb, "model_named_held_out": named,
        })

    named_all = sorted({p for h in history for p in h["model_named_held_out"]})
    named_iters = [h["iteration"] for h in history if h["model_named_held_out"]]
    best_result = json.loads((run / f"{best_iter:02d}_result.json").read_text()) \
        if best_iter > 0 else None

    summary = {
        "task": task, "model": model, "chip": "Apple M1 Pro",
        "redacted": redacted,
        "model_named_held_out": named_all,
        "model_named_held_out_iterations": named_iters,
        "n_iterations": n_done,
        "seed_score": seed_score,
        "best_score": best_score,
        "improvement": (best_score / seed_score
                        if best_score and seed_score else None),
        "reconstructed": {
            "by": "experiments/redaction/reconstruct_summary.py",
            "reason": "run hung mid-loop; terminal files rebuilt from "
                      "completed per-iteration artifacts",
            "completed_iterations": n_done,
            "winner_iteration": best_iter,
            "seed_score_source": "rounded value in 01_prompt.md "
                                 "(no 00_result.json persisted)",
            "elapsed_s_source": "estimated from file mtimes",
        },
        "history": history,
    }

    print(f"task={task} model={model} redacted={redacted}")
    print(f"completed iterations: {n_done}; winner: iter {best_iter} "
          f"score {best_score}")
    print(f"seed_score (rounded): {seed_score}  improvement: "
          f"{summary['improvement']}")
    if redacted:
        print(f"model named held-out unprompted: {named_all or '(none)'} "
              f"at iters {named_iters}")
    if not args.write:
        print("\ndry-run; pass --write to create summary.json, history.json, "
              "best.metal, best_result.json")
        return 0

    (run / "history.json").write_text(json.dumps(history, indent=2))
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    if best_iter > 0:
        shutil.copyfile(run / f"{best_iter:02d}_candidate.metal",
                        run / "best.metal")
        (run / "best_result.json").write_text(json.dumps(best_result, indent=2))
    print(f"\nwrote summary.json, history.json, best.metal, best_result.json "
          f"to {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
