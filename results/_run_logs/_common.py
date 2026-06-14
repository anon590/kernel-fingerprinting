"""Shared helpers for metal-zk result analysis scripts.

Run directories follow the pattern::

    <task>_<model>_<YYYYMMDD>_<HHMMSS>

Task names may contain underscores (e.g. ``goldilocks_ntt``); model names
do not (they use ``-`` and ``.``). We parse by splitting off the trailing
date/time tokens, then the model, then keeping the remainder as the task.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/anon/metal-zk/results")

# Models under analysis. Runs for any other model (e.g. exploratory
# gemini-3.5-flash sweeps) are ignored so the report and held-out eval
# stay scoped to the headline three-way comparison. Pass ``models=None``
# to ``latest_runs`` to include every model found on disk.
MODELS = (
    "claude-opus-4-7",
    "gemini-3.1-pro-preview",
    "gpt-5.5",
)


def parse_run_dir(name: str) -> tuple[str, str, str] | None:
    """Return (task, model, stamp) or None if the name doesn't match."""
    parts = name.split("_")
    if len(parts) < 4:
        return None
    date, hms = parts[-2], parts[-1]
    if not (len(date) == 8 and date.isdigit() and len(hms) == 6 and hms.isdigit()):
        return None
    model = parts[-3]
    task = "_".join(parts[:-3])
    if not task:
        return None
    return task, model, f"{date}_{hms}"


def latest_runs(
    models: tuple[str, ...] | None = MODELS,
) -> dict[tuple[str, str], Path]:
    """Map (task, model) -> latest run directory with a usable summary.

    Only models in ``models`` are kept (pass ``None`` to include every
    model found on disk). A run with no ``summary.json`` is treated as
    incomplete (e.g. a crashed sweep) and skipped, so the latest *usable*
    run is selected rather than the latest directory by timestamp.
    """
    latest: dict[tuple[str, str], tuple[str, Path]] = {}
    for d in ROOT.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        parsed = parse_run_dir(d.name)
        if parsed is None:
            continue
        task, model, stamp = parsed
        if models is not None and model not in models:
            continue
        if not (d / "summary.json").exists():
            continue
        prev = latest.get((task, model))
        if prev is None or stamp > prev[0]:
            latest[(task, model)] = (stamp, d)
    return {k: v[1] for k, v in latest.items()}


def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def load_best_result(run_dir: Path) -> dict:
    return json.loads((run_dir / "best_result.json").read_text())


def best_iter_from_history(summary: dict) -> int:
    """Iteration index that produced ``best_score`` (0 if it was the seed)."""
    best_score = summary["best_score"]
    for h in summary["history"]:
        if h["role"] == "candidate" and h.get("score") == best_score:
            return h["iteration"]
    return 0


def candidate_stats(summary: dict) -> dict:
    hist = summary["history"]
    cands = [h for h in hist if h["role"] == "candidate"]
    skipped = [h for h in hist if h["role"] == "skipped"]
    return {
        "n_candidates": len(cands),
        "n_skipped": len(skipped),
        "compile_fails": sum(1 for h in cands if not h["compile_ok"]),
        "correct_fails": sum(
            1 for h in cands if h["compile_ok"] and not h["correct"]
        ),
        "wall_seconds": sum(h.get("elapsed_s", 0.0) for h in hist),
    }
