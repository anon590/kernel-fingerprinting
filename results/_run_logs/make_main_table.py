#!/usr/bin/env python3
"""Emit the main results table (Metal-Sci Table 3 style) for metal-zk.

Pulls the same numbers used by ``analyze.py``:
- In-dist. x  = best/seed gmean self-speedup, from ``summary.json``.
- Held-out frac-of-ceiling = achieved/ceiling at the unseen size, from
  ``held_out_results.json`` (run ``eval_held_out.py`` first).
- Held-out x  = held-out best frac / held-out seed frac.

Bold marks meaningful speedups (>= 1.05x). Red marks a silent regression
(an in-dist gain that inverts to a severe held-out slowdown, < 0.5x) or a
held-out correctness FAIL. The per-task ``Outcome`` column is authored by
hand below; everything numeric is recomputed from disk.

Writes ``main_results_table.tex`` and prints a Markdown preview to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MODELS, ROOT, latest_runs, load_summary

MODEL_LABELS = {
    "claude-opus-4-7": "Opus",
    "gemini-3.1-pro-preview": "Gemini",
    "gpt-5.5": "GPT",
}

# Terse, paper-style verdict per task (held-out generalisation story).
OUTCOMES = {
    "binius_clmul": ("Opus silent regression; Gemini, GPT generalize",
                     "Opus silent regression; Gemini, GPT generalize"),
    "fri_round": ("Opus generalizes; Gemini, GPT flat",
                  "Opus generalizes; Gemini, GPT flat"),
    "goldilocks_ntt": (r"saturated (seed $\ge$ roofline)",
                       "saturated (seed >= roofline)"),
    "keccak_f1600_batch": ("generalizes", "generalizes"),
    "kyber_ntt": ("generalizes (far from ceiling)",
                  "generalizes (far from ceiling)"),
    "logup_gkr": ("Gemini wrong at held-out; Opus, GPT generalize",
                  "Gemini wrong at held-out; Opus, GPT generalize"),
    "merkle_build": (r"overfits (arity $2\!\to\!4$)",
                     "overfits (arity 2->4)"),
    "montgomery_msm": ("generalizes (far from ceiling)",
                       "generalizes (far from ceiling)"),
    "multilinear_sumcheck_round": ("GPT generalizes; Opus, Gemini overfit",
                                    "GPT generalizes; Opus, Gemini overfit"),
    "pippenger_buckets": ("overfits; GPT run n/a", "overfits; GPT run n/a"),
    "poseidon2_hash": (r"overfits (width $3\!\to\!4$)",
                       "overfits (width 3->4)"),
    "wots_chain": ("generalizes", "generalizes"),
}

BOLD = 1.05      # meaningful speedup threshold
REGRESS = 0.50   # held-out x below this (with an in-dist gain) is a red flag


def _x_str(v: float) -> str:
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def _frac_str(p: float) -> str:
    # p is a percentage already (0..>100)
    if p >= 10:
        return f"{p:.0f}"
    if p >= 1:
        return f"{p:.1f}"
    return f"{p:.2f}"


def _cell(value: str, *, bold: bool, red: bool, tex: bool) -> str:
    if tex:
        if red:
            value = rf"\textcolor{{red}}{{{value}}}"
        if bold:
            value = rf"\textbf{{{value}}}"
        return value
    if red:
        value = f"_{value}_"   # markdown italic flags the regression/FAIL
    if bold:
        value = f"**{value}**"
    return value


def build_rows():
    runs = latest_runs()
    by_task: dict[str, dict[str, Path]] = {}
    for (t, m), d in runs.items():
        by_task.setdefault(t, {})[m] = d
    held = {
        (r["task"], r["model"]): r
        for r in json.loads(
            (ROOT / "_run_logs" / "held_out_results.json").read_text()
        )
    }

    rows = []
    for task in sorted(by_task):
        ind, hf, hx = {}, {}, {}
        for m in MODELS:
            d = by_task[task].get(m)
            if d is None:
                ind[m] = hf[m] = hx[m] = None
                continue
            s = load_summary(d)
            ind[m] = (s["best_score"] / s["seed_score"]) if s["seed_score"] else None
            h = held.get((task, m))
            if h and h.get("best_frac") is not None:
                hf[m] = h["best_frac"] * 100.0
                hx[m] = (
                    h["best_frac"] / h["seed_frac"] if h["seed_frac"] else None
                )
            else:
                # run exists but held-out failed (compile/correctness)
                hf[m] = "FAIL" if h is not None else None
                hx[m] = "FAIL" if h is not None else None
        rows.append((task, ind, hf, hx))
    return rows


def render(tex: bool) -> str:
    rows = build_rows()
    out = []
    for task, ind, hf, hx in rows:
        cells = [task]
        # in-dist x
        for m in MODELS:
            v = ind[m]
            if v is None:
                cells.append("--" if tex else "—")
            else:
                cells.append(_cell(_x_str(v), bold=v >= BOLD, red=False, tex=tex))
        # held-out frac-of-ceiling (%)
        for m in MODELS:
            p = hf[m]
            if p is None:
                cells.append("--" if tex else "—")
            elif p == "FAIL":
                cells.append("--" if tex else "—")
            else:
                suffix = r"\%" if tex else "%"
                cells.append(_frac_str(p) + suffix)
        # held-out x
        for m in MODELS:
            v = hx[m]
            if v is None:
                cells.append("--" if tex else "—")
            elif v == "FAIL":
                cells.append(_cell("FAIL", bold=False, red=True, tex=tex))
            else:
                red = (ind[m] is not None and ind[m] >= BOLD and v < REGRESS)
                cells.append(_cell(_x_str(v), bold=v >= BOLD, red=red, tex=tex))
        # outcome
        cells.append(OUTCOMES[task][0 if tex else 1])
        out.append(cells)
    return out


def to_latex() -> str:
    rows = render(tex=True)
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor} or \usepackage{xcolor}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evolutionary kernel refinement sweeps on Apple M1 Pro for"
        r" \textsc{Metal-ZK} (Opus = claude-opus-4-7, Gemini ="
        r" gemini-3.1-pro-preview, GPT = gpt-5.5). \emph{In-dist.}\ $\times$ ="
        r" best\,/\,seed, gmean over three in-distribution size configurations."
        r" \emph{Held-out frac-of-ceiling} = achieved\,/\,ceiling at the unseen"
        r" size, measured in a single fresh session for all three models;"
        r" \emph{held-out} $\times$ = best\,/\,seed at that size config."
        r" \textbf{Bold} marks meaningful improvements ($\ge 1.05\times$);"
        r" \textcolor{red}{red} marks a silent regression (an in-distribution"
        r" gain that inverts to a held-out slowdown) or a held-out correctness"
        r" failure. GPT's \texttt{pippenger\_buckets} run did not complete.}",
        r"\label{tab:metalzk-main}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l ccc ccc ccc l}",
        r"\toprule",
        r"& \multicolumn{3}{c}{In-dist.\ $\times$}"
        r" & \multicolumn{3}{c}{Held-out frac-of-ceiling}"
        r" & \multicolumn{3}{c}{Held-out $\times$} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Task & Opus & Gemini & GPT & Opus & Gemini & GPT"
        r" & Opus & Gemini & GPT & Outcome \\",
        r"\midrule",
    ]
    for cells in rows:
        task = cells[0].replace("_", r"\_")
        body = [rf"\texttt{{{task}}}"] + cells[1:]
        lines.append(" & ".join(body) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def to_markdown() -> str:
    rows = render(tex=False)
    head1 = ("| Task | In-dist × Opus | Gemini | GPT "
             "| Held-out frac Opus | Gemini | GPT "
             "| Held-out × Opus | Gemini | GPT | Outcome |")
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines = [head1, sep]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    tex = to_latex()
    out_path = ROOT / "_run_logs" / "main_results_table.tex"
    out_path.write_text(tex + "\n")
    print(to_markdown())
    print(f"\nLaTeX written to: {out_path.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
