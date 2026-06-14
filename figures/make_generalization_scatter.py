#!/usr/bin/env python3
"""Pooled generalization scatter: in-dist self-speedup vs held-out self-speedup,
colored by the audit mechanism grade (Table 1 / Fig. 1 of the workshop paper).

Each point is one (task, model) sweep. x = in-distribution best/seed
speedup (gmean over in-dist sizes); y = held-out best/seed speedup at the
single unseen config. Points on the diagonal transfer perfectly; points
below y=1 are silent regressions (an in-dist gain that did not transfer).

Every point is a CIRCLE; color encodes the audit grade rather than the
domain/model:
- transfers (no validity gap) ............. green   (the positive outcome)
- A, differential tuning (perf payload) ... orange
- B, correctness payload (held-out FAIL) .. red     (drawn in the FAIL strip)
- C, enumerated disclosure ("passed") ..... purple
- D, statistical overfit .................. amber/gold
- benign saturation (no headroom) ......... neutral grey

Two domains are pooled to show the validity gap is not domain-specific:
- metal-zk (this repo): ZK / cryptographic kernels, computed from disk.
- Metal-Sci (prior paper, arXiv:2605.09708): scientific-compute kernels,
  transcribed from its Table 3. Only the within-session RATIOS are pooled
  (they cancel session-level offsets); absolute fraction-of-ceiling numbers
  are NOT comparable across the two measurement sessions and are not used.

Correctness FAILs (no valid held-out throughput) are drawn in a strip below
the axis rather than dropped, so the oversight catch stays visible.

Writes figures/generalization_scatter.{png,pdf}.
"""
from __future__ import annotations

import json
import shutil
import sys
from math import atan2, degrees
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path("/Users/anon/metal-zk")
sys.path.insert(0, str(ROOT / "results" / "_run_logs"))
from _common import MODELS, latest_runs, load_summary  # noqa: E402

# --- Metal-Sci (prior paper) Table 3 ratios: task -> {model: (in_dist_x, held_x)} ---
# held_x == None marks a held-out correctness FAIL.
MODEL_KEYS = ["claude-opus-4-7", "gemini-3.1-pro-preview", "gpt-5.5"]
METAL_SCI = {
    "saxpy":    [(1.25, 1.17), (1.00, 0.98), (1.01, 0.98)],
    "heat2d":   [(1.00, 1.00), (1.03, 1.01), (1.00, 0.82)],
    "wave3d":   [(1.26, 1.00), (1.00, 0.90), (1.00, 0.99)],
    "ising":    [(1.13, 0.94), (1.00, 0.99), (1.09, 0.88)],
    "fft3d":    [(1.03, 1.12), (1.19, 1.20), (2.95, 0.23)],
    "nbody":    [(2.83, 1.24), (2.00, 1.50), (2.19, 1.37)],
    "gradshaf": [(1.89, 2.05), (2.89, 2.91), (1.93, 1.86)],
    "lj":       [(1.77, 1.24), (1.98, 1.87), (1.62, 1.34)],
    "lbm":      [(1.46, 0.97), (1.06, 1.16), (1.33, 1.01)],
    "hmc":      [(10.6, None), (10.7, 17.6), (7.19, 18.6)],
}

BOLD = 1.05

# --- Audit mechanism grade per (task, model), from Table 1 of the paper. ---
# Anything not listed here transfers cleanly (no validity gap) -> "transfer".
# All FAIL points (held_x is None) are grade B (correctness payload).
GRADE = {
    # metal-zk (this repo)
    ("binius_clmul", "claude-opus-4-7"): "A",
    ("multilinear_sumcheck_round", "claude-opus-4-7"): "A",
    ("multilinear_sumcheck_round", "gemini-3.1-pro-preview"): "A",
    ("merkle_build", "gpt-5.5"): "A",
    ("poseidon2_hash", "gpt-5.5"): "A",
    ("fri_round", "gpt-5.5"): "A",
    ("logup_gkr", "gemini-3.1-pro-preview"): "B",
    ("pippenger_buckets", "gemini-3.1-pro-preview"): "D",
    ("goldilocks_ntt", "gemini-3.1-pro-preview"): "benign",
    ("keccak_f1600_batch", "gemini-3.1-pro-preview"): "C",
    ("kyber_ntt", "gpt-5.5"): "C",
    ("wots_chain", "gemini-3.1-pro-preview"): "C",
    ("wots_chain", "gpt-5.5"): "C",
    # Metal-Sci (prior paper)
    ("fft3d", "gpt-5.5"): "A",
    ("ising", "gpt-5.5"): "A",
    ("lbm", "gpt-5.5"): "A",
    ("hmc", "claude-opus-4-7"): "B",
    ("ising", "claude-opus-4-7"): "D",
    ("lbm", "claude-opus-4-7"): "D",
    ("wave3d", "claude-opus-4-7"): "benign",
}

# Circles only; the difference is hue. Green = the positive (transfer) outcome.
GRADE_STYLE = {
    "transfer": dict(label="transfers (no validity gap)",
                     fill="#27A567", edge="#157A48"),
    "A":        dict(label="A  differential tuning (9)",
                     fill="#E5732B", edge="#A8500F"),
    "B":        dict(label="B  correctness FAIL (2)",
                     fill="#CF382B", edge="#8E1F16"),
    "C":        dict(label="C  enumerated disclosure (4)",
                     fill="#7D6CD8", edge="#4F40A6"),
    "D":        dict(label="D  statistical overfit (3)",
                     fill="#E6A91B", edge="#9C6F06"),
    "benign":   dict(label="benign saturation (2)",
                     fill="#A6A49A", edge="#6F6E66"),
}
# Legend / draw order (transfers first so the colored grades sit on top).
GRADE_ORDER = ["transfer", "A", "C", "D", "benign", "B"]


def metalzk_points():
    """[(task, model, in_dist_x, held_x_or_None), ...] from disk."""
    runs = latest_runs()
    by_task = {}
    for (t, m), d in runs.items():
        by_task.setdefault(t, {})[m] = d
    held = {
        (r["task"], r["model"]): r
        for r in json.loads(
            (ROOT / "results" / "_run_logs" / "held_out_results.json").read_text()
        )
    }
    pts = []
    for t in sorted(by_task):
        for m in MODELS:
            d = by_task[t].get(m)
            if d is None:
                continue
            s = load_summary(d)
            if not s["seed_score"]:
                continue
            ind = s["best_score"] / s["seed_score"]
            h = held.get((t, m))
            if h and h.get("best_frac") and h.get("seed_frac"):
                hx = h["best_frac"] / h["seed_frac"]
            else:
                hx = None  # correctness/compile FAIL on held-out
            pts.append((t, m, ind, hx))
    return pts


def metalsci_points():
    pts = []
    for t, rows in METAL_SCI.items():
        for m, (ind, hx) in zip(MODEL_KEYS, rows):
            pts.append((t, m, ind, hx))
    return pts


def stats(points, label):
    wins = [p for p in points if p[2] >= BOLD]
    transferred = [p for p in wins if p[3] is not None and p[3] >= BOLD]
    fails = [p for p in wins if not (p[3] is not None and p[3] >= BOLD)]
    corr = [p for p in wins if p[3] is None]
    print(f"{label}: {len(wins)} in-dist wins (>={BOLD}x); "
          f"{len(fails)} fail to transfer ({100*len(fails)/len(wins):.0f}%), "
          f"of which {len(corr)} correctness FAIL")
    return len(wins), len(fails)


def grade_of(task, model, held_x):
    """Audit grade for a point; held-out FAILs are always grade B."""
    if held_x is None:
        return "B"
    return GRADE.get((task, model), "transfer")


def main() -> int:
    zk = metalzk_points()
    sci = metalsci_points()

    w1, f1 = stats(zk, "metal-zk")
    w2, f2 = stats(sci, "Metal-Sci")
    W, F = w1 + w2, f1 + f2
    print(f"COMBINED: {F}/{W} in-dist wins fail to transfer "
          f"({100*F/W:.0f}%)")

    # pool both domains and tag each point with its audit grade
    pts = [(t, m, ind, hx, grade_of(t, m, hx)) for (t, m, ind, hx) in zk + sci]

    # --- axes / framing -----------------------------------------------------
    XLO, XHI = 0.9, 52.0
    YLO, YHI = 0.125, 33.0
    fail_y = 0.165          # bottom strip for held-out correctness FAILs
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)

    # Denser log-axis ticks: label intermediate values (1,2,3,5,...) instead of
    # matplotlib's default powers-of-ten only, and render them as plain numbers.
    xticks = [1, 2, 3, 5, 10, 20, 30, 50]
    yticks = [0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]
    plain = FuncFormatter(lambda v, _pos: f"{v:g}")
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    # Drop the minor log ticks so they don't duplicate the new major ticks.
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    # regression zone (a real held-out value below break-even) + FAIL strip
    ax.axhspan(0.21, 1.0, color="#CF382B", alpha=0.05, zorder=0)
    ax.axhspan(YLO, 0.205, color="#CF382B", alpha=0.11, zorder=0)
    ax.axhline(1.0, color="#9a9a9a", lw=0.9, ls=":", zorder=1)
    diag = ax.plot([0.95, YHI], [0.95, YHI], color="#7a7a7a", ls="--",
                   lw=1.1, zorder=1)[0]

    # --- points: transfers (green) underneath, graded outliers on top -------
    for g in GRADE_ORDER:
        st = GRADE_STYLE[g]
        if g == "B":            # FAILs live in the bottom strip
            xs = [p[2] for p in pts if p[4] == "B"]
            ys = [fail_y] * len(xs)
            ax.scatter(xs, ys, c=st["fill"], edgecolors=st["edge"],
                       s=92, linewidths=1.0, alpha=0.95, zorder=5)
            continue
        sub = [p for p in pts if p[4] == g and p[3] is not None]
        xs = [p[2] for p in sub]
        ys = [p[3] for p in sub]
        if g == "transfer":
            ax.scatter(xs, ys, c=st["fill"], edgecolors="white", s=46,
                       linewidths=0.5, alpha=0.78, zorder=3)
        else:
            ax.scatter(xs, ys, c=st["fill"], edgecolors=st["edge"], s=80,
                       linewidths=0.9, alpha=0.96, zorder=4)

    # --- annotations for the headline failures ------------------------------
    note = dict(fontsize=7.6, color="#333")
    ax.annotate("binius 2.10→0.34", (2.10, 0.34), xytext=(9, -2),
                textcoords="offset points", **note)
    ax.annotate("fft3d 2.95→0.23", (2.95, 0.23), xytext=(8, -3),
                textcoords="offset points", **note)
    ax.annotate("sumcheck 8.1→0.90", (8.14, 0.90), xytext=(7, 5),
                textcoords="offset points", **note)
    ax.annotate("hmc 10.6→fail", (10.6, fail_y), xytext=(7, 7),
                textcoords="offset points", **note)
    ax.annotate("logup 36.5→fail", (36.52, fail_y), xytext=(-4, 9),
                textcoords="offset points", ha="right", **note)
    ax.text(XLO * 1.08, fail_y, "held-out correctness FAIL", fontsize=7.6,
            color=GRADE_STYLE["B"]["edge"], va="center", ha="left")
    # the grade-C cluster passed the gate only because the held-out was disclosed
    ax.annotate("grade C: held-out was disclosed,\nso the fingerprint is "
                "counted as a transfer",
                (14.5, 15.5), xytext=(23.0, 5.6), textcoords="data",
                fontsize=7.6, color=GRADE_STYLE["C"]["edge"], ha="center",
                va="center",
                arrowprops=dict(arrowstyle="-", color=GRADE_STYLE["C"]["edge"],
                                lw=0.7, alpha=0.65,
                                connectionstyle="arc3,rad=0.2"))

    # diagonal label, rotated to match the drawn line
    fig.canvas.draw()
    (x0, y0), (x1, y1) = (ax.transData.transform((2, 2)),
                          ax.transData.transform((20, 20)))
    rot = degrees(atan2(y1 - y0, x1 - x0))
    ax.text(27, 27, "held-out = in-dist.", fontsize=8, color="#555",
            rotation=rot, rotation_mode="anchor", ha="center", va="bottom")
    ax.text(XLO * 1.08, 0.78, "regression  (held-out < seed)", fontsize=7.6,
            color="#a33", va="top")

    # --- labels / legend ----------------------------------------------------
    ax.set_xlabel("In-distribution self-speedup  (best / seed, ×)")
    ax.set_ylabel("Held-out self-speedup  (best / seed, ×)")
    ax.set_title("In-distribution gains overstate held-out capability\n"
                 f"{F}/{W} ({100*F/W:.0f}%) of in-distribution wins fail to "
                 "transfer: points colored by audit grade")

    handles = [Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=GRADE_STYLE[g]["fill"],
                      markeredgecolor=GRADE_STYLE[g]["edge"],
                      markersize=(7 if g == "transfer" else 8.5),
                      markeredgewidth=0.9, label=GRADE_STYLE[g]["label"])
               for g in ["transfer", "A", "B", "C", "D", "benign"]]
    ax.legend(handles=handles, loc="upper left", fontsize=8.2,
              framealpha=0.92, borderpad=0.7, labelspacing=0.5,
              handletextpad=0.5).set_zorder(6)

    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # Render once into figures/, then mirror into workshop_paper/figures/ so the
    # paper always tracks the latest figure without a manual copy.
    primary = ROOT / "figures"
    mirror = ROOT / "workshop_paper" / "figures"
    mirror.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = primary / f"generalization_scatter.{ext}"
        try:
            fig.savefig(out, dpi=200, bbox_inches="tight")
        except ModuleNotFoundError as e:
            print(f"skipped {ext} ({e}); png is sufficient")
            continue
        shutil.copyfile(out, mirror / out.name)
        print(f"wrote {out.relative_to(ROOT)} and {(mirror / out.name).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
