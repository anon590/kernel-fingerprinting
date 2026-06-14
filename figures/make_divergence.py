#!/usr/bin/env python3
"""The divergence trajectory (main-text Fig.).

For three exemplar (task, model) cells, plot the (1+1) incumbent lineage
over iterations with two signals:
  - in-distribution score the loop actually saw (blue), and
  - held-out score evaluated retroactively on each iteration's incumbent
    (red), post-hoc now that the search is over.

Both are normalised to the seed (y-axis = x seed, log). The two signals
co-move until the fingerprint enters, then split -- the "scissor": the
loop keeps promoting on the rising blue signal while the red signal turns
down into the regression zone (or FAILs bit-exact). Selection pressure,
not intent, is the strategist.

Reuses the red regression-zone / FAIL-strip grammar of
make_generalization_scatter.py. Reads figures/divergence_data.json
(produced by extract_divergence.py). Writes figures/divergence.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/Users/anon/metal-zk")

IN_COLOR = "#2B6CB0"     # in-distribution: the signal the loop optimised
IN_EDGE = "#1B4A80"
HO_COLOR = "#CF382B"     # held-out: the retroactive truth (paper's grade-B red)
HO_EDGE = "#8E1F16"
SEED_GREY = "#8a8a8a"

YLO, YHI = 0.22, 52.0
FAIL_TOP = 0.30          # below this: held-out correctness FAIL strip
FAIL_Y = 0.255           # where FAIL markers sit

# Per-cell fingerprint callout: text box + arrow to the divergence event.
ANNOT = {
    "multilinear_sumcheck_round": dict(
        text="iter 4:  if (d==2) fast path\nheld-out (d=3) takes generic else",
        xy_text=(4.6, 19.0), target=(5, 0.88)),
    "logup_gkr": dict(
        text="iter 3:  wrong Barrett constant\nheld-out FAILs bit-exact, persists",
        xy_text=(5.3, 7.4), target=(3, FAIL_Y)),
    "binius_clmul": dict(
        text="iter 5:  windowed-scan rewrite\nno predicate — register spill on tower",
        xy_text=(5.0, 13.5), target=(5, 0.35)),
}


def main() -> int:
    data = json.loads((ROOT / "figures" / "divergence_data.json").read_text())
    runs = data["runs"]
    n = len(runs)

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(
        1, n, figsize=(4.05 * n, 4.35), sharey=True,
    )
    if n == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        its = run["iters"]
        xs = [it["iter"] for it in its]
        y_in = [it["in_dist_ratio"] for it in its]
        y_ho = [it["held_out_ratio"] for it in its]          # None on FAIL
        fp = run["fingerprint_iter"]

        # --- background zones (paper grammar) ------------------------------
        ax.axhspan(FAIL_TOP, 1.0, color=HO_COLOR, alpha=0.05, zorder=0)
        ax.axhspan(YLO, FAIL_TOP, color=HO_COLOR, alpha=0.11, zorder=0)
        ax.axhline(1.0, color=SEED_GREY, lw=1.0, ls=":", zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(-0.35, 10.35)
        ax.set_ylim(YLO, YHI)

        # fingerprint onset marker
        ax.axvline(fp, color="#555", lw=1.1, ls="--", alpha=0.7, zorder=1)

        # --- the scissor wedge: gap between the two signals ----------------
        y_ho_fill = [(v if v is not None else FAIL_Y) for v in y_ho]
        ax.fill_between(xs, y_ho_fill, y_in, color=HO_COLOR, alpha=0.10,
                        zorder=1, linewidth=0)

        # --- in-distribution signal (what the loop saw) --------------------
        ax.plot(xs, y_in, "-", color=IN_COLOR, lw=2.3, zorder=4)
        ax.plot(xs, y_in, "o", color=IN_COLOR, mec=IN_EDGE, mew=0.8,
                ms=6.2, zorder=5)

        # --- held-out signal (retroactive truth) ---------------------------
        valid = [(x, v) for x, v in zip(xs, y_ho) if v is not None]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, "-", color=HO_COLOR, lw=2.3, zorder=4)
            ax.plot(vx, vy, "s", color=HO_COLOR, mec=HO_EDGE, mew=0.8,
                    ms=6.0, zorder=5)
        # FAIL run (e.g. logup from iter 3 on): drop to the strip + X markers
        fail = [x for x, v in zip(xs, y_ho) if v is None]
        if fail:
            last_ok = valid[-1] if valid else (fail[0] - 1, 1.0)
            ax.plot([last_ok[0], fail[0]], [last_ok[1], FAIL_Y], ":",
                    color=HO_COLOR, lw=1.6, zorder=4)
            ax.plot(fail, [FAIL_Y] * len(fail), ":", color=HO_COLOR,
                    lw=1.6, zorder=4)
            ax.plot(fail, [FAIL_Y] * len(fail), "X", color=HO_COLOR,
                    mec=HO_EDGE, mew=0.8, ms=8.0, zorder=5)

        # --- end-of-run value labels ---------------------------------------
        ax.annotate(f"{y_in[-1]:.1f}×", (xs[-1], y_in[-1]),
                    xytext=(6, 3), textcoords="offset points",
                    fontsize=9.5, color=IN_EDGE, fontweight="bold",
                    ha="left", va="bottom")
        if y_ho[-1] is None:
            ax.annotate("FAIL", (xs[-1], FAIL_Y), xytext=(6, 0),
                        textcoords="offset points", fontsize=9, color=HO_EDGE,
                        fontweight="bold", ha="left", va="center")
        else:
            ax.annotate(f"{y_ho[-1]:.2f}×", (xs[-1], y_ho[-1]),
                        xytext=(6, -1), textcoords="offset points",
                        fontsize=9.5, color=HO_EDGE, fontweight="bold",
                        ha="left", va="top")

        # --- fingerprint callout (box + arrow to the divergence event) -----
        an = ANNOT.get(run["task"])
        if an:
            ax.annotate(
                an["text"], xy=an["target"], xytext=an["xy_text"],
                fontsize=8.2, color="#222", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.34", fc="white",
                          ec="#bbbbbb", lw=0.8, alpha=0.92),
                arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.1,
                                connectionstyle="arc3,rad=-0.18"),
                zorder=6)

        ax.set_title(run["label"], pad=8, fontweight="bold")
        ax.set_xlabel("iteration")
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.grid(True, which="major", axis="y", ls=":", lw=0.4, alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("incumbent score  (× seed)")
    # explicit "x seed" ticks (shared axis)
    yt = [0.3, 0.5, 1, 2, 5, 10, 20, 50]
    axes[0].set_yticks(yt)
    axes[0].set_yticklabels([f"{t:g}×" for t in yt])
    axes[0].minorticks_off()

    # zone definitions on the leftmost panel
    axes[0].text(0.15, 0.62, "regression zone\n(held-out < seed)", fontsize=7.4,
                 color=HO_EDGE, va="center", ha="left", alpha=0.9)
    axes[0].text(0.15, FAIL_Y, "held-out FAIL zone", fontsize=7.4,
                 color=HO_EDGE, va="center", ha="left", alpha=0.9)

    # shared legend
    handles = [
        Line2D([0], [0], color=IN_COLOR, marker="o", mec=IN_EDGE, lw=2.3,
               ms=6.5, label="in-distribution score (what the (1+1) loop saw)"),
        Line2D([0], [0], color=HO_COLOR, marker="s", mec=HO_EDGE, lw=2.3,
               ms=6.0, label="held-out score (retroactive, post-hoc)"),
        Line2D([0], [0], color="#555", lw=1.1, ls="--",
               label="fingerprint enters the incumbent"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8.7,
               frameon=False, bbox_to_anchor=(0.5, 1.012),
               handletextpad=0.5, columnspacing=1.8)

    fig.suptitle(
        "The divergence trajectory: in-distribution and held-out signals "
        "co-move — until the fingerprint enters, then split",
        y=1.10, fontsize=11.5, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "pdf"):
        out = ROOT / "figures" / f"divergence.{ext}"
        try:
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"wrote {out.relative_to(ROOT)}")
        except ModuleNotFoundError as e:
            print(f"skipped {ext} ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
