#!/usr/bin/env python3
"""Timing-noise figure for Appendix C (measurement protocol paragraph).

Two panels from figures/timing_noise_data.json (produced by
figures/measure_timing_noise.py, which re-runs the production timing
path on four seed kernels spanning the int64-mul, bitop, and DRAM-BW
roofline anchors):

  (a) The R=8 repeated production scores per task, normalized to the
      task's median score, against the 1.05x win threshold. Three
      compute-bound tasks collapse onto 1.0 (CV <= 0.25%); the
      Goldilocks NTT --- whose in-distribution lengths are all small and
      SLC-resident --- spreads to ~12% and straddles the threshold.
  (b) Single-rep CV per configuration vs. that configuration's median
      kernel runtime: noise is a function of runtime, not task ---
      sub-millisecond dispatches are timer/cache-residency limited,
      everything >= ~5 ms sits below 2%.

Writes figures/timing_noise.{png,pdf} and copies the pdf+png into
workshop_paper/figures/.
"""
from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("/Users/anon/metal-zk")
DATA = ROOT / "figures" / "timing_noise_data.json"

TASK_STYLE = {
    # task -> (display label, color, marker)
    "goldilocks_ntt":     ("goldilocks_ntt",  "#d95f02", "o"),
    "poseidon2_hash":     ("poseidon2_hash",  "#1b9e77", "s"),
    "keccak_f1600_batch": ("keccak_f1600",    "#7570b3", "D"),
    "montgomery_msm":     ("montgomery_msm",  "#666666", "^"),
}
WIN = 1.05


def main() -> int:
    data = json.loads(DATA.read_text())
    tasks = data["tasks"]

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 3.5))

    # --- Panel (a): repeated production scores vs the 1.05x threshold ---
    names = [n for n in TASK_STYLE if n in tasks]
    all_norm: list[float] = []
    for i, name in enumerate(names):
        label, color, marker = TASK_STYLE[name]
        scores = tasks[name]["scores"]
        med = statistics.median(scores)
        norm = [s / med for s in scores]
        all_norm.extend(norm)
        # deterministic horizontal jitter so R points don't overprint
        xs = [i + (j - (len(norm) - 1) / 2) * 0.07 for j in range(len(norm))]
        axA.scatter(xs, norm, s=26, color=color, marker=marker,
                    alpha=0.75, linewidths=0.6, edgecolors="white",
                    zorder=3)
        c = statistics.pstdev(scores) / statistics.mean(scores)
        axA.text(i, 0.035, f"CV {c*100:.2f}%",
                 transform=axA.get_xaxis_transform(),
                 ha="center", fontsize=7.6, color=color)

    axA.axhline(WIN, color="#b22222", lw=1.0, ls="--", zorder=2)
    axA.axhline(1 / WIN, color="#b22222", lw=1.0, ls="--", zorder=2)
    axA.text(len(names) - 0.42, WIN * 1.012, r"win threshold $1.05\times$",
             fontsize=7.6, color="#b22222", ha="right", va="bottom")
    axA.axhline(1.0, color="#999", lw=0.7, zorder=1)
    axA.set_yscale("log")
    ylo = min(min(all_norm) * 0.93, 0.90)
    yhi = max(max(all_norm) * 1.07, 1.12)
    axA.set_ylim(ylo, yhi)
    yticks = [t for t in (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2,
                          1.3, 1.4)
              if ylo <= t <= yhi]
    axA.set_yticks(yticks)
    axA.set_yticklabels([f"{t:g}" for t in yticks])
    axA.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axA.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axA.set_xticks(range(len(names)))
    axA.set_xticklabels([TASK_STYLE[n][0] for n in names],
                        fontsize=8, rotation=12)
    axA.set_xlim(-0.5, len(names) - 0.5)
    axA.set_ylabel(r"score $S_{\mathcal{T}}$ / task median")
    axA.set_title(f"(a) {data['r_score']} repeated measurements of the "
                  "scored quantity", fontsize=9.5)

    # --- Panel (b): single-rep CV vs configuration runtime ---
    for name in names:
        label, color, marker = TASK_STYLE[name]
        for cfg in tasks[name]["configs"]:
            t_ms = statistics.median(cfg["gpu_seconds"]) * 1e3
            cv_pct = max(cfg["cv"] * 100, 1e-3)
            face = "none" if cfg["held_out"] else color
            axB.scatter([t_ms], [cv_pct], s=30, marker=marker,
                        facecolors=face, edgecolors=color,
                        linewidths=1.0, alpha=0.85, zorder=3)

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.axhline(5.0, color="#b22222", lw=1.0, ls="--", zorder=2)
    axB.text(axB.get_xlim()[0] * 1.6, 5.0 * 1.15,
             r"win-threshold margin (5%)", fontsize=7.6, color="#b22222",
             va="bottom")
    axB.set_xlabel("configuration median GPU time (ms)")
    axB.set_ylabel("single-rep CV (%)")
    axB.set_title("(b) noise vs. kernel runtime, per configuration",
                  fontsize=9.5)

    handles = [
        Line2D([], [], marker=TASK_STYLE[n][2], linestyle="none",
               markerfacecolor=TASK_STYLE[n][1],
               markeredgecolor=TASK_STYLE[n][1], markersize=5.5,
               label=TASK_STYLE[n][0])
        for n in names
    ]
    handles.append(Line2D([], [], marker="o", linestyle="none",
                          markerfacecolor="none", markeredgecolor="#444",
                          markersize=5.5, label="held-out config"))
    axB.legend(handles=handles, loc="upper right", fontsize=7.2,
               framealpha=0.9, borderpad=0.5, handletextpad=0.3)

    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = ROOT / "figures" / f"timing_noise.{ext}"
        try:
            fig.savefig(out, dpi=200, bbox_inches="tight")
        except ModuleNotFoundError as e:
            print(f"skipped {ext} ({e})")
            continue
        print(f"wrote {out.relative_to(ROOT)}")
        dst = ROOT / "workshop_paper" / "figures" / f"timing_noise.{ext}"
        shutil.copy(out, dst)
        print(f"wrote {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
