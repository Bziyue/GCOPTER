#!/usr/bin/env python3
"""Plot paired core-volume, generation-time and FIRI changes from validate_core.py."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing paired.csv")
    args = parser.parse_args()
    with (args.results / "paired.csv").open() as file:
        rows = [r for r in csv.DictReader(file) if r.get("core_pct")]
    # A CJK collection on Linux may expose its Japanese face to Matplotlib.
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "PingFang SC") if f in installed), "sans-serif")
    plt.rcParams.update({"font.family": family, "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharex=True, constrained_layout=True)
    styles = [(False, "#31719c", "净空半径初值"), (True, "#d2772f", "中间半径回退")]
    for ax, field, title, ylabel in zip(axes, ("time_pct", "firi_delta_pp"),
            ("生成耗时", "FIRI 空间覆盖"), ("耗时变化（%，负值表示提速）", "覆盖率变化（百分点）")):
        for fallback, color, label in styles:
            group = [r for r in rows if r[field] and (r["intermediate_radius_used"] == "True") == fallback]
            ax.scatter([float(r["core_pct"]) for r in group], [float(r[field]) for r in group],
                       c=color, label=label, alpha=.8, s=24, edgecolors="white", linewidths=.4, zorder=3)
        ax.axhline(0, color="#697787", lw=.9)
        ax.axvline(0, color="#697787", lw=.9)
        ax.grid(color="#e0e5e9", lw=.55)
        for r in rows:
            if r["case"] in ("0", "41") and r[field]:
                ax.annotate(r["case"], (float(r["core_pct"]), float(r[field])),
                            xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.margins(x=.12, y=.14)
        ax.set(title=title, xlabel="锚点间去重体积变化（%）", ylabel=ylabel)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(args.results / "paired.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
