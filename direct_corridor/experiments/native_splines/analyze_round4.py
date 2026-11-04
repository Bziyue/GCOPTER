#!/usr/bin/env python3
"""Archive the fourth-round experiments and matched final measurements."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_round3 import archive, rows

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / ".scratch/native_splines/round4"
OUT = ROOT / "docs/validation/native_splines_round4"
STAGES = ["initialization_ms", "query_ms", "separation_ms", "setup_ms", "assembly_ms",
          "solve_ms", "line_search_ms", "jacobian_check_ms", "verification_ms"]
MODELS = ["H5-S", "C3-before", "C3-after"]


def generation(folder):
    rr = rows(folder/"rows.jsonl")
    runs = [r for r in rr if r["kind"] == "generation" and
            r["model"] == "C3-OPT-native" and r["repeat"] >= 0]
    return {"status": [r["status"] for r in runs], "runs": runs, "sample": runs[0],
            "median_ms": statistics.median(r["total_ms"] for r in runs),
            "p95_ms": float(np.percentile([r["total_ms"] for r in runs], 95)),
            "h5_ms": statistics.median(r["total_ms"] for r in rr if r["kind"] == "generation"
                                         and r["model"] == "H5-S" and r["repeat"] >= 0),
            "stages": {key: statistics.median(r[key] for r in runs) for key in STAGES},
            "configuration": next((r for r in rr if r["kind"] == "configuration"
                                   and r["model"] == "C3-OPT-native"), {}),
            "gradient_check": next(r for r in rr if r["kind"] == "generation"
                                   and r["model"] == "C3-OPT-native" and r["repeat"] == -1),
            "fresh_certificate": next(r for r in rr if r["kind"] == "fresh_map_certificate"
                                      and r["model"] == "C3-OPT-native"),
            "index_ms": next(r["index_ms"] for r in rr if r["kind"] == "input"),
            "peak_rss_kib": next(r["peak_rss_kib"] for r in rr if r["kind"] == "resource")}


def trajectory(folder):
    rr = rows(folder/"runs.jsonl")
    runs = [r for r in rr if r["method"] == "hermite" and r["repeat"] >= 0]
    return {"sample": runs[0], "runs": runs,
            "median_ms": statistics.median(r["optimize_ms"] for r in runs),
            "gradient_error": max(r["gradient_relative_error"] for r in rr),
            "peak_rss_kib": int((folder/"peak_rss_kib.txt").read_text())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = {}
    for profile in ("baseline", "final"):
        source = WORK/profile
        archive(source/"manifest.json", OUT/"generation"/profile/"manifest.json")
        expected_repeats = json.loads((source/"manifest.json").read_text())["repeats"]
        final[profile] = []
        for case in range(7):
            folder = source/f"case_{case:02d}"
            record = generation(folder)
            assert len(record["runs"]) == expected_repeats
            assert all(r["status"] == "SUCCESS" and r["support_pass"] and r["jacobian_pass"]
                       for r in record["runs"])
            assert record["fresh_certificate"]["status"] == "SUCCESS"
            final[profile].append(record)
            for path in folder.iterdir():
                if path.suffix in (".jsonl", ".csv"):
                    archive(path, OUT/"generation"/profile/folder.name/path.name)

    volumes = []
    for name in ("manifest.json", "volume_manifest.json"):
        archive(WORK/"joint"/name, OUT/"geometry"/name)
    for case in range(7):
        folder = WORK/"joint"/f"case_{case:02d}"
        rr = rows(folder/"volume_refined.jsonl")
        assert all(r["unresolved"] == 0 and r["samples"] == 8*262144 for r in rr)
        volumes.append({r["model"]: r for r in rr})
        for path in folder.iterdir():
            if path.suffix in (".csv", ".jsonl"):
                archive(path, OUT/"geometry"/folder.name/path.name)

    trajectories = {}
    archive(WORK/"trajectory"/"manifest.json", OUT/"trajectory"/"manifest.json")
    for model in MODELS:
        trajectories[model] = []
        for case in range(7):
            folder = WORK/"trajectory"/model/f"case_{case:02d}"
            record = trajectory(folder)
            assert len(record["runs"]) == 3
            trajectories[model].append(record)
            for path in folder.iterdir():
                if path.name == "firi_trajectory.csv" and model != "H5-S":
                    continue
                if path.suffix in (".csv", ".jsonl", ".txt"):
                    archive(path, OUT/"trajectory"/model/folder.name/path.name)
    for case in range(7):
        assert len({v[case]["sample"]["initial_hash"] for v in trajectories.values()}) == 1

    exploration = {}
    folders = ["baseline_early", "baseline_previous", "baseline_unpinned", "final_unpinned", "final_cached", "exact_ablations", "cache", "separator", "exchange", "lazy_exchange", "adaptive", "continuation",
               "combined", "trajectory_screen", "trajectory_screen_more"]
    with tarfile.open(OUT/"exploration.tar.gz", "w:gz") as tar:
        contents = {}
        for name in folders:
            source = WORK/name
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                arcname = str(path.relative_to(WORK))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in contents:
                    info = tar.gettarinfo(path, arcname=arcname)
                    info.type, info.linkname, info.size = tarfile.LNKTYPE, contents[digest], 0
                    tar.addfile(info)
                else:
                    tar.add(path, arcname=arcname)
                    contents[digest] = arcname
            if name.startswith("trajectory_"):
                exploration[name] = {m.name: [trajectory(p) for p in sorted(m.glob("case_*"))]
                                     for m in source.glob("C3-*")}
            else:
                paths = sorted(source.glob("**/case_*/rows.jsonl"))
                exploration[name] = {str(p.parent.relative_to(source)): generation(p.parent) for p in paths}

    delta = []
    for case in range(7):
        before = np.loadtxt(WORK/"baseline"/f"case_{case:02d}/C3-OPT-native.csv", delimiter=",", skiprows=1)
        after = np.loadtxt(WORK/"final"/f"case_{case:02d}/C3-OPT-native.csv", delimiter=",", skiprows=1)
        delta.append(float(np.max(np.abs(after-before))))
        assert (WORK/"final"/f"case_{case:02d}/C3-OPT-native.csv").read_bytes() == (WORK/"joint"/f"case_{case:02d}/C3-after.csv").read_bytes()
    summary = {"reference": "round3-final-uncommitted", "head": "5ed3f99", "generation": final,
               "volume": volumes, "trajectory": trajectories, "coefficient_deltas": delta}
    (OUT/"results.json").write_text(json.dumps(summary, indent=2)+"\n")
    (OUT/"exploration.json").write_text(json.dumps(exploration, indent=2)+"\n")
    for name in ("ctest-native.log", "ctest-common.log", "optimizer-math.log", "build-final.log", "independent_audit.json"):
        archive(WORK/name, OUT/name)
    for name in ("manifest.json", "source.tar.gz"):
        archive(WORK.parent/"round4_baseline"/name, OUT/"baseline_source"/name)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(7)
    colors = {"H5-S": "#c45e46", "C3-before": "#919da8", "C3-after": "#167d93"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for model, data in [("H5-S", [r["h5_ms"] for r in final["final"]]),
                        ("C3-before", [r["median_ms"] for r in final["baseline"]]),
                        ("C3-after", [r["median_ms"] for r in final["final"]])]:
        axes[0].plot(x, data, "o-", color=colors[model], label=model)
    axes[0].set(title="Complete hot generation", ylabel="Time (ms)")
    axes[1].bar(x, [100*(1-a["median_ms"]/b["median_ms"]) for a, b in
                          zip(final["final"], final["baseline"])], color=colors["C3-after"])
    axes[1].set(title="Same output coefficients", ylabel="Generation time reduction (%)")
    for model in MODELS:
        axes[2].plot(x, [r["median_ms"] for r in trajectories[model]], "o-", color=colors[model], label=model)
    axes[2].set(title="Same initial trajectory", ylabel="Optimization time (ms)")
    for ax in axes:
        ax.set_xticks(x);ax.set_xlabel("Frozen case");ax.grid(alpha=.15)
    axes[0].legend();axes[2].legend()
    fig.savefig(OUT/"performance.png", dpi=175);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    keys = ["separation_ms", "setup_ms", "assembly_ms", "solve_ms", "line_search_ms"]
    labels = ["Separation", "Setup", "Assembly", "Band solve", "Line search"]
    for offset, profile, label in [(-.18, "baseline", "C3-before"), (.18, "final", "C3-after")]:
        axes[0].bar(np.arange(len(keys))+offset,
                    [statistics.median(r["stages"][k] for r in final[profile]) for k in keys],
                    width=.36, color=colors[label], label=label)
    axes[0].set_xticks(np.arange(len(keys)), labels, rotation=15)
    axes[0].set(title="Generation stages", ylabel="Median across cases (ms)")
    eager = [generation(WORK/"lazy_exchange/eager"/f"case_{i:02d}")["sample"]["priority_evaluations"] for i in range(7)]
    lazy = [r["sample"]["priority_evaluations"] for r in final["final"]]
    axes[1].bar(x-.18, np.array(eager)/1e6, .36, color=colors["C3-before"], label="Eager")
    axes[1].bar(x+.18, np.array(lazy)/1e6, .36, color=colors["C3-after"], label="Lazy")
    axes[1].set(title="Exact obstacle priority evaluations", ylabel="Evaluations (million)", xlabel="Frozen case")
    for ax in axes:ax.legend();ax.grid(axis="y", alpha=.15)
    fig.savefig(OUT/"stages.png", dpi=175);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    candidates = [("C3-predict-fast", "Predict / fewer steps", "continuation/predict_one_c3", "continuation/reference", "trajectory_screen_more"),
                  ("C3-predict-quality", "Predict / full budget", "continuation/predict_one", "continuation/reference", "trajectory_screen"),
                  ("C3-volume", "Volume correction", "continuation/volume1", "continuation/reference", "trajectory_screen_more"),
                  ("C3-adaptive", "Adaptive knots", "adaptive/mode2", "adaptive/mode0", "trajectory_screen"),
                  ("C3-predict", "Predict + volume", "combined/predict_c3_volume", "continuation/reference", "trajectory_screen")]
    palette = ["#287fa3", "#7766a3", "#d78b3f", "#cc605d", "#4b9777"]
    offsets = [(-8, -25), (5, 8), (5, -20), (7, 5), (4, 12)]
    for color, offset, (model, label, profile, reference, screen) in zip(palette, offsets, candidates):
        ratio = []
        for case in range(7):
            a = generation(WORK/profile/f"case_{case:02d}")["median_ms"]
            b = generation(WORK/reference/f"case_{case:02d}")["median_ms"]
            ratio.append(100*(a/b-1))
        paired = [100*v[model]["paired_delta"]/v[model]["baseline"] for v in volumes]
        accepted = sum(r["sample"]["sampled_feasible"] for r in exploration[screen][model])
        axes[0].plot(x, paired, "o-", color=color, label=f"{label} ({accepted}/7)")
        axes[1].scatter(statistics.median(ratio), statistics.median(paired), color=color, s=70)
        axes[1].annotate(label, (statistics.median(ratio), statistics.median(paired)),
                         xytext=offset, textcoords="offset points", fontsize=8)
    axes[0].set(title="Shared-domain union volume", xlabel="Frozen case", ylabel="Change from C3-before (%)")
    axes[0].set_xticks(x);axes[0].legend(fontsize=8, title="Trajectory sampling acceptance")
    axes[1].set(title="Exploratory time / volume tradeoff", xlabel="Generation time change (%)", ylabel="Median union-volume change (%)")
    axes[1].margins(x=.3,y=.25)
    for ax in axes:ax.axhline(0, color="#888888", ls="--", lw=.8);ax.grid(alpha=.15)
    axes[1].axvline(0, color="#888888", ls="--", lw=.8)
    fig.savefig(OUT/"alternatives.png", dpi=175);plt.close(fig)

    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "sha256.json"}
    (OUT/"sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps({"coefficient_deltas": delta,
                      "time_reduction_percent": [100*(1-a["median_ms"]/b["median_ms"]) for a,b in
                                                  zip(final["final"],final["baseline"])]}, indent=2))


if __name__ == "__main__":
    main()
