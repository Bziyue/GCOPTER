#!/usr/bin/env python3
"""Archive round-three measurements and plot the matched comparisons."""
import gzip
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

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / ".scratch/native_splines/round3"
OUT = ROOT / "docs/validation/native_splines_round3"
STAGES = ["initialization_ms", "cover_ms", "setup_ms", "assembly_ms", "solve_ms",
          "line_search_ms", "jacobian_check_ms", "verification_ms"]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def archive(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".csv":
        with source.open("rb") as a, gzip.open(str(target) + ".gz", "wb") as b:
            shutil.copyfileobj(a, b)
    elif source.suffix == ".log":
        target.write_text("\n".join(line.rstrip() for line in source.read_text().splitlines()) + "\n")
    else:
        shutil.copy2(source, target)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generation = {}
    for profile in ["baseline", "blocks", "merged", "final"]:
        source = WORK / profile
        archive(source / "manifest.json", OUT / "generation" / profile / "manifest.json")
        generation[profile] = []
        for case in range(7):
            folder = source / f"case_{case:02d}"
            records = rows(folder / "rows.jsonl")
            runs = [r for r in records if r.get("kind") == "generation" and
                    r["model"] == "C3-OPT-native" and r["repeat"] >= 0]
            h5 = [r for r in records if r.get("kind") == "generation" and
                  r["model"] == "H5-S" and r["repeat"] >= 0]
            expected = 7 if profile in ("baseline", "final") else 3
            assert len(runs) == expected
            assert all(r["status"] == "SUCCESS" and r["support_pass"] and r["jacobian_pass"] for r in runs)
            fresh = next(r for r in records if r["kind"] == "fresh_map_certificate" and r["model"] == "C3-OPT-native")
            assert fresh["status"] == "SUCCESS" and fresh["sampled_penetration_m"] <= 1e-6
            generation[profile].append({
                "case": case, "median_ms": statistics.median(r["total_ms"] for r in runs),
                "p95_ms": float(np.percentile([r["total_ms"] for r in runs], 95)),
                "h5_ms": statistics.median(r["total_ms"] for r in h5),
                "stages": {k: statistics.median(r[k] for r in runs) for k in STAGES},
                "sample": runs[0], "runs": runs,
                "gradient_check": next(r for r in records if r["kind"] == "generation" and
                                       r["model"] == "C3-OPT-native" and r["repeat"] == -1),
                "index_ms": next(r["index_ms"] for r in records if r["kind"] == "input"),
                "peak_rss_kib": next(r["peak_rss_kib"] for r in records if r["kind"] == "resource")})
            for path in folder.iterdir():
                if path.suffix == ".jsonl" or (profile in ("baseline", "final") and path.suffix == ".csv"):
                    archive(path, OUT / "generation" / profile / folder.name / path.name)

    volume = []
    joint = WORK / "joint"
    archive(joint / "manifest.json", OUT / "geometry" / "manifest.json")
    archive(joint / "volume_manifest.json", OUT / "geometry" / "volume_manifest.json")
    for case in range(7):
        folder = joint / f"case_{case:02d}"
        rr = rows(folder / "volume_refined.jsonl")
        assert all(r["unresolved"] == 0 and r["samples"] == 8 * 262144 for r in rr)
        volume.append({r["model"]: r for r in rr})
        for path in folder.iterdir():
            if path.suffix in (".csv", ".jsonl"):
                archive(path, OUT / "geometry" / folder.name / path.name)

    trajectory = {}
    source = WORK / "trajectory"
    archive(source / "manifest.json", OUT / "trajectory" / "manifest.json")
    for model in ["H5-S", "C3-before", "C3-after"]:
        trajectory[model] = []
        for case in range(7):
            folder = source / model / f"case_{case:02d}"
            rr = rows(folder / "runs.jsonl")
            runs = [r for r in rr if r["method"] == "hermite" and r["repeat"] >= 0]
            assert len(runs) == 3
            trajectory[model].append({"case": case, "sample": runs[0], "runs": runs,
                "median_ms": statistics.median(r["optimize_ms"] for r in runs),
                "gradient_error": max(r["gradient_relative_error"] for r in rr),
                "peak_rss_kib": int((folder / "peak_rss_kib.txt").read_text())})
            for path in folder.iterdir():
                if path.name == "firi_trajectory.csv" and model != "H5-S":
                    continue
                if path.suffix in (".csv", ".jsonl", ".txt"):
                    archive(path, OUT / "trajectory" / model / folder.name / path.name)
    for case in range(7):
        assert len({data[case]["sample"]["initial_hash"] for data in trajectory.values()}) == 1

    exploration = {}
    folders = ["schedule", "reserve", "no_tensor", "progress", "inflation", "energy",
               "trajectory_screen", "trajectory_energy_screen", "trajectory_candidates"]
    # Retain raw exploratory outputs in one compressed artifact, not hundreds
    # of CSVs in the working tree. Each implementation stage has its own hash.
    with tarfile.open(OUT / "exploration.tar.gz", "w:gz") as tar:
        contents = {}
        for name in folders:
            source = WORK / name
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    arcname = str(path.relative_to(WORK))
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest in contents:
                        info = tar.gettarinfo(path, arcname=arcname)
                        info.type = tarfile.LNKTYPE
                        info.linkname = contents[digest]
                        info.size = 0
                        tar.addfile(info)
                    else:
                        tar.add(path, arcname=arcname)
                        contents[digest] = arcname
            if (source / "summary.json").exists():
                exploration[name] = json.loads((source / "summary.json").read_text())
    (OUT / "exploration.json").write_text(json.dumps(exploration, indent=2) + "\n")
    for name in ["ctest-native.log", "ctest-common.log", "build-final.log", "independent_audit.json"]:
        archive(WORK / name, OUT / name)
    archive(WORK.parent / "round3_baseline/manifest.json", OUT / "baseline_binary.json")
    coefficient_delta = 0.
    for case in range(7):
        a = np.loadtxt(WORK / "baseline" / f"case_{case:02d}/C3-OPT-native.csv", delimiter=",", skiprows=1)
        b = np.loadtxt(WORK / "blocks" / f"case_{case:02d}/C3-OPT-native.csv", delimiter=",", skiprows=1)
        coefficient_delta = max(coefficient_delta, float(np.max(np.abs(a-b))))
    summary = {"baseline_commit": "5ed3f99", "generation": generation, "volume": volume,
               "trajectory": trajectory, "assembly_only_coefficient_delta": coefficient_delta}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(7)
    colors = {"H5-S": "#be523b", "C3-before": "#8d97a3", "C3-after": "#147e9d"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), layout="constrained")
    for label, values in [("H5-S", [r["h5_ms"] for r in generation["final"]]),
                          ("C3-before", [r["median_ms"] for r in generation["baseline"]]),
                          ("C3-after", [r["median_ms"] for r in generation["final"]])]:
        axes[0].plot(x, values, "o-", label=label, color=colors[label])
    axes[0].set(title="Complete hot generation", ylabel="Time (ms)")
    axes[1].errorbar(x, [100*r["C3-after"]["paired_delta"]/r["C3-after"]["baseline"] for r in volume],
                     yerr=[100*r["C3-after"]["paired_ci95"]/r["C3-after"]["baseline"] for r in volume],
                     marker="o", color=colors["C3-after"], label="Paired 95% interval")
    axes[1].axhline(0, color="#8d97a3", ls="--", lw=1)
    axes[1].set(title="Deduplicated union volume", ylabel="Change from C3-before (%)")
    for model, values in trajectory.items():
        axes[2].plot(x, [r["median_ms"] for r in values], "o-", color=colors[model], label=model)
    axes[2].set(title="Same initial trajectory", ylabel="Optimization time (ms)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xlabel("Frozen case"); ax.grid(alpha=.15); ax.legend(fontsize=9)
    fig.savefig(OUT / "performance.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    names = ["Obstacle cover", "Setup", "Hessian assembly", "Band solve", "Line search"]
    keys = ["cover_ms", "setup_ms", "assembly_ms", "solve_ms", "line_search_ms"]
    pos = np.arange(len(names))
    for offset, profile, label in [(-.18, "baseline", "C3-before"), (.18, "final", "C3-after")]:
        values = [statistics.median(r["stages"][k] for r in generation[profile]) for k in keys]
        bars = ax.bar(pos+offset, values, width=.36, label=label, color=colors[label])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(pos, names)
    ax.set(ylabel="Median across cases (ms)", title="Where the generation time was removed", ylim=(0, 30))
    ax.legend(); ax.grid(axis="y", alpha=.15)
    fig.savefig(OUT / "stages.png", dpi=175)
    plt.close(fig)
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Archived {len(hashes)} files in {OUT}")


if __name__ == "__main__":
    main()
