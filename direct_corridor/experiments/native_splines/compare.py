#!/usr/bin/env python3
"""Archive full serial runs and make paired tables/figures from frozen outputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics as stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[3]
MODELS = ["H5-S", "C3-OPT-native", "C3-OPT-project", "C3-NBC-project"]
COLORS = ["#b84637", "#157899", "#8060a8", "#24955e"]

def median(rows, key):
    return stats.median(r[key] for r in rows if r.get("repeat", -1) >= 0)

def curve_points(path, stride=1):
    rows = np.genfromtxt(path, delimiter=",", skip_header=1)
    degree = int(rows[0, 0])
    polygons, centers = [], []
    angle = np.linspace(0, 2*np.pi, 32)
    disk = np.array([np.cos(angle), np.sin(angle)])
    for index, piece in enumerate(rows.reshape(-1, degree+1, 12)):
        if index % stride:
            continue
        for t in (0., .5):
            from math import comb
            b = np.array([comb(degree, j)*t**j*(1-t)**(degree-j) for j in range(degree+1)])
            y = b @ piece[:, 3:]
            m = y[3:].reshape(2, 3).T
            polygons.append((y[:3, None]+m @ disk)[:2].T)
            centers.append(y[:2])
    return polygons, np.asarray(centers)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".scratch/native_splines/final")
    parser.add_argument("--trajectory", default=".scratch/native_splines/trajectory")
    parser.add_argument("--output", default="docs/validation/native_splines")
    args = parser.parse_args()
    source, out = ROOT / args.input, ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    summary = {"schema": "native-splines-summary-v1", "generation_manifest": manifest,
               "cases": [], "trajectory": [], "completed_scope": "seven-case development experiment, phases 0/A/native-B and common-backend C"}
    all_rows = []
    table = ["| Case | H5-S ms | C3 native ms | Native ΔV | C3 projection ms* | Projection ΔV |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in manifest["cases"]:
        i = run["case"]
        directory = ROOT / run["output"]
        rows = [json.loads(s) for s in (directory / "rows.jsonl").read_text().splitlines()]
        refined = directory / "volume_refined.jsonl"
        if refined.exists():
            rows = [r for r in rows if r["kind"] != "volume"] + [json.loads(s) for s in refined.read_text().splitlines()]
        all_rows.append(rows)
        archived = out / f"case_{i:02d}"
        archived.mkdir(exist_ok=True)
        shutil.copy2(directory / "rows.jsonl", archived / "rows.jsonl")
        if refined.exists():
            shutil.copy2(refined, archived / "volume_refined.jsonl")
        for path in directory.glob("*.csv"):
            with gzip.GzipFile(filename=str(archived / (path.name+".gz")), mode="wb", mtime=0) as f:
                f.write(path.read_bytes())
        old = ROOT / f".scratch/smooth_sweep/final/case_{i:02d}/smooth_controls.csv"
        a = np.genfromtxt(directory / "H5-S.csv", delimiter=",", skip_header=1)[:, 1:]
        b = np.genfromtxt(old, delimiter=",", skip_header=1)
        error = float(np.max(np.abs(a-b))) if a.shape == b.shape else None
        case = {"case": i, "baseline_coefficient_max_error": error,
                "baseline_previous_sha256": hashlib.sha256(old.read_bytes()).hexdigest(), "models": {}}
        baseline_volume = next(r["baseline"] for r in rows if r["kind"] == "volume")
        for model in MODELS:
            runs = [r for r in rows if r["kind"] == "generation" and r["model"] == model]
            successes = [r for r in runs if r["status"] == "SUCCESS"]
            quality_row = next((r for r in rows if r.get("model") == model and r.get("repeat") == 0 and "center_energy" in r), {})
            volume = next((r for r in rows if r["kind"] == "volume" and r["model"] == model),
                          {"volume": baseline_volume, "paired_delta": 0., "paired_ci95": 0.})
            fresh = next((r for r in rows if r["kind"] == "fresh_map_certificate" and r["model"] == model), {})
            record = {"successes": sum(r.get("repeat", -1) >= 0 for r in successes),
                      "attempts": sum(r.get("repeat", -1) >= 0 for r in runs),
                      "median_ms": median(successes, "total_ms") if successes else None,
                      "p95_ms": float(np.percentile([r["total_ms"] for r in successes if r["repeat"] >= 0], 95)) if successes else None,
                      "volume": volume, "volume_change_pct": 100*volume["paired_delta"]/baseline_volume,
                      "volume_ci95_pct": 100*volume["paired_ci95"]/baseline_volume,
                      "quality": quality_row, "fresh_certificate": fresh,
                      "stage_medians": {k: median(successes, k) for k in ("assembly_ms", "solve_ms", "setup_ms", "query_ms", "separation_ms", "line_search_ms") if successes and k in successes[0]}}
            case["models"][model] = record
        summary["cases"].append(case)
        h, n, p = [case["models"][m] for m in MODELS[:3]]
        table.append(f"| {i} | {h['median_ms']:.2f} | {n['median_ms']:.2f} | {n['volume_change_pct']:+.2f}% ± {n['volume_ci95_pct']:.2f}% | {p['median_ms']:.2f} | {p['volume_change_pct']:+.2f}% ± {p['volume_ci95_pct']:.2f}% |")
    (out / "generation_table.md").write_text("\n".join(table)+"\n\n*Projection includes H5-S generation; ± is a paired 95% QMC interval.\n")
    traj = ROOT / args.trajectory
    if (traj / "manifest.json").exists():
        tm = json.loads((traj / "manifest.json").read_text())
        summary["trajectory_manifest"] = tm
        for record in tm["cases"]:
            if record["status"] != "completed":
                summary["trajectory"].append(record)
                continue
            rows = [json.loads(s) for s in (ROOT / record["output"] / "runs.jsonl").read_text().splitlines()]
            saved = out / "trajectory" / record["model"] / f"case_{record['case']:02d}"
            saved.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / record["output"] / "runs.jsonl", saved / "runs.jsonl")
            for path in (ROOT / record["output"]).glob("*_trajectory.csv"):
                if path.name == "firi_trajectory.csv" and record["model"] != "H5-S":
                    continue  # One shared FIRI trajectory per input.
                with gzip.GzipFile(filename=str(saved / (path.name+".gz")), mode="wb", mtime=0) as f:
                    f.write(path.read_bytes())
            summary["trajectory"].append(record | {"runs": rows})
        by_case = {}
        for record in summary["trajectory"]:
            for row in record.get("runs", []):
                by_case.setdefault(record["case"], set()).add(row["initial_hash"])
        summary["trajectory_initial_hashes_match"] = all(len(v) == 1 for v in by_case.values())
        if not summary["trajectory_initial_hashes_match"]:
            raise RuntimeError("trajectory initial states differ across corridors")
        table_t = ["| Model | Optimizer ms (median across cases) | Sampled feasible cases | Median objective / H5-S |",
                   "| --- | ---: | ---: | ---: |"]
        for model in MODELS + ["H5-C3-T", "FIRI"]:
            times, objectives, feasible = [], [], []
            for i in range(7):
                base = next(r for r in summary["trajectory"] if r["case"] == i and r["model"] == "H5-S")
                target = base if model == "FIRI" else next(r for r in summary["trajectory"] if r["case"] == i and r["model"] == model)
                br = [r for r in base["runs"] if r["method"] == "hermite" and r["repeat"] >= 0]
                tr = [r for r in target["runs"] if r["method"] == ("firi" if model == "FIRI" else "hermite") and r["repeat"] >= 0]
                times.append(median(tr, "optimize_ms"))
                objectives.append(median(tr, "objective")/median(br, "objective"))
                feasible.append(all(r["sampled_feasible"] for r in tr))
            table_t.append(f"| {model} | {stats.median(times):.1f} | {sum(feasible)}/7 | {stats.median(objectives):.5f} |")
        (out / "trajectory_table.md").write_text("\n".join(table_t)+"\n")
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, color in zip(MODELS, COLORS):
        times, volumes, energy = [], [], []
        for case in summary["cases"]:
            r, h = case["models"][model], case["models"]["H5-S"]
            times.append(r["median_ms"]/h["median_ms"])
            volumes.append(100+r["volume_change_pct"])
            q, b = r["quality"], h["quality"]
            energy.append((q["center_energy"]+q["shape_energy"])/(b["center_energy"]+b["shape_energy"]))
        axes[0].scatter(times, volumes, color=color, label=model)
        for i, x, y in zip(range(7), times, volumes):
            if model != "H5-S": axes[0].annotate(str(i), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        axes[1].plot(range(7), energy, "o-", color=color, label=model, lw=1)
    axes[0].axhline(100, color="#999999", ls="--", lw=.7)
    axes[0].set(xlabel="Generation time / H5-S", ylabel="Union volume (% of H5-S)", title="Complete generation, shared safety model")
    axes[1].set(xlabel="Frozen case", ylabel="Second-derivative energy / H5-S", title="Center + matrix energy in seed arc coordinates", yscale="log")
    axes[0].legend(fontsize=8)
    fig.savefig(out / "quality_time.png");plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    for degree, color in [(3, COLORS[1]), (5, COLORS[0])]:
        values=[]
        for depth in range(4):
            ratios=[]
            for rows in all_rows:
                selected=[r for r in rows if r["kind"] == "elevation" and r["degree"] == degree and r["depth"] == depth]
                ratios.append(median(selected, "support_ms"))
            values.append(stats.median(ratios))
        ax.plot(range(4), values, "o-", color=color, label=f"Degree {degree}")
    ax.set(xlabel="Uniform subdivision depth", ylabel="Median support evaluation (ms)", title="Identical cubic geometry: native vs exact elevation", xticks=range(4))
    ax.legend();fig.savefig(out / "certificate_degree.png");plt.close(fig)
    fig, axes=plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
    for row, case in enumerate((1, 6)):
        meta=manifest["cases"][case]
        points=np.loadtxt(ROOT / meta["map"])[::7]
        seed=np.loadtxt(ROOT / meta["seed"])
        lo=seed[:, :2].min(0)-4;hi=seed[:, :2].max(0)+4
        points=points[(points[:, 0]>lo[0])&(points[:, 0]<hi[0])&(points[:, 1]>lo[1])&(points[:, 1]<hi[1])&(abs(points[:, 2]-2.5)<.4)]
        for col, model in enumerate(MODELS[:3]):
            ax=axes[row, col]
            ax.scatter(points[:, 0], points[:, 1], s=3, c="#aab3bd", alpha=.25, edgecolors="none", rasterized=True)
            polygons, centers=curve_points(ROOT / meta["output"] / f"{model}.csv")
            ax.add_collection(PolyCollection(polygons, facecolors=COLORS[col], edgecolors="none", alpha=.06))
            ax.plot(seed[:, 0], seed[:, 1], "--", color="#a38e52", lw=.7)
            ax.plot(centers[:, 0], centers[:, 1], color=COLORS[col], lw=1)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect="equal",title=f"Case {case}: {model}",xlabel="x (m)",ylabel="y (m)")
    fig.savefig(out / "corridor_comparison.png");plt.close(fig)
    if summary["trajectory"]:
        fig, axes=plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, case in zip(axes, (1, 6)):
            for model, color in zip(MODELS, COLORS):
                path=traj / model / f"case_{case:02d}" / "hermite_trajectory.csv"
                if path.exists():
                    t=np.genfromtxt(path, delimiter=",", skip_header=1)
                    ax.plot(t[:, 1], t[:, 2], c=color, lw=1, label=model)
            p=traj / "H5-S" / f"case_{case:02d}" / "firi_trajectory.csv"
            t=np.genfromtxt(p, delimiter=",", skip_header=1)
            ax.plot(t[:, 1], t[:, 2], "--", c="#888888", lw=1, label="FIRI")
            ax.set(title=f"Case {case}: identical trajectory initialization", xlabel="x (m)", ylabel="y (m)", aspect="equal")
        axes[0].legend(fontsize=8)
        fig.savefig(out / "trajectory_comparison.png");plt.close(fig)
    print(out / "results.json")
    print("\n".join(table))

if __name__ == "__main__":
    main()
