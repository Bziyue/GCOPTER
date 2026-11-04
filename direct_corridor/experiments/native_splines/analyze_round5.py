#!/usr/bin/env python3
"""Archive all fifth-round trials, matched timings and quality checks."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_round3 import archive, rows
from analyze_round4 import generation, trajectory

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT/".scratch/native_splines/round5"
OUT = ROOT/"docs/validation/native_splines_round5"
PROFILES = ["frozen", "axis_derivatives", "axis_bounded", "projected_axis_bounded"]
MODELS = ["H5-S", "C3-before", "C3-after", "C3-pruned"]
SCREENS = ["pruning_screen", "direct_screen", "initial_screen", "lag_screen", "weak_screen", "axis_screen", "derivatives_screen", "bounded_screen", "corridor_screen", "combined_screen"]


def archive_tree(paths, target):
    """Keep complete raw results while storing repeated geometry only once."""
    contents = {}
    with tarfile.open(target, "w:gz") as tar:
        for source in paths:
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    measured = {profile: [generation(WORK/"formal_bounded"/profile/f"case_{i:02d}") for i in range(7)]
                for profile in PROFILES}
    for data in measured.values():
        for r in data:
            assert len(r["runs"]) == 21
            assert all(v["status"] == "SUCCESS" and v["support_pass"] and v["jacobian_pass"] for v in r["runs"])
            assert r["fresh_certificate"]["status"] == "SUCCESS"
    coefficient_equal = []
    for i in range(7):
        folder = WORK/"joint"/f"case_{i:02d}"
        coefficient_equal.append((folder/"C3-before.csv").read_bytes() == (folder/"C3-after.csv").read_bytes())
        for model, config in [("C3-before", "frozen"), ("C3-after", "axis_bounded"), ("C3-pruned", "projected_axis_bounded")]:
            assert (folder/(model+".csv")).read_bytes() == (WORK/"formal_bounded"/config/folder.name/"C3-OPT-native.csv").read_bytes()
    assert all(coefficient_equal)
    volumes = [{r["model"]: r for r in rows(WORK/"joint"/f"case_{i:02d}/volume_refined.jsonl")} for i in range(7)]
    assert all(r["unresolved"] == 0 and r["samples"] == 8*262144 for case in volumes for r in case.values())
    trajectories = {model: [trajectory(WORK/"trajectory"/model/f"case_{i:02d}") for i in range(7)] for model in MODELS}
    trajectory_equal = []
    for i in range(7):
        assert len({data[i]["sample"]["initial_hash"] for data in trajectories.values()}) == 1
        a = WORK/"trajectory/C3-before"/f"case_{i:02d}/hermite_trajectory.csv"
        b = WORK/"trajectory/C3-after"/f"case_{i:02d}/hermite_trajectory.csv"
        trajectory_equal.append(a.read_bytes() == b.read_bytes())
    assert all(trajectory_equal)
    exploration = {name: json.loads((WORK/name/"summary.json").read_text()) for name in SCREENS}
    results = {"reference_commit": "f72f3fe3838d359a79915ffbc9594661d5c2e15e",
               "generation": measured, "volume": volumes, "trajectory": trajectories,
               "coefficient_equal": coefficient_equal, "trajectory_equal": trajectory_equal}
    corridor_profiles = ["frozen", "projected_axis_bounded", "pruned_predict3_volume1"]
    corridor = {profile: [generation(WORK/"corridor_final"/profile/f"case_{i:02d}") for i in range(7)]
                for profile in corridor_profiles}
    for profile, data in corridor.items():
        model={"frozen": "C3-before", "projected_axis_bounded": "C3-pruned", "pruned_predict3_volume1": "C3-selected"}[profile]
        for i, r in enumerate(data):
            assert len(r["runs"]) == 21
            assert all(v["status"] == "SUCCESS" and v["support_pass"] and v["jacobian_pass"] for v in r["runs"])
            assert r["fresh_certificate"]["status"] == "SUCCESS"
            assert (WORK/"corridor_final"/profile/f"case_{i:02d}/C3-OPT-native.csv").read_bytes() == (WORK/"corridor_joint"/f"case_{i:02d}"/(model+".csv")).read_bytes()
    corridor_volumes = [{r["model"]: r for r in rows(WORK/"corridor_joint"/f"case_{i:02d}/volume_refined.jsonl")} for i in range(7)]
    assert all(r["unresolved"] == 0 and r["samples"] == 8*(1048576 if i==3 else 262144)
               for i, case in enumerate(corridor_volumes) for r in case.values())
    results |= {"selection_criteria": "corridor generation, union volume, geometric regularity and continuous safety",
                "trajectory_used_for_selection": False, "corridor_generation": corridor, "corridor_volume": corridor_volumes}
    (OUT/"results.json").write_text(json.dumps(results, indent=2)+"\n")
    (OUT/"exploration.json").write_text(json.dumps(exploration, indent=2)+"\n")
    archive_tree([WORK/"formal", WORK/"formal_bounded", WORK/"joint", WORK/"trajectory", WORK/"corridor_final", WORK/"corridor_joint"], OUT/"formal_raw.tar.gz")
    archive_tree([WORK/name for name in SCREENS], OUT/"exploration_raw.tar.gz")
    for name in ("ctest-native.log", "ctest-common.log", "build-final.log", "independent_audit.json", "evaluation.log", "corridor_audit.json", "corridor_evaluation.log"):
        archive(WORK/name, OUT/name)
    archive(WORK.parent/"round5_baseline/manifest.json", OUT/"baseline_manifest.json")
    for name in ("manifest.json", "source.tar.gz"):
        archive(WORK/"phase_bd_snapshot"/name, OUT/"phase_bd_snapshot"/name)

    colors = {"H5-S": "#c45e46", "C3-before": "#8996a5", "C3-after": "#167d93", "C3-pruned": "#7961a7"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(7)
    timings = {"H5-S": [r["h5_ms"] for r in measured["axis_bounded"]],
               "C3-before": [r["median_ms"] for r in measured["frozen"]],
               "C3-after": [r["median_ms"] for r in measured["axis_bounded"]],
               "C3-pruned": [r["median_ms"] for r in measured["projected_axis_bounded"]]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for model, data in timings.items():
        axes[0].plot(x, data, "o-", color=colors[model], label=model)
        axes[2].plot(x, [r["median_ms"] for r in trajectories[model]], "o-", color=colors[model], label=model)
    for offset, model in [(-.18, "C3-after"), (.18, "C3-pruned")]:
        axes[1].bar(x+offset, 100*(1-np.array(timings[model])/timings["C3-before"]), .36, color=colors[model], label=model)
    axes[0].set(title="Complete hot generation (21 runs)", ylabel="Median time (ms)")
    axes[1].set(title="Compared with frozen C3", ylabel="Generation time reduction (%)")
    axes[2].set(title="Common trajectory optimizer (3 runs)", ylabel="Median time (ms)")
    for ax in axes:
        ax.set_xticks(x);ax.set_xlabel("Frozen case");ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(OUT/"performance.png", dpi=175);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    keys = ["setup_ms", "assembly_ms", "solve_ms", "line_search_ms"]
    for offset, profile, model in [(-.25, "frozen", "C3-before"), (0, "axis_bounded", "C3-after"),
                                   (.25, "projected_axis_bounded", "C3-pruned")]:
        axes[0].bar(np.arange(len(keys))+offset,
                    [statistics.median(r["stages"][k] for r in measured[profile]) for k in keys],
                    .25, color=colors[model], label=model)
    axes[0].set_xticks(np.arange(len(keys)), ["Setup", "Assembly", "Band solve", "Line search"])
    axes[0].set(title="Median stage cost across cases", ylabel="Time (ms)")
    for offset, key, label in [(-.18, "gradient_support_rows", "Derivative rows"), (.18, "boundary_support_rows", "Step boundary rows")]:
        before = [r["sample"][key] for r in measured["axis_bounded"]]
        after = [r["sample"][key] for r in measured["projected_axis_bounded"]]
        axes[1].bar(x+offset, 100*(1-np.array(after)/before), .36, label=label)
    axes[1].set(title="Proof pruning across all Newton steps", xlabel="Frozen case", ylabel="Rows removed (%)")
    for ax in axes:ax.legend(fontsize=8);ax.grid(axis="y", alpha=.15)
    fig.savefig(OUT/"stages.png", dpi=175);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    pruned = [v["C3-pruned"] for v in volumes]
    axes[0].errorbar(x, [100*v["paired_delta"]/v["baseline"] for v in pruned],
                     yerr=[100*v["paired_ci95"]/v["baseline"] for v in pruned], fmt="o", capsize=4,
                     color=colors["C3-pruned"], label="C3-pruned vs C3-before")
    axes[0].axhline(0, color=colors["C3-after"], ls="--", label="C3-after: identical geometry")
    axes[0].set(title="Shared-domain union volume", xlabel="Frozen case", ylabel="Paired change (%)")
    for model in MODELS:
        axes[1].plot(x, [1000*r["sample"]["max_corridor_error_m"] for r in trajectories[model]], "o-", color=colors[model], label=model)
    axes[1].axhline(5, color="#888888", ls="--", lw=.8, label="5 mm sampling tolerance")
    axes[1].set(title="Trajectory corridor residual", xlabel="Frozen case", ylabel="Maximum sampled residual (mm)")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15);ax.set_xticks(x)
    fig.savefig(OUT/"quality.png", dpi=175);plt.close(fig)

    chosen = [("initial_screen", "distance", "Distance barrier"),
              ("initial_screen", "initial1", "Initial radius sweep"),
              ("lag_screen", "lag2", "Reuse Hessian x2"),
              ("lag_screen", "lag2_i10_c6", "Reuse + more steps"),
              ("weak_screen", "weak30", "Weak shape curvature"),
              ("pruning_screen", "projected_full", "Proof pruning")]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for screen, config, label in chosen:
        ref = {r["case"]: r for r in exploration[screen]["reference"]}
        data = exploration[screen][config]
        dt = statistics.median(100*(r["median_ms"]/ref[r["case"]]["median_ms"]-1) for r in data)
        dv = statistics.median(100*(r["swept_integral"]/ref[r["case"]]["swept_integral"]-1) for r in data)
        ax.scatter(dt, dv, s=45, label=label)
        ax.annotate(label, (dt, dv), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#888888", ls="--", lw=.8);ax.axvline(0, color="#888888", ls="--", lw=.8)
    ax.set(title="Diagnostic screens (3 runs per configuration)", xlabel="Generation time change (%)",
           ylabel="Swept integral change (%) — overlaps counted")
    ax.grid(alpha=.15);ax.margins(x=.3, y=.2)
    fig.savefig(OUT/"alternatives.png", dpi=175);plt.close(fig)

    colors["C3-selected"]="#248957"
    final_timings = {"H5-S": [r["h5_ms"] for r in corridor["pruned_predict3_volume1"]],
                     "C3-before": [r["median_ms"] for r in corridor["frozen"]],
                     "C3-pruned": [r["median_ms"] for r in corridor["projected_axis_bounded"]],
                     "C3-selected": [r["median_ms"] for r in corridor["pruned_predict3_volume1"]]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for model, data in final_timings.items():
        axes[0].plot(x, data, "o-", color=colors[model], label=model)
    for offset, model in [(-.18,"C3-pruned"),(.18,"C3-selected")]:
        axes[1].bar(x+offset, 100*(1-np.array(final_timings[model])/final_timings["C3-before"]), .36, color=colors[model], label=model)
    stages=["setup_ms","assembly_ms","solve_ms","line_search_ms"]
    for offset, profile, model in [(-.18,"frozen","C3-before"),(.18,"pruned_predict3_volume1","C3-selected")]:
        axes[2].bar(np.arange(4)+offset, [statistics.median(r["stages"][k] for r in corridor[profile]) for k in stages], .36, color=colors[model], label=model)
    axes[0].set(title="Corridor generation (21 runs)", xlabel="Frozen case", ylabel="Median time (ms)")
    axes[1].set(title="Generation time reduction", xlabel="Frozen case", ylabel="Reduction from frozen C3 (%)")
    axes[2].set(title="Median stage time across cases", ylabel="Time (ms)")
    axes[2].set_xticks(np.arange(4), ["Setup","Assembly","Solve","Line search"])
    for ax in axes:ax.grid(axis="y", alpha=.15);ax.legend(fontsize=8)
    fig.savefig(OUT/"corridor-performance.png", dpi=175);plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for model in ("C3-pruned", "C3-selected"):
        data=[v[model] for v in corridor_volumes]
        axes[0].errorbar(x, [100*v["paired_delta"]/v["baseline"] for v in data],
                        yerr=[100*v["paired_ci95"]/v["baseline"] for v in data], fmt="o-", capsize=3,
                        color=colors[model], label=model)
    for profile, model in [("frozen","C3-before"),("projected_axis_bounded","C3-pruned"),("pruned_predict3_volume1","C3-selected")]:
        data=corridor[profile]
        axes[1].plot(x, [r["sample"]["max_curvature"] for r in data], "o-", color=colors[model], label=model)
        axes[2].plot(x, [r["sample"]["shape_energy"] for r in data], "o-", color=colors[model], label=model)
    axes[0].axhline(0, color="#888888", ls="--", lw=.8)
    axes[0].set(title="Paired QMC union volume", ylabel="Change from frozen C3 (%)")
    axes[1].set(title="Maximum center curvature", ylabel="Curvature (1/m)")
    axes[2].set(title="Section-shape bending energy", ylabel="Arc-length normalized energy")
    for ax in axes:ax.set_xlabel("Frozen case");ax.set_xticks(x);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(OUT/"corridor-quality.png", dpi=175);plt.close(fig)
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "sha256.json"}
    (OUT/"sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps({"time_ms": timings, "coefficient_equal": coefficient_equal,
                      "trajectory_equal": trajectory_equal}, indent=2))


if __name__ == "__main__":
    main()
