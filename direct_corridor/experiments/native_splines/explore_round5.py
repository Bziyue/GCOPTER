#!/usr/bin/env python3
"""Paired serial experiments for proved support pruning and solver kernels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
CONFIGURATIONS = {
    "frozen": {},
    "reference": {},
    "polar": {"POLAR_PRUNING": 1},
    "projected_assembly": {"PROJECTED_PRUNING": 1},
    "projected_full": {"PROJECTED_PRUNING": 2},
    "direct": {"GROUPED_ASSEMBLY": 0},
    "projected_direct": {"PROJECTED_PRUNING": 2, "GROUPED_ASSEMBLY": 0},
    "distance": {"DISTANCE_BARRIER": 1},
    "projected_distance": {"PROJECTED_PRUNING": 2, "DISTANCE_BARRIER": 1},
    "initial1": {"INITIAL_SWEEPS": 1},
    "initial2": {"INITIAL_SWEEPS": 2},
    "initial4": {"INITIAL_SWEEPS": 4},
    "initial1_half": {"INITIAL_SWEEPS": 1, "INITIAL_FRACTION": .5},
    "projected_initial1": {"PROJECTED_PRUNING": 2, "INITIAL_SWEEPS": 1},
    "projected_distance_initial1": {"PROJECTED_PRUNING": 2, "DISTANCE_BARRIER": 1, "INITIAL_SWEEPS": 1},
    "lag2": {"HESSIAN_INTERVAL": 2},
    "lag3": {"HESSIAN_INTERVAL": 3},
    "lag2_c5": {"HESSIAN_INTERVAL": 2, "CORRECTOR_ITERATIONS": 5},
    "lag2_i10_c6": {"HESSIAN_INTERVAL": 2, "INITIAL_ITERATIONS": 10, "ITERATIONS": 10, "CORRECTOR_ITERATIONS": 6},
    "projected_lag2": {"PROJECTED_PRUNING": 2, "HESSIAN_INTERVAL": 2},
    "projected_lag2_c5": {"PROJECTED_PRUNING": 2, "HESSIAN_INTERVAL": 2, "CORRECTOR_ITERATIONS": 5},
    "weak15": {"WEAK_CURVATURE_RATIO": .15},
    "weak30": {"WEAK_CURVATURE_RATIO": .3},
    "weak50": {"WEAK_CURVATURE_RATIO": .5},
    "weak70": {"WEAK_CURVATURE_RATIO": .7},
    "projected_weak30": {"PROJECTED_PRUNING": 2, "WEAK_CURVATURE_RATIO": .3},
    "projected_weak50": {"PROJECTED_PRUNING": 2, "WEAK_CURVATURE_RATIO": .5},
    "projected_weak70": {"PROJECTED_PRUNING": 2, "WEAK_CURVATURE_RATIO": .7},
    "axis": {"AXIS_KERNELS": 1},
    "projected_axis": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1},
    "axis_weak30": {"AXIS_KERNELS": 1, "WEAK_CURVATURE_RATIO": .3},
    "projected_axis_weak50": {"AXIS_KERNELS": 1, "PROJECTED_PRUNING": 2, "WEAK_CURVATURE_RATIO": .5},
    "derivatives": {"SKIP_UNUSED_VALUES": 1},
    "axis_derivatives": {"AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1},
    "projected_derivatives": {"PROJECTED_PRUNING": 2, "SKIP_UNUSED_VALUES": 1},
    "projected_axis_derivatives": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1},
    "axis_bounded": {"AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1},
    "projected_axis_bounded": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1},
    "pruned_volume1": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1, "VOLUME_REFINEMENTS": 1},
    "pruned_predict3": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1, "PATH_PREDICTOR": 1, "CORRECTOR_ITERATIONS": 3},
    "pruned_predict3_volume1": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1, "PATH_PREDICTOR": 1, "CORRECTOR_ITERATIONS": 3, "VOLUME_REFINEMENTS": 1},
    "pruned_predict4": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1, "PATH_PREDICTOR": 1},
    "pruned_weak50": {"PROJECTED_PRUNING": 2, "AXIS_KERNELS": 1, "SKIP_UNUSED_VALUES": 1, "BOUNDED_LINE": 1, "WEAK_CURVATURE_RATIO": .5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".scratch/native_splines/round5/screen")
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument("--cases", default="0,1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    from explore import swept_integral

    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((ROOT/"docs/validation/native_splines/frozen_manifest.json").read_text())["inputs"]
    candidate = ROOT/".scratch/native_splines/build/native_spline_benchmark"
    frozen = ROOT/".scratch/native_splines/round5_baseline/native_spline_benchmark"
    names = args.configurations.split(",")
    binaries = {name: frozen if name == "frozen" else candidate for name in names}
    baseline_settings = {"PROJECTED_PRUNING": 0, "POLAR_PRUNING": 0,
                         "AXIS_KERNELS": 0, "SKIP_UNUSED_VALUES": 0,
                         "BOUNDED_LINE": 0,
                         "DISTANCE_BARRIER": 0, "INITIAL_SWEEPS": 0,
                         "HESSIAN_INTERVAL": 1, "WEAK_CURVATURE_RATIO": 0,
                         "PATH_PREDICTOR": 0, "VOLUME_REFINEMENTS": 0,
                         "INITIAL_ITERATIONS": 8, "ITERATIONS": 8, "CORRECTOR_ITERATIONS": 4,
                         "TENSOR_ASSEMBLY": 0, "CACHE_VALUES": 0, "CACHED_LINE": 1,
                         "PROGRESS_DEPTH": 0, "REUSE_ENERGY": 1, "LAZY_PRIORITIES": 1}
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "address_space_limit_bytes": 4*1024**3, "cpu": args.cpu, "repeats": args.repeats,
                "binaries": {name: {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                             for name, p in binaries.items()},
                "sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.rglob("*")
                                   if p.is_file() and p.suffix in {".cpp", ".hpp", ".py", ".txt"}},
                "cases": []}
    summary = {name: [] for name in names}
    for case in map(int, args.cases.split(",")):
        row = inputs[case]
        for kind in ("seed", "map"):
            assert hashlib.sha256((ROOT/row[kind]).read_bytes()).hexdigest() == row[kind+"_sha256"]
        for name in names if case % 2 == 0 else list(reversed(names)):
            target = out/name/f"case_{case:02d}"
            target.mkdir(parents=True)
            settings = baseline_settings | CONFIGURATIONS[name]
            env = {k: v for k, v in os.environ.items() if not k.startswith("NATIVE_")}
            env |= {"NATIVE_"+k: str(v) for k, v in settings.items()}
            env |= {"NATIVE_CANDIDATES": "1", "NATIVE_PRESET": "balanced",
                    "NATIVE_GENERATION_ONLY": "1", "NATIVE_METHOD_COUNT": "1"}
            command = [str(binaries[name]), str(ROOT/row["seed"]), str(ROOT/row["map"]), str(target),
                       *map(str, row["bounds_lower"]+row["bounds_upper"]), str(args.repeats), "0"]
            started = time.monotonic()
            with (target/"process.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            record = row | {"configuration": name, "settings": settings, "command": command,
                            "returncode": result.returncode, "elapsed_s": time.monotonic()-started,
                            "output": str(target.relative_to(ROOT))}
            manifest["cases"].append(record)
            (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            result.check_returncode()
            rows = [json.loads(line) for line in (target/"rows.jsonl").read_text().splitlines()]
            runs = [r for r in rows if r.get("kind") == "generation" and r.get("model") == "C3-OPT-native" and r["repeat"] >= 0]
            value = {"case": case, "status": [r["status"] for r in runs], "runs": runs,
                     "median_ms": statistics.median(r["total_ms"] for r in runs),
                     "h5_ms": statistics.median(r["total_ms"] for r in rows if r.get("model") == "H5-S" and r.get("kind") == "generation" and r["repeat"] >= 0)}
            if all(r["status"] == "SUCCESS" for r in runs):
                geometry = target/"C3-OPT-native.csv"
                value["geometry_sha256"] = hashlib.sha256(geometry.read_bytes()).hexdigest()
                value["swept_integral"] = swept_integral(geometry)
            summary[name].append(value)
            (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            print(name, case, value["status"][0], round(value["median_ms"], 3),
                  "ms, rows", runs[0].get("active_constraints"), "pruned", runs[0].get("pruned_supports"), flush=True)


if __name__ == "__main__":
    main()
