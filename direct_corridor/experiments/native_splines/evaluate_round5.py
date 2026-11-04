#!/usr/bin/env python3
"""Serial union-volume, independent geometry and common-trajectory checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=".scratch/native_splines/round5")
    parser.add_argument("--samples", type=int, default=262144)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--trajectories", action="store_true", help="optional historical end-to-end comparison")
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    work = ROOT/args.work
    joint = work/"joint"
    joint.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((ROOT/"docs/validation/native_splines/frozen_manifest.json").read_text())["inputs"]
    configurations = {"C3-before": "frozen", "C3-after": "axis_derivatives",
                      "C3-pruned": "projected_axis_derivatives", "H5-S": "axis_derivatives"}
    manifest = {"cases": [], "configurations": configurations, "same_backend_for_all_models": True,
                "cpu": args.cpu, "address_space_limit_bytes": 4*1024**3}
    for row in inputs:
        target = joint/f"case_{row['case']:02d}"
        target.mkdir(exist_ok=True)
        hashes = {}
        for name, config in configurations.items():
            source = work/"formal"/config/target.name/("H5-S.csv" if name == "H5-S" else "C3-OPT-native.csv")
            shutil.copy2(source, target/(name+".csv"))
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashes["C3-before"] != hashes["C3-after"]:
            raise RuntimeError("exact-kernel candidate changed frozen geometry")
        manifest["cases"].append(row | {"output": str(target.relative_to(ROOT)), "geometry_sha256": hashes})
    (joint/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    binary = ROOT/".scratch/native_splines/round5_baseline/native_volume_refine"
    volume_manifest = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                       "samples_per_shift": args.samples, "shifts": 8, "cases": []}
    # C3-after is byte-identical to C3-before, so its volume is exactly equal.
    # Include the other geometries in one common QMC domain and sequence.
    for row in manifest["cases"]:
        target = ROOT/row["output"]
        command = [str(binary), str(ROOT/row["seed"]), str(target), str(args.samples),
                   "C3-before,H5-S,C3-pruned"]
        start = time.monotonic()
        with (target/"volume.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        volume_manifest["cases"].append({"case": row["case"], "command": command,
                                         "elapsed_s": time.monotonic()-start})
        (joint/"volume_manifest.json").write_text(json.dumps(volume_manifest, indent=2)+"\n")
        print(f"union volume case {row['case']}: completed", flush=True)

    env = os.environ.copy()
    dependencies = ROOT/".scratch/native_splines/audit_dependencies/usr/lib/python3/dist-packages"
    if dependencies.exists():
        env["PYTHONPATH"] = str(dependencies)+os.pathsep+env.get("PYTHONPATH", "")
    script = ROOT/"direct_corridor/experiments/native_splines"
    subprocess.run([sys.executable, str(script/"independent_audit.py"), "--geometry", str(joint),
                    "--models", ",".join(configurations), "--output", str(work/"independent_audit.json")],
                   cwd=ROOT, env=env, check=True, timeout=600)
    if args.trajectories:
        subprocess.run([sys.executable, str(script/"trajectory.py"), "--geometry", str(joint),
                    "--binary", ".scratch/native_splines/round5_baseline/native_trajectory_comparison",
                    "--output", str(work/"trajectory"), "--models", ",".join(configurations), "--repeats", "3"],
                   cwd=ROOT, check=True, timeout=3600)


if __name__ == "__main__":
    main()
