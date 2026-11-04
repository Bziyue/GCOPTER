#!/usr/bin/env python3
"""Compare corridor volume and geometry only; no trajectory optimization."""
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
WORK = ROOT/".scratch/native_splines/round5"
SOURCES = {"C3-before": ("combined_screen", "frozen"),
           "C3-pruned": ("corridor_screen", "projected_axis_bounded"),
           "C3-predict": ("combined_screen", "pruned_predict3"),
           "C3-volume": ("corridor_screen", "pruned_volume1"),
           "C3-selected": ("combined_screen", "pruned_predict3_volume1"),
           "H5-S": ("combined_screen", "frozen")}


def main():
    os.sched_setaffinity(0, {2})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    inputs = json.loads((ROOT/"docs/validation/native_splines/frozen_manifest.json").read_text())["inputs"]
    joint = WORK/"corridor_joint"
    joint.mkdir(exist_ok=True)
    manifest = {"selection_criteria": ["generation_time", "union_volume", "geometric_regularity", "continuous_safety"],
                "trajectory_optimization": False, "sources": SOURCES, "cases": []}
    for row in inputs:
        target = joint/f"case_{row['case']:02d}"
        target.mkdir(exist_ok=True)
        hashes = {}
        for model, (screen, config) in SOURCES.items():
            source = WORK/screen/config/target.name/("H5-S.csv" if model == "H5-S" else "C3-OPT-native.csv")
            shutil.copy2(source, target/(model+".csv"))
            hashes[model] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest["cases"].append(row | {"output": str(target.relative_to(ROOT)), "geometry_sha256": hashes})
    (joint/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    binary = ROOT/".scratch/native_splines/round5_baseline/native_volume_refine"
    record = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "default_samples_per_shift": 262144,
              "high_resolution_cases": {"3": 1048576},
              "shifts": 8, "cpu": 2, "address_space_limit_bytes": 4*1024**3, "cases": []}
    for row in manifest["cases"]:
        target = ROOT/row["output"]
        samples=1048576 if row["case"]==3 else 262144
        command = [str(binary), str(ROOT/row["seed"]), str(target), str(samples), ",".join(SOURCES)]
        start = time.monotonic()
        with (target/"volume.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        record["cases"].append({"case": row["case"], "command": command, "elapsed_s": time.monotonic()-start})
        (joint/"volume_manifest.json").write_text(json.dumps(record, indent=2)+"\n")
        print(f"corridor union volume case {row['case']}: completed", flush=True)
    env = os.environ.copy()
    dependencies = ROOT/".scratch/native_splines/audit_dependencies/usr/lib/python3/dist-packages"
    if dependencies.exists():
        env["PYTHONPATH"] = str(dependencies)+os.pathsep+env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(Path(__file__).with_name("independent_audit.py")),
                    "--geometry", str(joint), "--models", ",".join(SOURCES),
                    "--output", str(WORK/"corridor_audit.json")], cwd=ROOT, env=env, check=True, timeout=600)


if __name__ == "__main__":
    main()
