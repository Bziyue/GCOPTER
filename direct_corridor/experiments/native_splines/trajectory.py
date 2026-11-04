#!/usr/bin/env python3
"""Replay certified frozen geometry through the unchanged common optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=".scratch/native_splines/build/native_trajectory_comparison")
    parser.add_argument("--geometry", default=".scratch/native_splines/final")
    parser.add_argument("--output", default=".scratch/native_splines/trajectory")
    parser.add_argument("--models", default="H5-S,C3-OPT-native,C3-OPT-project,C3-NBC-project,H5-C3-T")
    parser.add_argument("--cases", default="0,1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quintic-replay", action="store_true")
    parser.add_argument("--profile", action="store_true", help="separate diagnostic run with per-call timers")
    args = parser.parse_args()
    inputs = json.loads((ROOT / "docs/validation/native_splines/frozen_manifest.json").read_text())["inputs"]
    binary = ROOT / args.binary
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ | {k: "1" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}
    env["LD_LIBRARY_PATH"] = str(ROOT / ".scratch/ompl/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
    env |= {"GCOPTER_TUBE_MODE": "fast", "GCOPTER_PROFILE": "1" if args.profile else "0"}
    manifest = {"schema": "native-splines-trajectory-v1", "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "corridor_time_is_replay_and_audit": True, "trajectory_representation": "common-quintic-minjerk",
                "repeats": args.repeats, "max_iterations": 800, "constraint": "bezier", "depth": 2,
                "spacing": 1., "penalty_weight": 1e5, "integration_steps": 16, "cases": []}
    manifest["profile_enabled"] = args.profile
    manifest["native_cubic_query"] = not args.quintic_replay
    if args.quintic_replay: env["GCOPTER_QUINTIC_REPLAY"] = "1"
    else: env.pop("GCOPTER_QUINTIC_REPLAY", None)
    for case in map(int, args.cases.split(",")):
        row = inputs[case]
        for kind in ("seed", "map"):
            if hashlib.sha256((ROOT / row[kind]).read_bytes()).hexdigest() != row[kind+"_sha256"]:
                raise RuntimeError("frozen input modified")
        for model in args.models.split(","):
            geometry = ROOT / args.geometry / f"case_{case:02d}" / f"{model}.csv"
            dest = output / model / f"case_{case:02d}"
            dest.mkdir(parents=True, exist_ok=True)
            if not geometry.exists():
                manifest["cases"].append({"case": case, "model": model, "status": "no_certified_geometry"})
                continue
            env |= {"GCOPTER_FROZEN_GEOMETRY": str(geometry), "GCOPTER_SHAPE_MODE": model}
            command = ["/usr/bin/time", "-f", "%M", "-o", str(dest / "peak_rss_kib.txt"), str(binary),
                       str(ROOT / row["seed"]), str(ROOT / row["map"]), str(dest),
                       *map(str, row["bounds_lower"]+row["bounds_upper"]), ".25", str(args.repeats),
                       "1.0", "100000", "16", "1", "800", "bezier", "2"]
            start = time.monotonic()
            with (dest / "process.log").open("w") as log:
                try:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                    status = "completed" if result.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            record = row | {"case": case, "model": model, "status": status, "elapsed_s": time.monotonic()-start,
                            "geometry_sha256": hashlib.sha256(geometry.read_bytes()).hexdigest(),
                            "output": str(dest.relative_to(ROOT))}
            manifest["cases"].append(record)
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print(f"{model} case {case}: {status}", flush=True)

if __name__ == "__main__":
    main()
