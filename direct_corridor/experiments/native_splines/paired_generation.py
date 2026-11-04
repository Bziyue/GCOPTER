#!/usr/bin/env python3
"""Alternate frozen-reference and candidate processes on one CPU core."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".scratch/native_splines/round4")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    out = ROOT/args.output
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.rglob("*")
               if p.is_file() and p.suffix in (".hpp", ".cpp", ".py", ".txt")}
    manifests = {}
    for name in ("baseline", "final"):
        folder = out/name
        manifests[name] = json.loads((folder/"manifest.json").read_text())
        folder.rename(out/(name+"_unpinned"))
        folder.mkdir()
        m = manifests[name]
        m["cases"] = []
        m["repeats"] = args.repeats
        m["cpu_affinity"] = [args.cpu]
        m["order"] = "alternate baseline/final order by frozen case"
        m["address_space_limit_bytes"] = 4*1024**3
        if name == "final":
            m["sources_sha256"] = sources
    inputs = json.loads((ROOT/"docs/validation/native_splines/frozen_manifest.json").read_text())["inputs"]
    binaries = {"baseline": ROOT/".scratch/native_splines/round4_baseline/native_spline_benchmark",
                "final": ROOT/".scratch/native_splines/build/native_spline_benchmark"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("NATIVE_")}
    env |= {"NATIVE_CANDIDATES": "1", "NATIVE_PRESET": "balanced", "NATIVE_GENERATION_ONLY": "1",
            "NATIVE_METHOD_COUNT": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for i, row in enumerate(inputs):
        for kind in ("seed", "map"):
            assert hashlib.sha256((ROOT/row[kind]).read_bytes()).hexdigest() == row[kind+"_sha256"]
        order = ("baseline", "final") if i % 2 == 0 else ("final", "baseline")
        for name in order:
            binary = binaries[name]
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifests[name]["binary_sha256"]
            target = out/name/f"case_{i:02d}"
            target.mkdir()
            command = [str(binary), str(ROOT/row["seed"]), str(ROOT/row["map"]), str(target),
                       *map(str, row["bounds_lower"]+row["bounds_upper"]), str(args.repeats), "0"]
            started = time.monotonic()
            with (target/"process.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            manifests[name]["cases"].append(row | {"status": "completed" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "elapsed_s": time.monotonic()-started,
                "command": command, "output": str(target.relative_to(ROOT))})
            (out/name/"manifest.json").write_text(json.dumps(manifests[name], indent=2)+"\n")
            result.check_returncode()
            expected = out/"joint"/f"case_{i:02d}"/("C3-before.csv" if name == "baseline" else "C3-after.csv")
            assert (target/"C3-OPT-native.csv").read_bytes() == expected.read_bytes()
            print(f"{name} case {i}: {args.repeats} repetitions, identical geometry", flush=True)


if __name__ == "__main__":
    main()
