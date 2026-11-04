#!/usr/bin/env python3
"""Serial, frozen-input runner. Failed requests stay in the manifest."""
import argparse
import hashlib
import json
import os
import pathlib
import resource
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="0,1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--output", default=".scratch/native_splines/representation")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--binary", default=".scratch/native_splines/build/native_spline_benchmark")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--method-count", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    frozen = json.loads((ROOT / "docs/validation/native_splines/frozen_manifest.json").read_text())
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "native-splines-run-v1", "baseline": frozen["baseline"],
                "phase": "A-fixed-nodal-geometry", "repeats": args.repeats,
                "samples_per_shift": args.samples, "random_shifts": 8, "cases": []}
    manifest["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest["worktree_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    binary = ROOT / args.binary
    manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest["sources_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for tree in (pathlib.Path(__file__).parent, ROOT / "direct_corridor/src", ROOT / "direct_corridor/include")
        for p in tree.rglob("*")
        if p.is_file() and p.suffix in {".hpp", ".cpp", ".py", ".txt"}}
    cache = binary.parent / "CMakeCache.txt"
    manifest["build_config"] = [line for line in cache.read_text().splitlines()
                                if line.startswith(("CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE:", "NATIVE_SPLINE_NATIVE_ARCH:"))] if cache.exists() else []
    manifest["geometry_format"] = "native_sweep_v3_degree_explicit_bezier"
    env = os.environ | {k: "1" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}
    if args.native:
        env["NATIVE_CANDIDATES"] = "1"
        manifest["phase"] = "A-and-B-native-plus-projection"
    env["NATIVE_METHOD_COUNT"] = str(args.method_count)
    if args.profile: env["NATIVE_PROFILE"] = "1"
    if args.generation_only: env["NATIVE_GENERATION_ONLY"] = "1"
    manifest["native_environment"] = {k: v for k, v in env.items() if k.startswith("NATIVE_")}
    for i in map(int, args.cases.split(",")):
        row = frozen["inputs"][i]
        for kind in ("seed", "map"):
            p = ROOT / row[kind]
            if hashlib.sha256(p.read_bytes()).hexdigest() != row[kind + "_sha256"]:
                raise RuntimeError(f"input changed: {p}")
        target = out / f"case_{i:02d}"
        target.mkdir(exist_ok=True)
        command = [str(binary),
                   str(ROOT / row["seed"]), str(ROOT / row["map"]), str(target),
                   *map(str, row["bounds_lower"] + row["bounds_upper"]), str(args.repeats), str(args.samples)]
        start = time.monotonic()
        with (target / "process.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "completed" if result.returncode == 0 else "failed"
                code = result.returncode
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
        record = row | {"status": status, "returncode": code, "elapsed_s": time.monotonic() - start,
                        "command": command, "output": str(target.relative_to(ROOT))}
        manifest["cases"].append(record)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"case {i}: {status}, {record['elapsed_s']:.2f} s", flush=True)

if __name__ == "__main__":
    main()
