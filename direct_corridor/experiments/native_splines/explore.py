#!/usr/bin/env python3
"""Serial ablations of cubic assembly and warm barrier continuation."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def swept_integral(path):
    """Exact degree-eight Jacobian integral; overlaps retain multiplicity."""
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    q = a[:, 3:].reshape(-1, 4, 9)
    t, weights = np.polynomial.legendre.leggauss(5)
    t, weights = (t + 1) / 2, weights / 2
    basis = np.array([math.comb(3, i)*t**i*(1-t)**(3-i) for i in range(4)]).T
    derivative = np.array([math.comb(2, i)*t**i*(1-t)**(2-i) for i in range(3)]).T
    value = basis @ q
    tangent = derivative @ (3*np.diff(q, axis=1))
    return float(math.pi*np.sum(weights*np.sum(np.cross(value[:, :, 3:6], value[:, :, 6:9])*tangent[:, :, :3], axis=2)))

CONFIGURATIONS = {
    "tensor": {},
    "trial": {"TRIAL_FIRST": 1},
    "warm1": {"TRIAL_FIRST": 1, "WARM_STAGE": 1},
    "warm1_c5": {"TRIAL_FIRST": 1, "WARM_STAGE": 1, "CORRECTOR_ITERATIONS": 5},
    "warm1_c6": {"TRIAL_FIRST": 1, "WARM_STAGE": 1, "CORRECTOR_ITERATIONS": 6},
    "warm2_c6": {"TRIAL_FIRST": 1, "WARM_STAGE": 2, "CORRECTOR_ITERATIONS": 6},
    "warm1_i6_c5": {"TRIAL_FIRST": 1, "WARM_STAGE": 1, "INITIAL_ITERATIONS": 6, "CORRECTOR_ITERATIONS": 5},
    "warm2_c8": {"TRIAL_FIRST": 1, "WARM_STAGE": 2, "CORRECTOR_ITERATIONS": 8},
    "polar": {"TRIAL_FIRST": 1, "WARM_STAGE": 1, "CORRECTOR_ITERATIONS": 5, "POLAR_PRUNING": 1},
    "working_set": {"TRIAL_FIRST": 1, "WARM_STAGE": 1, "CORRECTOR_ITERATIONS": 5, "WORKING_SET": 1},
    "reserve1": {"WARM_STAGE": 1, "WARM_RESERVE": .01},
    "reserve3": {"WARM_STAGE": 1, "WARM_RESERVE": .03},
    "reserve5": {"WARM_STAGE": 1, "WARM_RESERVE": .05},
    "reserve10": {"WARM_STAGE": 1, "WARM_RESERVE": .1},
    "reserve3_c6": {"WARM_STAGE": 1, "WARM_RESERVE": .03, "CORRECTOR_ITERATIONS": 6},
    "reserve5_c5": {"WARM_STAGE": 1, "WARM_RESERVE": .05, "CORRECTOR_ITERATIONS": 5},
    "reserve5_last": {"WARM_STAGE": 2, "WARM_RESERVE": .05, "CORRECTOR_ITERATIONS": 6},
    "compact": {"TENSOR_ASSEMBLY": 0},
    "progress1": {"TENSOR_ASSEMBLY": 0, "PROGRESS_SAMPLES": 1},
    "progress1_full": {"TENSOR_ASSEMBLY": 0, "PROGRESS_SAMPLES": 1, "PROGRESS_DEPTH": 0},
    "progress3_full": {"TENSOR_ASSEMBLY": 0, "PROGRESS_DEPTH": 0},
    "progress1_warm": {"TENSOR_ASSEMBLY": 0, "PROGRESS_SAMPLES": 1, "WARM_STAGE": 1},
    "line": {"TENSOR_ASSEMBLY": 0, "CACHED_LINE": 1},
    "line_full": {"TENSOR_ASSEMBLY": 0, "CACHED_LINE": 1, "PROGRESS_DEPTH": 0},
    "inflate1": {"TENSOR_ASSEMBLY": 0, "COORDINATE_SWEEPS": 1},
    "inflate2": {"TENSOR_ASSEMBLY": 0, "COORDINATE_SWEEPS": 2},
    "inflate_full": {"TENSOR_ASSEMBLY": 0, "PROGRESS_DEPTH": 0, "COORDINATE_SWEEPS": 2},
    "inflate_warm": {"TENSOR_ASSEMBLY": 0, "WARM_STAGE": 1, "COORDINATE_SWEEPS": 2},
    "inflate_fast": {"TENSOR_ASSEMBLY": 0, "ITERATIONS": 6, "INITIAL_ITERATIONS": 6, "COORDINATE_SWEEPS": 2},
    "inflate_fast_c3": {"TENSOR_ASSEMBLY": 0, "ITERATIONS": 6, "INITIAL_ITERATIONS": 6, "CORRECTOR_ITERATIONS": 3, "COORDINATE_SWEEPS": 2},
    "shape07": {"TENSOR_ASSEMBLY": 0, "PROGRESS_DEPTH": 0, "SHAPE_ENERGY": 7e-6},
    "shape05": {"TENSOR_ASSEMBLY": 0, "PROGRESS_DEPTH": 0, "SHAPE_ENERGY": 5e-6},
    "shape03": {"TENSOR_ASSEMBLY": 0, "PROGRESS_DEPTH": 0, "SHAPE_ENERGY": 3e-6},
    "energy07": {"TENSOR_ASSEMBLY": 0, "PROGRESS_DEPTH": 0, "ENERGY": 7e-6},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".scratch/native_splines/round3/schedule")
    parser.add_argument("--cases", default="0,1,2,3,4,5,6")
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument("--repeats", default=1, type=int)
    args = parser.parse_args()
    summary = {}
    for name in args.configurations.split(","):
        target = ROOT/args.output/name
        env = {k: v for k, v in os.environ.items() if not k.startswith("NATIVE_")}
        # Freeze the pre-default-change ablation base explicitly.
        settings = {"TENSOR_ASSEMBLY": 1, "CACHED_LINE": 0, "PROGRESS_DEPTH": 1}
        if name.startswith(("inflate", "shape", "energy")):
            settings["CACHED_LINE"] = 1
        settings.update(CONFIGURATIONS[name])
        env |= {"NATIVE_"+k: str(v) for k, v in settings.items()}
        subprocess.run(["python3", str(HERE/"run.py"), "--native", "--generation-only", "--method-count", "1",
                        "--preset", "balanced", "--repeats", str(args.repeats), "--samples", "0",
                        "--cases", args.cases, "--output", str(target)], env=env, cwd=ROOT, check=True)
        values = []
        for case in map(int, args.cases.split(",")):
            folder = target/f"case_{case:02d}"
            rows = [json.loads(line) for line in (folder/"rows.jsonl").read_text().splitlines()]
            rows = [r for r in rows if r.get("kind") == "generation" and r.get("model") == "C3-OPT-native" and r["repeat"] >= 0]
            sample = rows[0]
            record = {"case": case, "status": sample["status"], "median_ms": statistics.median(r["total_ms"] for r in rows), "sample": sample}
            if sample["status"] == "SUCCESS":
                record["swept_integral"] = swept_integral(folder/"C3-OPT-native.csv")
            values.append(record)
        summary[name] = values
        (ROOT/args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(name, [(r["case"], r["status"], round(r["median_ms"], 2), round(r.get("swept_integral", 0), 3)) for r in values], flush=True)

if __name__ == "__main__":
    main()
