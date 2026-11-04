#!/usr/bin/env python3
"""Serial continuation, knot-density and swept-volume experiments."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess

from explore import swept_integral

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIGURATIONS = {
    "reference": {},
    "predict_half": {"PATH_PREDICTOR": .5},
    "predict_one": {"PATH_PREDICTOR": 1},
    "predict_one_c3": {"PATH_PREDICTOR": 1, "CORRECTOR_ITERATIONS": 3},
    "volume1": {"VOLUME_REFINEMENTS": 1},
    "volume2": {"VOLUME_REFINEMENTS": 2},
    "adaptive_volume1": {"ADAPTIVE_KNOTS": 2, "VOLUME_REFINEMENTS": 1},
    "adaptive_volume2": {"ADAPTIVE_KNOTS": 2, "VOLUME_REFINEMENTS": 2},
    "adaptive_c5": {"ADAPTIVE_KNOTS": 2, "CORRECTOR_ITERATIONS": 5},
    "predict_c3_volume": {"PATH_PREDICTOR": 1, "CORRECTOR_ITERATIONS": 3, "VOLUME_REFINEMENTS": 1, "VOLUME_TRUST": .5},
    "predict_c3_volume_large": {"PATH_PREDICTOR": 1, "CORRECTOR_ITERATIONS": 3, "VOLUME_REFINEMENTS": 1, "VOLUME_TRUST": 1},
    "predict_i7_c3_volume": {"PATH_PREDICTOR": 1, "INITIAL_ITERATIONS": 7, "ITERATIONS": 7, "CORRECTOR_ITERATIONS": 3, "VOLUME_REFINEMENTS": 1, "VOLUME_TRUST": .5},
    "volume_large": {"VOLUME_REFINEMENTS": 1, "VOLUME_TRUST": .5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".scratch/native_splines/round4/continuation")
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument("--cases", default="0,1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    # This limit applies to each serial child, including its NumPy runner.
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.configurations.split(","):
        environment = {k: v for k, v in os.environ.items() if not k.startswith("NATIVE_")}
        configuration = {"LAZY_PRIORITIES": 1, "CACHE_VALUES": 1, "REUSE_ENERGY": 1, "ADAPTIVE_KNOTS": 0,
                         "PATH_PREDICTOR": 0, "VOLUME_REFINEMENTS": 0}
        configuration.update(CONFIGURATIONS[name])
        environment |= {"NATIVE_"+k: str(v) for k, v in configuration.items()}
        target = out/name
        subprocess.run(["python3", str(HERE/"run.py"), "--native", "--preset", "balanced",
                        "--generation-only", "--method-count", "1", "--repeats", str(args.repeats),
                        "--samples", "0", "--cases", args.cases, "--output", str(target)],
                       cwd=ROOT, env=environment, check=True)
        values = []
        for case in map(int, args.cases.split(",")):
            folder = target/f"case_{case:02d}"
            rows = [json.loads(s) for s in (folder/"rows.jsonl").read_text().splitlines()]
            runs = [r for r in rows if r.get("kind") == "generation" and
                    r["model"] == "C3-OPT-native" and r["repeat"] >= 0]
            sample = runs[0]
            record = {"case": case, "status": sample["status"], "sample": sample,
                      "median_ms": statistics.median(r["total_ms"] for r in runs)}
            if sample["status"] == "SUCCESS":
                record["swept_integral"] = swept_integral(folder/"C3-OPT-native.csv")
            values.append(record)
        summary[name] = values
        (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(name, [(r["case"], round(r["median_ms"], 2), round(r.get("swept_integral", 0), 2))
                     for r in values], flush=True)


if __name__ == "__main__":
    main()
