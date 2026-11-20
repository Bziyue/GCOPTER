#!/usr/bin/env python3
"""Paired H3 union volumes, FIRI intersections and generator timings."""
import argparse
import csv
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".scratch/h3_inflation"))
    parser.add_argument("--firi", type=Path, default=Path("build/gcopter/h3_firi_overlap"))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--samples", type=int, default=65536, help="Sobol samples per shift, eight shifts")
    parser.add_argument("--measure-only", action="store_true", help="measure existing generation outputs")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here.parents[2])
    inputs = Path("docs/validation/h3_volume")
    modes = ["h3_adaptive_baseline", "h3_pre_core_spheres", "h3_pre_core"]
    if not args.firi.is_file():
        parser.error("build h3_firi_overlap with GCOPTER_BUILD_CORRIDOR_BENCHMARK=ON or pass --firi")
    os.sched_setaffinity(0, {2})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    os.environ.pop("SWEEP_PROFILE", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if not args.measure_only:
        with tarfile.open(inputs / "inputs.tar.gz") as archive:
            archive.extractall(".scratch/h3_volume/inputs", filter="data")
        subprocess.run([sys.executable, here / "run.py", "--inputs", inputs / "inputs.json",
                        "--output", args.output, "--cases", args.cases, "--modes", ",".join(modes),
                        "--repeats", str(args.repeats), "--samples", "0"], check=True)
    frozen = json.loads((inputs / "inputs.json").read_text())["cases"]
    resolutions = {r["map_sha256"]: r["resolution"] for r in frozen if "resolution" in r}
    manifest = json.loads((args.output / "manifest.json").read_text())
    cases = manifest["cases"]
    runs = {(r["case"], r["mode"]): r for r in json.loads((args.output / "summary.json").read_text())}
    if args.measure_only and (cases[0]["case"], "h3_new") in runs:
        modes = ["h3_adaptive_baseline", "h3_spheres", "h3_new"]
    table, measurements = [], []
    for case in cases:
        number = case["case"]
        rows = [runs[number, mode] for mode in modes]
        valid = [r for r in rows if r["success"]]
        files = [Path(r["output"]) / "corridor.csv" for r in valid]
        row = {"case": number, "family": case["family"]}
        detail = {"case": number}
        labels = dict(zip(modes, ("old", "sphere", "voxel")))
        for r in rows:
            label = labels[r["mode"]]
            if r["returncode"] != 0 or len(r["runs"]) != manifest["repeats"]:
                raise RuntimeError(f"case {number}, {r['mode']} failed in {r['output']}")
            statuses = {sample["status"] for sample in r["runs"]}
            if len(statuses) != 1:
                raise RuntimeError(f"case {number}, {r['mode']} changed status between repetitions")
            row[label + "_status"] = statuses.pop()
            row[label + "_ms"] = r["median_ms"]
            row[label + "_index_ms"] = r["runs"][0]["index_ms"]
            if r["success"] and label != "old":
                assert r["runs"][0]["obstacle_audit"]
        if rows[0]["success"]:
            for r in valid:
                if r is rows[0]:
                    continue
                label = labels[r["mode"]]
                pair = json.loads(subprocess.check_output([
                    ".scratch/h3/measure", Path(rows[0]["output"]) / "corridor.csv",
                    Path(r["output"]) / "corridor.csv", case["seed"], str(args.samples)], text=True))
                assert pair["unresolved"] == 0
                detail[label + "_union_vs_old"] = pair
                row[label + "_union_gain_pct"] = 100 * pair["paired_delta"] / pair["baseline"]
                if label == "voxel":
                    row.update(old_union_m3=pair["baseline"], voxel_union_m3=pair["candidate"],
                               gain_ci95_m3=pair["paired_ci95"])
        command = [args.firi, case["seed"], case["map"], *map(str, case["bounds_lower"]+case["bounds_upper"]),
                   str(resolutions[case["map_sha256"]]), str(args.repeats), str(args.samples), *files]
        proc = subprocess.run(command, text=True, capture_output=True)
        if proc.returncode:
            detail["firi_error"] = proc.stderr.strip()
            row["firi_status"] = "FAILED"
        else:
            firi = json.loads(proc.stdout)
            detail["firi"] = firi
            row["firi_ms"] = statistics.median(firi["firi_ms"])
            row["firi_status"] = "CONNECTED" if firi["connected"] else "DISCONNECTED"
            for r, comparison in zip(valid, firi["comparisons"]):
                volume = comparison["firi"]
                row["firi_union_m3"] = volume
                if volume > 0:
                    row[labels[r["mode"]] + "_firi_covered_pct"] = 100 * comparison["intersection"] / volume
        measurements.append(detail)
        table.append(row)
    (args.output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    manifest.update(samples_per_shift=args.samples, firi_repeats=args.repeats,
                    firi_sha256=hashlib.sha256(args.firi.read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fields = dict.fromkeys(key for row in table for key in row)
    with (args.output / "paired.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
