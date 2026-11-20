#!/usr/bin/env python3
"""Measure two certified H3 generations from run.py on identical frozen inputs."""
import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def run_json(command):
    return json.loads(subprocess.check_output(command, text=True))


def change(before, after):
    return 100 * (after / before - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="run.py output with h3_pre_core and h3_new")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", default="h3_pre_core")
    parser.add_argument("--candidate", default="h3_new")
    parser.add_argument("--replicate", type=Path, help="second timing run on the same binaries and inputs, with mode order reversed")
    parser.add_argument("--samples", default=65536, type=int, help="Sobol points per shift")
    parser.add_argument("--workers", default=2, type=int, help="concurrent offline measurements")
    parser.add_argument("--firi", type=Path, default=Path("build/gcopter/h3_firi_overlap"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here.parents[2])
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(here.parent / "free_boundary"))
    from geometry_metrics import bending

    config = read_json(args.root / "manifest.json")
    runs = {(r["case"], r["mode"]): r for r in read_json(args.root / "summary.json")}
    repeat_counts = [config["repeats"]]
    if args.replicate:
        second = read_json(args.replicate / "manifest.json")
        for key in ("cases", "binaries", "cpu", "memory_limit"):
            if config[key] != second[key]:
                raise ValueError(f"Timing sessions disagree on {key}")
        repeat_counts.append(second["repeats"])
        replica = {(r["case"], r["mode"]): r for r in read_json(args.replicate / "summary.json")}
        if runs.keys() != replica.keys():
            raise ValueError("Timing sessions contain different cases or modes")
        for key, r in runs.items():
            twin = replica[key]
            if (r["success"] != twin["success"] or r.get("geometry_sha256") != twin.get("geometry_sha256") or
                twin["returncode"] or len(r["runs"]) != repeat_counts[0] or len(twin["runs"]) != repeat_counts[1]):
                raise ValueError(f"Timing replica did not reproduce {key}")
            r["runs"] += twin["runs"]
            r["median_ms"] = statistics.median(x["wall_ms"] for x in r["runs"])
    cases = config["cases"]
    resolutions = {r["map_sha256"]: r["resolution"] for r in cases if "resolution" in r}
    measure = Path(".scratch/h3/measure")

    def wall(path):
        previous = bending(path, 24, 48)["surface"]
        for order in (48, 96, 192, 384, 768):
            current = bending(path, order, 2 * order)["surface"]
            error = abs(change(previous, current))
            if error <= .5:
                return current, error
            previous = current
        raise ValueError(f"Core wall quadrature did not converge for {path}")

    def compare(case):
        number = case["case"]
        old, new = (runs[number, m] for m in (args.reference, args.candidate))
        row = dict(case=number, family=case["family"])
        for label, run in (("before", old), ("after", new)):
            samples = run["runs"]
            if (run["returncode"] or len(samples) != sum(repeat_counts) or
                    len({r["status"] for r in samples}) != 1):
                raise ValueError(f"Incomplete or unstable {label} run for case {number}")
            row[label + "_status"] = samples[0]["status"]
            row[label + "_ms"] = run["median_ms"]
            if run["success"]:
                if not samples[0].get("obstacle_audit") or not all(
                    all(r[key] for key in ("support_pass", "jacobian_pass", "anchors_pass")) for r in samples
                ):
                    raise ValueError(f"Missing certificate for case {number}")
                row[label + "_sha256"] = run["geometry_sha256"]
        row["time_pct"] = change(old["median_ms"], new["median_ms"])
        if not (old["success"] and new["success"]):
            return row
        sample = new["runs"][0]
        for key in ("intermediate_radius_used", "area_selected_spans", "core_smoothed",
                    "c1_error", "jacobian_relative_lower_bound", "core_swept_volume", "swept_volume"):
            row[key] = sample[key]
        before, after = Path(old["output"]), Path(new["output"])
        for label, file in (("full", "corridor.csv"), ("core", "core.csv")):
            v = run_json([measure, before / file, after / file, case["seed"], str(args.samples)])
            if v["unresolved"] or v["samples"] != 8 * args.samples:
                raise ValueError(f"Incomplete {label} QMC measurement for case {number}")
            row["before_" + label + "_m3"], row["after_" + label + "_m3"] = v["baseline"], v["candidate"]
            row[label + "_pct"] = 100 * v["paired_delta"] / v["baseline"]
            row[label + "_ci95_pct"] = 100 * v["paired_ci95"] / v["baseline"]
        firi = run_json([args.firi, case["seed"], case["map"], *map(str, case["bounds_lower"] + case["bounds_upper"]),
                         str(case.get("resolution", resolutions.get(case["map_sha256"], .25))), "1", str(args.samples),
                         before / "corridor.csv", after / "corridor.csv"])
        row["firi_connected"] = firi["connected"]
        a, b = firi["comparisons"]
        if firi["connected"] and a["firi"] > 0:
            if a["firi"] != b["firi"]:
                raise ValueError(f"Unpaired FIRI domains for case {number}")
            delta = [100 * (y-x) / a["firi"] for x, y in zip(a["intersection_shifts"], b["intersection_shifts"])]
            row["before_firi_pct"], row["after_firi_pct"] = (100 * r["intersection"] / a["firi"] for r in (a,b))
            row["firi_delta_pp"] = statistics.mean(delta)
            row["firi_ci95_pp"] = 2.364624251 * statistics.stdev(delta) / len(delta)**.5
        initial, err_a = wall(before / "core.csv")
        final, err_b = wall(after / "core.csv")
        row.update(before_wall=initial, after_wall=final, wall_pct=change(initial, final),
                   wall_quadrature_change_pct=max(err_a, err_b))
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(compare, cases))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dict.fromkeys(k for r in rows for k in r))
        writer.writeheader()
        writer.writerows(rows)
    paired = [r for r in rows if "core_pct" in r]
    stats = {}
    for name, group in (("paired", paired), ("intermediate", [r for r in paired if r["intermediate_radius_used"]])):
        stats[name] = {}
        for key in ("time_pct", "full_pct", "core_pct", "firi_delta_pp", "wall_pct", "before_ms", "after_ms"):
            values = [r[key] for r in group if key in r]
            if values:
                stats[name][key] = dict(count=len(values), median=statistics.median(values),
                                        minimum=min(values), maximum=max(values), mean=statistics.mean(values))
    details = dict(reference=args.reference, candidate=args.candidate, cases=len(rows), paired_success=len(paired), samples_per_shift=args.samples, shifts=8,
                   repetitions_per_session=repeat_counts, cpu=config["cpu"], compiler=config["compiler"], commit=config["commit"],
                   input_manifest_sha256=hashlib.sha256(Path("docs/validation/h3_volume/inputs.json").read_bytes()).hexdigest(),
                   sources=config["sources"], binaries={Path(k).name:v for k,v in config["binaries"].items()},
                   firi_sha256=hashlib.sha256(args.firi.read_bytes()).hexdigest(), statistics=stats)
    details["offline_source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__).resolve().relative_to(Path.cwd()), (here.parent / "free_boundary/geometry_metrics.py").relative_to(Path.cwd()))}
    details["wall_ms"] = {str(r["case"]): {m: [v["wall_ms"] for v in runs[r["case"], m]["runs"]]
                           for m in (args.reference, args.candidate)} for r in rows}
    (args.output / "summary.json").write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    main()
