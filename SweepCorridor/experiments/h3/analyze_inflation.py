#!/usr/bin/env python3
"""Measure the anchor-to-anchor geometry and archive the paired H3 experiment."""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".scratch/h3_inflation"))
    parser.add_argument("--output", type=Path, default=Path("docs/validation/h3_inflation"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    os.chdir(here.parents[2])
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str(here.parent / "free_boundary"))
    from geometry_metrics import bending

    def curvature(file):
        coarse = bending(file, 24, 48)
        fine = bending(file, 48, 96)
        order = 48
        if abs(fine["surface"] / coarse["surface"] - 1) > .01:
            coarse, fine = fine, bending(file, 96, 192)
            order = 96
        return dict(fine, order=order, relative_change=abs(fine["surface"] / coarse["surface"] - 1))

    data = {name: json.loads((args.root / (name + ".json")).read_text())
            for name in ("manifest", "summary", "measurements")}
    config = data["manifest"]
    cases = {r["case"]: r for r in config["cases"]}
    lookup = {(r["case"], r["mode"]): r for r in data["summary"]}
    data["core"] = {}
    with (args.root / "paired.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row["old_status"] != "SUCCESS" or row["voxel_status"] != "SUCCESS":
            continue
        number = int(row["case"])
        voxel_mode = "h3_pre_core" if (number, "h3_pre_core") in lookup else "h3_new"
        files = [Path(lookup[number, name]["output"]) / "core.csv"
                 for name in ("h3_adaptive_baseline", voxel_mode)]
        pair = json.loads(subprocess.check_output([
            ".scratch/h3/measure", *files, cases[number]["seed"], str(config["samples_per_shift"])
        ], text=True))
        if pair["unresolved"]:
            raise RuntimeError(f"unresolved core membership in case {number}")
        old, new = map(curvature, files)
        data["core"][number] = dict(union=pair, old_bending=old, voxel_bending=new)
        row.update(core_union_gain_pct=100 * pair["paired_delta"] / pair["baseline"],
                   old_core_surface_bending=old["surface"], voxel_core_surface_bending=new["surface"])
    args.output.mkdir(parents=True, exist_ok=True)
    fields = dict.fromkeys(key for row in rows for key in row)
    with (args.output / "paired.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    archive = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    (args.output / "results.json.gz").write_bytes(gzip.compress(archive, mtime=0))


if __name__ == "__main__":
    main()
