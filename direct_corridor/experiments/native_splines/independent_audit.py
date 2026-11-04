#!/usr/bin/env python3
"""Independent NumPy polynomial / SciPy nearest-sphere sampling audit.

This reports collision witnesses, not a replacement for continuous certificates.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--geometry", default=".scratch/native_splines/final")
    parser.add_argument("--models", default="H5-S,C3-I,Q5-U,C3-OPT-native,C3-OPT-project,C3-NBC-project")
    parser.add_argument("--output", default="docs/validation/native_splines/independent_audit.json")
    args=parser.parse_args()
    manifest = json.loads((ROOT / args.geometry / "manifest.json").read_text())
    audit = []
    t = np.linspace(0, 1, 17)
    angles = np.arange(48)*2*math.pi/48
    disk = np.array([np.cos(angles), np.sin(angles)])
    for entry in manifest["cases"]:
        tree = cKDTree(np.loadtxt(ROOT / entry["map"]))
        length = np.linalg.norm(np.diff(np.loadtxt(ROOT / entry["seed"]), axis=0), axis=1).sum()
        for model in args.models.split(","):
            path = ROOT / entry["output"] / (model+".csv")
            data = np.genfromtxt(path, delimiter=",", skip_header=1)
            degree = int(data[0, 0])
            pieces = data.reshape(-1, degree+1, 12)
            basis = np.array([math.comb(degree, j)*t**j*(1-t)**(degree-j) for j in range(degree+1)]).T
            y = basis @ pieces[:, :, 3:]
            c = y[:, :, :3]
            m = y[:, :, 3:].reshape(-1, len(t), 2, 3).swapaxes(-1, -2)
            points = np.concatenate([(c[:, :, :, None]+radius*m@disk).transpose(0, 1, 3, 2).reshape(-1, 3) for radius in (0., .5, 1.)])
            distances = tree.query(points, workers=1)[0]
            obstacle = max(0., float((math.sqrt(3)*.25/2+.25-distances).max()))
            world = max(0., float((np.array(entry["bounds_lower"])+.25-points).max()),
                        float((points+.25-np.array(entry["bounds_upper"])).max()))
            controls = pieces[:, :, 3:]
            h = (pieces[:, 0, 2]-pieces[:, 0, 1])*length
            jump = 0.
            for order in range(3):
                jump = max(jump, float(np.linalg.norm(controls[:-1, -1]-controls[1:, 0], axis=1).max(initial=0)))
                if order < 2:
                    controls = (degree-order)*np.diff(controls, axis=1)/h[:, None, None]
            row = {"case": entry["case"], "model": model, "sample_points": len(points),
                   "obstacle_penetration_m": obstacle, "world_violation_m": world, "c2_error_seed_arc": jump}
            audit.append(row)
            if model not in ("C3-I", "Q5-U") and max(obstacle, world) > 1e-6:
                raise RuntimeError(f"independent sampling found unsafe certified output: {row}")
        print(f"independent audit case {entry['case']}: completed", flush=True)
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2)+"\n")

if __name__ == "__main__":
    main()
