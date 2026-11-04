#!/usr/bin/env python3
"""Freeze the existing requests; never run a random map generator/path search."""
import hashlib
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
BASELINE = "15c70cdff162ed289aa23f8842bd5877b9b71fac"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    out = ROOT / ".scratch/native_splines/frozen"
    out.mkdir(parents=True, exist_ok=True)
    inputs = []
    for i, name in enumerate(json.loads((HERE / "inputs.json").read_text())):
        request = ROOT / name
        meta = json.loads((request / "metrics.json").read_text())
        files = {"seed": request / "seed.xyz", "map": request.parent / f"map_{meta['map_version']}.xyz"}
        row = {"case": i, "request": name, "bounds_lower": meta["bounds_lower"],
               "bounds_upper": meta["bounds_upper"], "resolution": meta["resolution"]}
        for kind, src in files.items():
            sha = digest(src)
            dst = out / f"{kind}_{sha}.xyz"
            if not dst.exists():
                shutil.copyfile(src, dst)
            if digest(dst) != sha:
                raise RuntimeError(f"snapshot hash mismatch: {dst}")
            row[kind] = str(dst.relative_to(ROOT))
            row[kind + "_sha256"] = sha
        inputs.append(row)
    sources = {}
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASELINE,
                                       "direct_corridor", "gcopter/experiments/trajectory", "gcopter/include"], cwd=ROOT, text=True).splitlines()
    for name in tracked:
        content = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=ROOT)
        sources[name] = hashlib.sha256(content).hexdigest()
    manifest = {"schema": "native-splines-freeze-v1", "baseline": BASELINE,
                "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                "worktree_at_freeze": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True),
                "settings": {"voxel_resolution": .25, "robot_radius": .2, "safety_margin": .05,
                             "certificate_margin": .002, "local_range": 3., "local_progress": 7.,
                             "control_spacing": .4, "frame_smoothing": 1.5, "section_smoothing": .2,
                             "jet_smoothing": .01, "certify_jacobian": True, "initial_iterations": 8,
                             "min_radius": .03, "budget_ms": 10000},
                "inputs": inputs, "baseline_source_sha256": sources,
                "frozen_binaries": {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.suffix != ".xyz"}}
    target = ROOT / "docs/validation/native_splines/frozen_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(target)

if __name__ == "__main__":
    main()
