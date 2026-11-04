#!/usr/bin/env python3
"""Freeze existing requests and deterministic additional corridor environments."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/".scratch/native_splines/e1/inputs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "E1 corridor generation; inputs frozen before expanded measurements",
                "voxel_resolution": .25, "cases": [], "excluded_inputs": []}
    seen = set()

    def append(seed, cloud, lower, upper, family, source, group):
        key = (digest(seed), digest(cloud))
        if key in seen:
            return
        seen.add(key)
        row = dict(case=len(manifest["cases"]), family=family, group=group, source=source,
                   bounds_lower=list(lower), bounds_upper=list(upper), resolution=.25)
        for kind, path, sha in [("seed", seed, key[0]), ("map", cloud, key[1])]:
            target = OUT/f"{kind}_{sha}.xyz"
            if not target.exists():
                shutil.copy2(path, target)
            row[kind] = str(target.relative_to(ROOT))
            row[kind+"_sha256"] = sha
        manifest["cases"].append(row)

    frozen = json.loads((ROOT/"docs/validation/native_splines/frozen_manifest.json").read_text())["inputs"]
    for row in frozen:
        append(ROOT/row["seed"], ROOT/row["map"], row["bounds_lower"], row["bounds_upper"],
               "maze" if row["case"] in (2,5) else "perlin", row["request"], "development")
    for seed in sorted((ROOT/".scratch/interactive").rglob("seed.xyz")):
        metrics = seed.parent/"metrics.json"
        if not metrics.exists():
            continue
        data = json.loads(metrics.read_text())
        cloud = seed.parent.parent/f"map_{data.get('map_version',1)}.xyz"
        lines = [s for s in seed.read_text().splitlines() if s.strip() and not s.startswith("#")]
        if not cloud.exists() or len(lines)<2:
            manifest["excluded_inputs"].append(dict(source=str(seed.relative_to(ROOT)), reason="missing map or fewer than two seed points"))
            continue
        append(seed, cloud, data["bounds_lower"], data["bounds_upper"],
               "maze" if "maze" in str(seed) else "perlin", str(seed.parent.relative_to(ROOT)), "historical")

    source = OUT/"generated"
    source.mkdir(exist_ok=True)
    for family in ("slalom", "variable_tube", "helix", "maze_turns"):
        for random_seed in (1104,1105,1106):
            rng = np.random.default_rng(random_seed)
            phase = rng.uniform(-.4,.4)
            x = np.linspace(-8,8,33)
            bounds_lower = np.array([-11.,-7.,0.])
            bounds_upper = np.array([11.,7.,7.])
            if family == "slalom":
                path = np.column_stack((x, 1.6*np.sin(.45*x+phase), 3+.4*np.cos(.32*x)))
            elif family == "variable_tube":
                path = np.column_stack((x, .4*np.sin(.3*x+phase), np.full_like(x,3.)))
            elif family == "helix":
                angle = np.linspace(-.8*np.pi,.8*np.pi,41)+phase
                path = np.column_stack((3*np.cos(angle),3*np.sin(angle),np.linspace(1.5,6.5,len(angle))))
                bounds_lower = np.array([-6.,-6.,0.]);bounds_upper = np.array([6.,6.,8.])
            else:
                path = np.array([[-8,-4,3],[-8,4,3],[-3,4,3],[-3,-4,3],[3,-4,3],[3,4,3],[8,4,3]],float)
                path[:,1] *= 1+.1*phase
            axes = [np.arange(a+.125,b,.25) for a,b in zip(bounds_lower,bounds_upper)]
            grid = np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3)
            best = np.full(len(grid), np.inf); progress = np.zeros(len(grid))
            lengths = np.linalg.norm(np.diff(path,axis=0),axis=1)
            accumulated = 0.
            for a,b,length in zip(path[:-1],path[1:],lengths):
                direction = b-a
                t = np.clip((grid-a)@direction/(length*length),0,1)
                distance = np.linalg.norm(grid-a-t[:,None]*direction,axis=1)
                take = distance<best
                progress[take] = (accumulated+t[take]*length)/lengths.sum()
                best[take] = distance[take]
                accumulated += length
            if family == "variable_tube":
                radius = .78+1.25*(1-np.exp(-((progress-.5-phase*.1)/.14)**2))
            else:
                radius = 1.25+.22*np.sin(2*np.pi*progress+phase)
            # Two voxel layers around a known path tube; every arm sees the
            # exact same voxel spheres, never a different distance-field model.
            points = grid[(best>=radius)&(best<radius+.36)]
            seed_path = source/f"{family}_{random_seed}_seed.xyz"
            map_path = source/f"{family}_{random_seed}_map.xyz"
            np.savetxt(seed_path,path,fmt="%.17g");np.savetxt(map_path,points,fmt="%.17g")
            append(seed_path,map_path,bounds_lower,bounds_upper,family,
                   f"procedural/{family}/seed={random_seed}","independent")
    # Open 3-D motion and two failure controls are frozen alongside valid cases.
    for family,path in [("open_straight",[[-8,0,3],[8,0,3]]),
                        ("open_vertical",[[0,0,1],[0,0,7]]),
                        ("backtracking_cusp",[[-4,0,3],[0,0,3],[-4,0,3]]),
                        ("narrow_003m",[[-3,0,3],[3,0,3]])]:
        points = np.array([[-9,-6,.125],[9,6,7.875]])
        if family=="narrow_003m":
            gap=np.sqrt(3)*.25/2+.25+.015
            points=np.array([(x,sign*gap,z) for x in np.arange(-3,3.01,.25)
                             for z in np.arange(1,5.01,.25) for sign in (-1,1)])
        seed_path=source/f"{family}_seed.xyz";map_path=source/f"{family}_map.xyz"
        np.savetxt(seed_path,path,fmt="%.17g");np.savetxt(map_path,points,fmt="%.17g")
        append(seed_path,map_path,[-10,-7,0],[10,7,8],family,f"procedural/{family}",
               "failure_control" if family in ("backtracking_cusp","narrow_003m") else "independent")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    from collections import Counter
    print(len(manifest["cases"]),"frozen cases",dict(Counter(r["group"] for r in manifest["cases"])))


if __name__ == "__main__":
    main()
