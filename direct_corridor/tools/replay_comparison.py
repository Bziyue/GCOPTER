#!/usr/bin/env python3
"""Matched-parameter replay of saved direct_demo requests, with an independent audit.
The baseline executable must support the same CLI arguments; nine-column cubic
B-spline and eleven-column quintic Bezier output are both decoded here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np


def sections(path, count=501):
    cp = np.loadtxt(path, delimiter=',', skiprows=1)
    values = []
    for u in np.linspace(0, 1, count):
        if cp.shape[1] == 11:
            i = min(np.searchsorted(cp[::6, 1], u, side='right'), len(cp)//6-1)
            block = cp[6*i:6*i+6]
            t = (u-block[0, 0])/(block[0, 1]-block[0, 0])
            b = [math.comb(5, j)*t**j*(1-t)**(5-j) for j in range(6)]
            q = np.array(b) @ block[:, 2:]
        else:
            t = u*(len(cp)-3)
            i = min(int(t), len(cp)-4)
            t -= i
            b = np.array([(1-t)**3, 3*t**3-6*t*t+4,
                          -3*t**3+3*t*t+3*t+1, t**3])/6
            q = b @ cp[i:i+4]
        values.append(q)
    return np.array(values)


def quality(values):
    axes = np.linalg.svd(values[:, 3:].reshape(-1, 2, 3), compute_uv=False)
    return dict(min_minor=float(axes[:, 1].min()),
                median_minor=float(np.median(axes[:, 1])),
                mean_area=float(np.mean(np.prod(axes, axis=1))*math.pi))


def audit(values, cloud, cfg):
    """Independent grid lookup; checks interiors as well as ellipse boundaries."""
    lo, hi = np.array(cfg['bounds_lower']), np.array(cfg['bounds_upper'])
    res = cfg['resolution']
    dims = np.ceil((hi-lo)/res).astype(int)
    cells = np.floor((cloud-lo)/res).astype(int)
    occupied = np.zeros(tuple(dims), dtype=bool)
    occupied[tuple(cells.T)] = True
    pad = cfg['robot_radius']+cfg['safety_margin']
    required = math.sqrt(3)*res/2+pad
    angles = np.arange(32)*2*np.pi/32
    disc = np.concatenate([r*np.column_stack([np.cos(angles), np.sin(angles)])
                           for r in (0, .25, .5, .75, 1)])
    p = (values[:, None, :3] + np.einsum('nkj,ak->naj',
          values[:, 3:].reshape(-1, 2, 3), disc)).reshape(-1, 3)
    assert (p >= lo+pad-1e-8).all() and (p <= hi-pad+1e-8).all(), 'map collision'
    pcells = np.floor((p-lo)/res).astype(int)
    best = math.inf
    radius = required+.05
    n = math.ceil(radius/res)
    for dx in range(-n, n+1):
        for dy in range(-n, n+1):
            for dz in range(-n, n+1):
                idx = pcells+[dx, dy, dz]
                valid = np.flatnonzero(((idx >= 0) & (idx < dims)).all(axis=1))
                hits = valid[occupied[tuple(idx[valid].T)]]
                if len(hits):
                    best = min(best, float(np.linalg.norm(p[hits]-(lo+(idx[hits]+.5)*res), axis=1).min()))
    assert best > required-1e-8, f'sphere collision: {best} <= {required}'
    return dict(sample_points=len(p), minimum_observed_clearance=best-required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    requests = sorted(set(p.parent for root in args.input for p in root.rglob('seed.xyz')))
    for request in requests:
        cfg = json.loads((request/'metrics.json').read_text())
        if cfg['status'] not in ('SUCCESS', 'PASS'):
            continue
        cloud = request.parent/f"map_{cfg['map_version']}.xyz"
        seed = request/'seed.xyz'
        record = dict(request=str(request), seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                      map_sha256=hashlib.sha256(cloud.read_bytes()).hexdigest())
        for name, exe in [('old', args.baseline), ('new', args.candidate)]:
            if exe is None:
                continue
            dest = args.output/f'{len(records)+1}_{name}'
            cmd = [str(exe.resolve()), str(seed), str(cloud), str(math.sqrt(3)*cfg['resolution']/2),
                   str(cfg['robot_radius']), *map(str, cfg['bounds_lower']), *map(str, cfg['bounds_upper']),
                   str(dest), str(cfg['safety_margin']), str(cfg['control_spacing']), '10000', str(cfg['min_radius'])]
            times = []
            for _ in range(args.repeats):
                run = subprocess.run(cmd, text=True, capture_output=True)
                if run.returncode:
                    record[name] = dict(status='FAILED', message=run.stdout+run.stderr)
                    break
                stats = json.loads((dest/'metrics.json').read_text())
                times.append(stats['total_ms'])
            else:
                values = sections(dest/'controls.csv')
                stats.update(quality(values), median_ms=float(np.median(times)), times_ms=times)
                if name == 'new' and args.audit:
                    stats['audit'] = audit(values, np.loadtxt(cloud), cfg)
                record[name] = stats
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.output/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    if not records or any(r['new']['status'] != 'SUCCESS' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
