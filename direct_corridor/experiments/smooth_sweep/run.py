#!/usr/bin/env python3
"""Serial same-input sweep benchmark; no address-space or cgroup memory limit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--inputs', type=Path, default=Path(__file__).with_name('inputs.json'))
p.add_argument('--repeats', type=int, default=7)
p.add_argument('--samples', type=int, default=65536, help='QMC points per shift; eight shifts')
p.add_argument('--frame', type=float, default=1.5)
p.add_argument('--smoothing', type=float, default=.2)
p.add_argument('--reserve', type=float, default=.01)
p.add_argument('--warmup', type=int, default=8)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
binary = ROOT/'.scratch/trajectory_comparison/build/smooth_sweep_benchmark'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
manifest = dict(parameters=dict(frame=a.frame, smoothing=a.smoothing, reserve=a.reserve,
    warmup=a.warmup, repeats=a.repeats, samples_per_shift=a.samples, shifts=8,
    memory_limit=None, binary_sha256=sha(binary)), cases=[])
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           SWEEP_JETS=str(a.reserve), SWEEP_WARMUP=str(a.warmup))
for i, request in enumerate(json.loads(a.inputs.read_text())):
    request = ROOT/request
    meta = json.loads((request/'metrics.json').read_text())
    seed = request/'seed.xyz'
    cloud = request.parent/f"map_{meta['map_version']}.xyz"
    dest = a.output/f'case_{i:02d}'
    dest.mkdir(exist_ok=True)
    command = [str(binary),str(seed),str(cloud),str(dest),str(a.frame),str(a.smoothing),
               '1',str(a.repeats),str(a.samples),*map(str,meta['bounds_lower']+meta['bounds_upper'])]
    with (dest/'run.log').open('w') as log:
        result = subprocess.run(['/usr/bin/time','-f','%M','-o',str(dest/'peak_rss_kib.txt'),
                                 *command],env=env,stdout=log,stderr=log)
    rows = [json.loads(line) for line in (dest/'results.jsonl').read_text().splitlines()]
    record = dict(case=i, request=str(request.relative_to(ROOT)), returncode=result.returncode,
                  seed_sha256=sha(seed), map_sha256=sha(cloud), bounds_lower=meta['bounds_lower'],
                  bounds_upper=meta['bounds_upper'], resolution=meta['resolution'], rows=rows,
                  peak_rss_kib=int((dest/'peak_rss_kib.txt').read_text().splitlines()[-1]))
    manifest['cases'].append(record)
    (a.output/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'case {i}: exit={result.returncode}, RSS={record["peak_rss_kib"]} KiB',flush=True)
    if result.returncode:
        print((dest/'run.log').read_text(),flush=True)
        raise SystemExit(result.returncode)
