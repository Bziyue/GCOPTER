#!/usr/bin/env python3
"""Serial, bounded trajectory comparison on frozen RViz map/seed snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs', type=Path, required=True, help='JSON list of request directories, relative to repo root')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--binary', type=Path, default=ROOT / '.scratch/trajectory_comparison/build/trajectory_comparison')
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--spacing', type=float, default=1.0)
p.add_argument('--weight', type=float, default=1e5)
p.add_argument('--steps', type=int, default=16)
p.add_argument('--max-iterations', type=int, default=800)
p.add_argument('--cpu-seconds', type=int, default=55)
p.add_argument('--wall-seconds', type=int, default=60)
p.add_argument('--check-gradients', action='store_true')
p.add_argument('--constraint', choices=['sampled','bezier','bezier-controls'], default='bezier',
               help='bezier retains half-weight integral position penalties for the nonconvex tube; bezier-controls is a control-only ablation')
p.add_argument('--bezier-depth', type=int, choices=range(6), default=2)
p.add_argument('--tube-mode', choices=['reference','fast','horner'], default='fast')
p.add_argument('--profile', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
def bounded():
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_seconds, a.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
env=os.environ.copy()
env['LD_LIBRARY_PATH']=str(ROOT / '.scratch/ompl/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
env['GCOPTER_TUBE_MODE']=a.tube_mode
env['GCOPTER_PROFILE']='1' if a.profile else '0'
env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
manifest=[]
for i, request in enumerate(json.loads(a.inputs.read_text())):
    request=ROOT / request
    meta=json.loads((request/'metrics.json').read_text())
    seed=request/'seed.xyz'
    cloud=request.parent/f"map_{meta['map_version']}.xyz"
    dest=a.output/f'case_{i:02d}'
    dest.mkdir(exist_ok=True)
    command=[str(a.binary),str(seed),str(cloud),str(dest),
             *map(str,meta['bounds_lower']),*map(str,meta['bounds_upper']),
             str(meta['resolution']),str(a.repeats),str(a.spacing),str(a.weight),str(a.steps),str(int(a.check_gradients)),str(a.max_iterations),a.constraint,str(a.bezier_depth)]
    record=dict(case=i,request=str(request.relative_to(ROOT)),seed_sha256=sha(seed),map_sha256=sha(cloud),
                bounds_lower=meta['bounds_lower'],bounds_upper=meta['bounds_upper'],resolution=meta['resolution'])
    with (dest/'stderr.log').open('w') as log:
        try:
            process=subprocess.Popen(['/usr/bin/time', '-f', '%M', '-o', str(dest/'peak_rss_kib.txt'), *command],env=env,stdout=log,stderr=log,preexec_fn=bounded,start_new_session=True)
            record['returncode']=process.wait(timeout=a.wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            record['returncode']='wall_timeout'
    peak_file=dest/'peak_rss_kib.txt'
    if peak_file.exists():
        lines=peak_file.read_text().splitlines()
        if lines and lines[-1].isdigit(): record['peak_rss_kib']=int(lines[-1])
    manifest.append(record)
    print(f"case {i}: {record['returncode']}",flush=True)
    (a.output/'manifest.json').write_text(json.dumps(dict(parameters=dict(repeats=a.repeats,spacing=a.spacing,weight=a.weight,steps=a.steps,max_iterations=a.max_iterations,
        constraint=a.constraint,bezier_depth=a.bezier_depth,tube_mode=a.tube_mode,profile=a.profile,memory_mib=512,cpu_seconds=a.cpu_seconds,wall_seconds=a.wall_seconds,
        binary_sha256=sha(a.binary),platform=platform.platform(),processor=platform.processor()),cases=manifest),indent=2)+'\n')
if any(r['returncode']!=0 for r in manifest): raise SystemExit(1)
