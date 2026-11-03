#!/usr/bin/env python3
"""Serial, bounded independent u/z experiment on frozen map/seed snapshots."""
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
p.add_argument('--binary', type=Path, default=ROOT / '.scratch/trajectory_comparison/build/uz_experiment')
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--quadrature', type=int, choices=range(2,65), default=16)
p.add_argument('--max-iterations', type=int, default=800)
p.add_argument('--cpu-seconds', type=int, default=55)
p.add_argument('--wall-seconds', type=int, default=60)
p.add_argument('--check-gradients', action='store_true')
p.add_argument('--no-time-scan', action='store_true', help='Disable the nine-point initial duration search')
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
env['GCOPTER_UZ_TIME_SCAN']='0' if a.no_time_scan else '1'
env['LD_LIBRARY_PATH']=str(ROOT / '.scratch/ompl/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
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
             str(meta['resolution']),str(a.repeats),'1','0',str(a.quadrature),str(int(a.check_gradients)),str(a.max_iterations),'uz','0']
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
    (a.output/'manifest.json').write_text(json.dumps(dict(parameters=dict(backend="uz",dimension=11,time_scan=not a.no_time_scan,repeats=a.repeats,quadrature=a.quadrature,max_iterations=a.max_iterations,
        memory_mib=512,cpu_seconds=a.cpu_seconds,wall_seconds=a.wall_seconds,
        binary_sha256=sha(a.binary),platform=platform.platform(),processor=platform.processor()),cases=manifest),indent=2)+'\n')
if any(r['returncode']!=0 for r in manifest): raise SystemExit(1)
