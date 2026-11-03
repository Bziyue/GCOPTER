#!/usr/bin/env python3
"""Inactive-constraint control: both backends should recover the same solution."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[3]
fixture=root/'.scratch/trajectory_comparison/smoke_fixture'
request=fixture/'request_1'
request.mkdir(parents=True,exist_ok=True)
(request/'seed.xyz').write_text('-4 0 2\n4 0 2\n')
(fixture/'map_1.xyz').write_text('0 8 2\n')
(request/'metrics.json').write_text(json.dumps(dict(map_version=1,bounds_lower=[-10,-10,0],bounds_upper=[10,10,5],resolution=.25)))
inputs=fixture/'inputs.json'
inputs.write_text(json.dumps([str(request.relative_to(root))]))
output=root/'.scratch/trajectory_comparison/smoke'
subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),'--inputs',str(inputs),
    '--output',str(output),'--repeats','1','--check-gradients'],check=True,cwd=root)
rows=[json.loads(s) for s in (output/'case_00/runs.jsonl').read_text().splitlines()]
f,h=[next(r for r in rows if r['repeat']==0 and r['method']==method) for method in ['firi','hermite']]
assert f['initial_hash']==h['initial_hash']
assert f['sampled_feasible'] and h['sampled_feasible']
for key in ['duration_s','length_m','jerk_energy']:
    assert abs(f[key]-h[key])<=1e-4*max(1,abs(f[key])),(key,f[key],h[key])
print('PASS: shared initial vector and matching feasible inactive-constraint solutions')
