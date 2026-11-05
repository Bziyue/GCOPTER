#!/usr/bin/env python3
"""Check that the compile-time pruning extension leaves the C3 baseline intact."""
import hashlib,json,os,resource,subprocess
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3]);os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
root=Path('.scratch/h3');records=json.loads((root/'final_development/summary.json').read_text())
cases={r['case']:r for r in json.loads(Path('docs/validation/free_boundary/inputs.json').read_text())['cases']}
with (root/'c3_unchanged.log').open('w') as log:
    for r in records:
        if r['mode']!='c3':continue
        case=cases[r['case']];out=root/'c3_unchanged'/f"case_{r['case']:02d}";out.mkdir(parents=True,exist_ok=True)
        command=[str(root/'current_c3'),case['seed'],case['map'],str(out),*map(str,case['bounds_lower']+case['bounds_upper']),'1']
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        for name in ('corridor.csv','core.csv','support_planes.csv'):
            assert (out/name).read_bytes()==(Path(r['output'])/name).read_bytes(),(r['case'],name)
        log.write(f"case {r['case']}: geometry, core and support planes byte-identical\n");log.flush()
print('C3 eight-case geometry / core / planes: byte-identical')
