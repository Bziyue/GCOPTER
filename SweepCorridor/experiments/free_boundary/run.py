#!/usr/bin/env python3
"""Serial, paired frozen-input comparison. Retain generation failures."""
import argparse,hashlib,json,os,platform,resource,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cases',default='0,1,2,3,4,5,6,67')
p.add_argument('--inputs',default='docs/validation/inflation_round1/inputs.json');p.add_argument('--repeats',type=int,default=7)
p.add_argument('--samples',type=int,default=65536);a=p.parse_args();repo=Path(__file__).resolve().parents[3];os.chdir(repo)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);build=Path('.scratch/free_boundary').resolve()
cases=json.loads(Path(a.inputs).read_text())['cases'];ids=None if a.cases=='all' else set(map(int,a.cases.split(',')))
cases=[r for r in cases if ids is None or r['case'] in ids];hashes={}
for r in cases:
    for key in ('seed','map'):
        path=r[key]
        if path not in hashes:hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if key+'_sha256' in r:assert r[key+'_sha256']==hashes[path]
manifest=dict(cases=cases,repeats=a.repeats,samples_per_shift=a.samples,shifts=8,memory_limit=4*1024**3,cpu=2,
              commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],platform=platform.platform(),
              worktree=subprocess.check_output(['git','status','--short'],text=True),
              binaries={name:hashlib.sha256((build/name).read_bytes()).hexdigest() for name in ('benchmark','baseline_benchmark','measure')},
              sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in Path('SweepCorridor/include').rglob('*.hpp')})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');summary=[]
for case in cases:
    records={}
    modes=['baseline','free_boundary'];modes=modes if case['case']%2==0 else modes[::-1]
    for mode in modes:
        target=out/f"case_{case['case']:02d}"/mode;target.mkdir(parents=True,exist_ok=True)
        binary=build/('baseline_benchmark' if mode=='baseline' else 'benchmark')
        command=[str(binary),case['seed'],case['map'],str(target),*map(str,case['bounds_lower']+case['bounds_upper']),str(a.repeats)]
        with (target/'process.log').open('w') as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        rows=[json.loads(line) for line in (target/'rows.jsonl').read_text().splitlines()];runs=[r for r in rows if r.get('repeat',-1)>=0]
        times=sorted(r['wall_ms'] for r in runs)
        r=dict(case=case['case'],family=case['family'],mode=mode,success=process.returncode==0 and len(runs)==a.repeats and all(r['status']=='SUCCESS' for r in runs),
               returncode=process.returncode,runs=runs,median_ms=statistics.median(times) if times else None,p95_ms=times[min(len(times)-1,int(.95*len(times)))] if times else None,
               max_rss_kib=rows[-1].get('max_rss_kib',0) if rows else 0,output=str(target))
        if r['success']:
            r.update({k:runs[0][k] for k in ('swept_volume','boundary_bending','c2_error','anchor_begin','anchor_end','start_distance','end_distance','steps','backtracks','jacobian_backtracks','controls','constraints','active_constraints')})
            r['geometry_sha256']=hashlib.sha256((target/'corridor.csv').read_bytes()).hexdigest()
        records[mode]=r;summary.append(r);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(case['case'],mode,'PASS' if r['success'] else ('FAIL '+str([t['status'] for t in runs])),r['median_ms'],r.get('swept_volume'),flush=True)
    if a.samples and all(r['success'] for r in records.values()):
        base=Path(records['baseline']['output'])/'corridor.csv';r=records['free_boundary'];target=Path(r['output'])
        for key,file in [('union','corridor.csv'),('core_union','core.csv')]:
            data=json.loads(subprocess.check_output([str(build/'measure'),str(base),str(target/file),case['seed'],str(a.samples)],text=True,timeout=180))
            assert data['unresolved']==0;r[key]=data
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('union',case['case'],100*r['union']['paired_delta']/r['union']['baseline'],'%',flush=True)
