#!/usr/bin/env python3
"""Serial frozen-input regression; compare exported geometry and solver decisions."""
import argparse,hashlib,json,os,resource,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--binary',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=7);p.add_argument('--cases',default='all');a=p.parse_args()
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.sched_setaffinity(0,{2})
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
inputs=json.loads(Path(a.inputs).read_text())['cases'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
selected=None if a.cases=='all' else set(map(int,a.cases.split(',')))
summary=[]
for row in inputs:
 if selected is not None and row['case'] not in selected:continue
 for k in ('map','seed'):assert hashlib.sha256(Path(row[k]).read_bytes()).hexdigest()==row[k+'_sha256']
 for degree in ([3,5] if row['case']<7 else [3]):
  names=['before','after'] if row['case']%2==0 else ['after','before'];pair={}
  for name in names:
   binary=a.baseline if name=='before' else a.binary;target=out/f'd{degree}'/name/str(row['case']);target.mkdir(parents=True,exist_ok=True)
   env={k:v for k,v in os.environ.items() if not k.startswith('NATIVE_')};env['NATIVE_GENERATION_ONLY']='1'
   if degree==3:env.update(NATIVE_CANDIDATES='1',NATIVE_SKIP_H5='1',NATIVE_METHOD_COUNT='1')
   cmd=[binary,row['seed'],row['map'],str(target),*map(str,row['bounds_lower']+row['bounds_upper']),str(a.repeats),'0']
   with (target/'process.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
   assert r.returncode==0,(row['case'],degree,name,r.returncode)
   model='C3-OPT-native' if degree==3 else 'H5-S';rows=[json.loads(x) for x in (target/'rows.jsonl').read_text().splitlines()]
   runs=[x for x in rows if x.get('kind')=='generation' and x.get('model')==model and x['repeat']>=0]
   geometry=target/(model+'.csv');planes=target/(model+'_support_planes.csv')
   if name=='before' and degree==5:planes=target/'support_planes.csv'
   pair[name]=dict(status=[r['status'] for r in runs],median_ms=statistics.median(x['total_ms' if degree==3 else 'wall_ms'] for x in runs),runs=runs,
     geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest() if geometry.exists() else None,
     support_sha256=hashlib.sha256(planes.read_bytes()).hexdigest() if planes.exists() else None,
     resources=[x for x in rows if x.get('kind')=='resource'],output=str(target),command=cmd)
  exact=all(pair['before'][k]==pair['after'][k] for k in ('status','geometry_sha256','support_sha256'))
  record=dict(case=row['case'],degree=degree,exact=exact,**pair);summary.append(record)
  (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  print(row['case'],degree,'exact' if exact else 'DIFFERENT',round(pair['before']['median_ms'],3),round(pair['after']['median_ms'],3),flush=True)
meta=dict(binaries={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in dict(before=a.baseline,after=a.binary).items()},inputs=a.inputs,repeats=a.repeats,memory_limit=4*1024**3,cpu=2)
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
assert all(r['exact'] for r in summary),'see differences in summary.json'
