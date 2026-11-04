import json,os,subprocess,resource,statistics,hashlib
from pathlib import Path
root=Path('docs/validation/c3_specialization');root.mkdir(parents=True,exist_ok=True)
cases=json.load(open('.scratch/native_splines/e1/inputs/manifest.json'))['cases']
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NATIVE_CANDIDATES='1',NATIVE_SKIP_H5='1',SWEEP_QUERY_MEASURE='1')
env.pop('GCOPTER_PROFILE',None)
def limit():resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));os.sched_setaffinity(0,{2})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for c in cases[:7]:
 pair={}
 for label in (['before','after'] if c['case']%2==0 else ['after','before']):
  binary=Path('.scratch/c3_specialize/baseline/sweep_frozen' if label=='before' else '.scratch/c3_specialize/build/sweep_frozen')
  target=root/'queries'/label/str(c['case']);target.mkdir(parents=True,exist_ok=True)
  cmd=[str(binary),c['seed'],c['map'],str(target),*map(str,c['bounds_lower']+c['bounds_upper']),'1','0']
  with (target/'process.log').open('w') as log:r=subprocess.run(cmd,env=env,preexec_fn=limit,stdout=log,stderr=log,timeout=120)
  assert r.returncode==0
  rows=[json.loads(x) for x in (target/'rows.jsonl').read_text().splitlines()]
  pair[label]={m:statistics.median(x['ms'] for x in rows if x.get('kind')=='query' and x['mode']==m and x['repeat']>=0) for m in ['full_hessian','first_gradient','inverse_unique','inverse_repeated']}
 records.append(dict(case=c['case'],**pair))
(root/'query_summary.json').write_text(json.dumps(records,indent=2)+'\n')
env.pop('SWEEP_QUERY_MEASURE')
records=[]
for c in cases[:7]:
 pair={}
 for label in (['before','after'] if c['case']%2==0 else ['after','before']):
  binary=Path('.scratch/c3_specialize/baseline/sweep_trajectory_comparison' if label=='before' else '.scratch/c3_specialize/trajectory/sweep_trajectory_comparison')
  target=root/'trajectory'/label/str(c['case']);target.mkdir(parents=True,exist_ok=True)
  cmd=[str(binary),c['seed'],c['map'],str(target),*map(str,c['bounds_lower']+c['bounds_upper']),'.25','3','1','100000','16','1','800','bezier','2']
  with (target/'process.log').open('w') as log:r=subprocess.run(['/usr/bin/time','-v',*cmd],env=env,preexec_fn=limit,stdout=log,stderr=log,timeout=120)
  assert r.returncode==0
  rows=[json.loads(x) for x in (target/'runs.jsonl').read_text().splitlines()]
  rows=[x for x in rows if x['method']=='C3'];measured=[x for x in rows if x['repeat']>=0]
  pair[label]=dict(median_ms=statistics.median(x['optimize_ms'] for x in measured),trajectory_sha256=sha(target/'C3_trajectory.csv'),initial_hash=rows[0]['initial_hash'],runs=rows,binary_sha256=sha(binary),command=cmd)
 records.append(dict(case=c['case'],exact=pair['before']['trajectory_sha256']==pair['after']['trajectory_sha256'],**pair))
 (root/'trajectory_summary.json').write_text(json.dumps(records,indent=2)+'\n')
 print(c['case'],records[-1]['exact'],pair['before']['median_ms'],pair['after']['median_ms'],flush=True)
