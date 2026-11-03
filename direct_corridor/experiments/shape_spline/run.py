#!/usr/bin/env python3
"""Serial, memory-bounded shape parameterization generation/volume experiments."""
import argparse,hashlib,json,os,resource,signal,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cases', default='0,1,2,3,4,5')
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--samples', type=int, default=65536)
variants = p.add_mutually_exclusive_group()
for name in ['integrated', 'fast', 'coordinate', 'boundary', 'coarse', 'continuation']:
 variants.add_argument('--' + name, action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
inputs=json.loads((R/'gcopter/experiments/trajectory/inputs.json').read_text());binary=R/'.scratch/trajectory_comparison/build/shape_spline_benchmark'
if binary.stat().st_size == 0:raise SystemExit('Benchmark executable is empty; rebuild before running.')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bounded():
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CPU,(55,)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if a.integrated:env['GCOPTER_INTEGRATED_COMPARE']='1'
else:env.pop('GCOPTER_INTEGRATED_COMPARE',None)
if a.fast:env['GCOPTER_FAST_COMPARE']='1'
else:env.pop('GCOPTER_FAST_COMPARE',None)
if a.coordinate or a.boundary or a.coarse or a.continuation:env['GCOPTER_COORDINATE_COMPARE']='1'
else:env.pop('GCOPTER_COORDINATE_COMPARE',None)
if a.boundary or a.coarse or a.continuation:env['GCOPTER_BOUNDARY_COMPARE']='1'
else:env.pop('GCOPTER_BOUNDARY_COMPARE',None)
if a.coarse:env['GCOPTER_COARSE_COMPARE']='1'
else:env.pop('GCOPTER_COARSE_COMPARE',None)
if a.continuation:env['GCOPTER_CONTINUATION_COMPARE']='1'
else:env.pop('GCOPTER_CONTINUATION_COMPARE',None)
manifest={'parameters':{'integrated':a.integrated,'fast':a.fast,'coordinate':a.coordinate,'boundary':a.boundary,'coarse':a.coarse,'continuation':a.continuation,'repeats':a.repeats,'samples_per_shift':a.samples,'shifts':8,'memory_mib':512,'cpu_seconds':55,'wall_seconds':60,'binary_sha256':sha(binary)},'cases':[]}
for i in map(int,a.cases.split(',')):
 req=R/inputs[i];meta=json.loads((req/'metrics.json').read_text());cloud=req.parent/f"map_{meta['map_version']}.xyz";seed=req/'seed.xyz';dest=a.output/f'case_{i:02}';dest.mkdir(exist_ok=True)
 cmd=[str(binary),str(seed),str(cloud),str(dest),*map(str,meta['bounds_lower']),*map(str,meta['bounds_upper']),str(meta['resolution']),str(a.repeats),str(a.samples)]
 with (dest/'run.log').open('w') as log:
  proc=subprocess.Popen(['/usr/bin/time','-f','%M','-o',str(dest/'peak_rss_kib.txt'),*cmd],cwd=R,env=env,stdout=log,stderr=log,preexec_fn=bounded,start_new_session=True)
  try:code=proc.wait(timeout=60)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code='timeout'
 record={'case':i,'request':inputs[i],'seed_sha256':sha(seed),'map_sha256':sha(cloud),'returncode':code};rss=(dest/'peak_rss_kib.txt').read_text().splitlines();record['peak_rss_kib']=int(rss[-1]) if rss and rss[-1].isdigit() else None
 for name in ['generation','volume']:
  f=dest/(name+'.jsonl');record[name]=[json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
 if code==0 and (len(record['generation'])!=3*(a.repeats+1) or len(record['volume'])!=2):
  record['returncode']=code='incomplete output'
 manifest['cases'].append(record);(a.output/'results.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f'case_{i:02}: {code}, RSS {record["peak_rss_kib"]} KiB',flush=True)
 if code!=0:print((dest/'run.log').read_text()[-3000:],flush=True)
if any(x['returncode']!=0 for x in manifest['cases']):raise SystemExit(1)
