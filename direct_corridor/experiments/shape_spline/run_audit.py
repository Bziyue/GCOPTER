#!/usr/bin/env python3
"""Recheck saved violating samples against all roots of the certified sweep."""
from pathlib import Path
import json,subprocess,resource,os,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--integrated',action='store_true');p.add_argument('--fast',action='store_true');a=p.parse_args()
R=Path(__file__).resolve().parents[3];root=(a.root or R/'.scratch/shape_spline').resolve();dest=root/'audit';dest.mkdir(exist_ok=True);inputs=json.loads((R/'gcopter/experiments/trajectory/inputs.json').read_text());result=[]
def bound():
 resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CPU,(55,)*2)
for i,request in enumerate(inputs):
 case=f'case_{i:02}'
 for mode in (['baseline','boundary-12','continuation'] if a.fast else ['baseline','hermite-free','integrated-quintic' if a.integrated else 'cubic-c2']):
  path=root/'trajectory'/mode/case
  if not a.integrated and not a.fast and i==1 and mode=='cubic-c2':path=root/'diagnostic_cubic_case01/case_00'
  link=dest/'inputs'/mode/case;link.parent.mkdir(parents=True,exist_ok=True)
  if not link.exists():link.symlink_to(path,target_is_directory=True)
 req=R/request;meta=json.loads((req/'metrics.json').read_text());cloud=req.parent/f"map_{meta['map_version']}.xyz";file=dest/f'{case}.jsonl'
 cmd=[str(R/'.scratch/trajectory_comparison/build/shape_spline_audit'),str(req/'seed.xyz'),str(cloud),str(dest/'inputs'),case,*map(str,meta['bounds_lower']),*map(str,meta['bounds_upper']),str(meta['resolution']),str(file)]
 subprocess.run(cmd,check=True,preexec_fn=bound,timeout=60,env={**{k:v for k,v in os.environ.items() if k not in ['GCOPTER_INTEGRATED_COMPARE','GCOPTER_FINAL_FAST_COMPARE']},'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',**({'GCOPTER_INTEGRATED_COMPARE':'1'} if a.integrated else {}),**({'GCOPTER_FINAL_FAST_COMPARE':'1'} if a.fast else {})})
 result.append({'case':i,'request':request,'runs':[json.loads(x) for x in file.read_text().splitlines()]});print(case,result[-1]['runs'],flush=True)
 (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
