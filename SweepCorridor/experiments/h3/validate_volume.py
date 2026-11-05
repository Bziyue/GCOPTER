#!/usr/bin/env python3
"""Frozen-input shared-jet study. Serial execution, one CPU, 4 GiB cap."""
import argparse,os,resource,subprocess,sys,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--skip-build',action='store_true');a=p.parse_args()
repo=Path(__file__).resolve().parents[3];os.chdir(repo)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
root=Path('.scratch/h3_volume');root.mkdir(parents=True,exist_ok=True)
scripts=Path('SweepCorridor/experiments/h3');archive=Path('docs/validation/h3_volume')
with tarfile.open(archive/'inputs.tar.gz') as tar:tar.extractall(root/'inputs',filter='data')
def call(name,*args,**kwargs):subprocess.run([sys.executable,str(scripts/name),*args],check=True,**kwargs)
if not a.skip_build:call('build.py','--targets','tests,volume_tests,benchmark,baseline,previous,measure')
for name in ('tests','volume_tests'):
    with (root/(name+'.log')).open('w') as log:subprocess.run([str(Path('.scratch/h3')/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
def run(name,cases,modes,repeats,samples,**kwargs):
    call('run.py','--output',str(root/name),'--inputs',str(archive/'inputs.json'),'--cases',cases,
         '--modes',modes,'--repeats',str(repeats),'--samples',str(samples),**kwargs)
run('final_development','0,1,2,3,4,5,6,67,68,69,70','c3,h3_previous,h3_new',7,65536)
run('final_regression',','.join(map(str,range(7,67))),'c3,h3_previous,h3_new',3,65536)
run('final_ablation','0,2,67,68','h3_new,h3_new_rebuild,h3_new_halving',7,0)
env=os.environ.copy();env['SWEEP_PROFILE']='1'
run('final_profile','0,1,67,68','c3,h3_previous,h3_new',3,0,env=env)
print('Generation, certificates and union measurements completed.',flush=True)
