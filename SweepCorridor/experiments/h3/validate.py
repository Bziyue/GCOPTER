#!/usr/bin/env python3
"""Rebuild and reproduce the final report, serially within 4 GiB."""
import os,resource,subprocess,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[3];os.chdir(repo);out=Path('.scratch/h3');out.mkdir(parents=True,exist_ok=True)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
scripts=Path('SweepCorridor/experiments/h3')
def python(name,*args,**kwargs):subprocess.run([sys.executable,str(scripts/name),*args],check=True,**kwargs)
def logged(command,name):
    with (out/name).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
python('build.py','--targets','tests,benchmark,baseline,measure,current_c3,world')
logged([str(out/'tests')],'tests.log')
logged(['cmake','-S','SweepCorridor','-B',str(out/'library'),'-DCMAKE_BUILD_TYPE=Release'],'library_configure.log')
logged(['cmake','--build',str(out/'library'),'-j1'],'library_build.log')
logged(['ctest','--test-dir',str(out/'library'),'--output-on-failure'],'library_test.log')
python('run.py','--output',str(out/'final_development'),'--cases','0,1,2,3,4,5,6,67','--modes','c3,h3,h3_jets,h3_mid','--repeats','7','--samples','65536')
python('run.py','--output',str(out/'final_regression'),'--cases',','.join(map(str,range(7,67))),'--modes','c3,h3_jets,h3_mid','--repeats','3','--samples','65536')
python('run.py','--output',str(out/'assembly_ablation'),'--cases','0,2,67','--modes','c3,h3_world,h3_jets','--repeats','7','--samples','65536')
env=os.environ.copy();env['SWEEP_PROFILE']='1'
python('run.py','--output',str(out/'profile'),'--cases','0,60,67','--modes','c3,h3_jets,h3_mid','--repeats','3','--samples','0',env=env)
python('verify_baseline.py');python('analyze.py')
