#!/usr/bin/env python3
import argparse,fcntl,os,resource,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--targets',default='tests,benchmark');a=p.parse_args()
repo=Path(__file__).resolve().parents[3];os.chdir(repo);out=Path('.scratch/free_boundary');out.mkdir(parents=True,exist_ok=True)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
lock=(out/'build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
for target in a.targets.split(','):
    assert target in ('tests','benchmark','measure','baseline')
    if target=='baseline' and not (out/'baseline/include').exists():
        names=subprocess.check_output(['git','ls-tree','-r','--name-only','4a210a1','SweepCorridor/include'],text=True).splitlines()
        for name in names:
            destination=out/'baseline'/Path(name).relative_to('SweepCorridor');destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(subprocess.check_output(['git','show','4a210a1:'+name]))
    include=str(out/'baseline/include') if target=='baseline' else 'SweepCorridor/include'
    source='benchmark' if target=='baseline' else target
    destination='baseline_benchmark' if target=='baseline' else target
    command=['c++','-std=c++17','-O3','-DNDEBUG','-I'+include,'-I/usr/include/eigen3',f'SweepCorridor/experiments/free_boundary/{source}.cpp','-o',str(out/destination)]
    if target=='tests':command.append('-UNDEBUG')
    with (out/(target+'_build.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Built',target,flush=True)
