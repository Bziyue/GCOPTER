#!/usr/bin/env python3
"""Serial builds, pinned CPU, 4 GiB address-space cap."""
import argparse,fcntl,os,resource,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--targets',default='tests,benchmark,baseline,measure');a=p.parse_args()
repo=Path(__file__).resolve().parents[3];os.chdir(repo);out=Path('.scratch/h3');out.mkdir(parents=True,exist_ok=True)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
lock=(out/'build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
baseline='7d4fc603607f131a1d282247c3e523c11da5d06e'
previous='d1e0ad66431f40836d7b82f65c8dfdcf27b95fb5'
joint='fd7a301e66bf8d836409ce508665ffedf457ef45'
for target in a.targets.split(','):
    if target not in ('tests','volume_tests','adaptive_tests','benchmark','baseline','measure','current_c3','world','previous','joint'):raise ValueError(target)
    if target in ('previous','joint'):
        commit=previous if target=='previous' else joint
        snapshot=out/target
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'SweepCorridor/include','SweepCorridor/experiments/h3','SweepCorridor/experiments/free_boundary/common.hpp','gcopter/tools/corridor_volume.hpp'],text=True).splitlines()
        for name in names:
            destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(subprocess.check_output(['git','show',commit+':'+name]))
        command=['c++','-std=c++17','-O3','-DNDEBUG','-I'+str(snapshot/'SweepCorridor/include'),'-I/usr/include/eigen3',str(snapshot/'SweepCorridor/experiments/h3/benchmark.cpp'),'-o',str(out/(target+'_benchmark'))]
        with (out/(target+'_build.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Built',target,commit,flush=True)
        continue
    if target=='baseline' and not (out/'baseline/include').exists():
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'SweepCorridor/include'],text=True).splitlines()
        for name in names:
            destination=out/'baseline'/Path(name).relative_to('SweepCorridor');destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(subprocess.check_output(['git','show',baseline+':'+name]))
    include=str(out/'baseline/include') if target=='baseline' else 'SweepCorridor/include'
    source='benchmark' if target in ('baseline','current_c3','world') else target
    folder='free_boundary' if target=='measure' else 'h3'
    destination='baseline_benchmark' if target=='baseline' else target
    command=['c++','-std=c++17','-O3','-DNDEBUG','-I'+include,'-I/usr/include/eigen3',f'SweepCorridor/experiments/{folder}/{source}.cpp','-o',str(out/destination)]
    if target in ('tests','volume_tests','adaptive_tests'):command.append('-UNDEBUG')
    if target=='world':command.append('-DH3_WORLD_ASSEMBLY')
    if target in ('baseline','current_c3'):command.append('-DH3_BASELINE')
    with (out/(target+'_build.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Built',target,flush=True)
