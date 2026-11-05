#!/usr/bin/env python3
"""Compare the latest H3 ROS exports with offline generation on identical inputs."""
import hashlib,json,os,resource,subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[3];os.chdir(repo)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
binary=repo/'.scratch/h3/benchmark';root=repo/'.scratch/h3_volume/rviz_replay';root.mkdir(parents=True,exist_ok=True)
rows=[]
for scene in ('perlin','maze'):
    smoke=json.loads((repo/f'.scratch/sweep_rviz_h3/{scene}_smoke.json').read_text())
    paths=list((repo/f'.scratch/sweep_rviz_h3/{scene}').glob('run_*/request_1/metrics.json'))
    source=max(paths,key=lambda p:p.stat().st_mtime).parent
    metrics=json.loads((source/'metrics.json').read_text())
    assert metrics['shape_mode']=='h3' and metrics['joint_steps']>0
    destination=root/scene;destination.mkdir(exist_ok=True)
    env=os.environ.copy();env['H3_MODE']='h3_new'
    command=[str(binary),str(source/'seed.xyz'),str(source.parent/f"map_{metrics['map_version']}.xyz"),str(destination),
        *map(str,metrics['bounds_lower']+metrics['bounds_upper']),'1']
    with (destination/'process.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    x=np.loadtxt(source/'cubic_controls.csv',delimiter=',',skiprows=1)
    y=np.loadtxt(destination/'corridor.csv',delimiter=',',skiprows=1)
    error=float(abs(x-y).max());assert error<1e-7
    rows.append(dict(scene=scene,source=str(source.relative_to(repo)),max_coefficient_error=error,
        joint_steps=metrics['joint_steps'],corridor_ms=metrics['corridor_ms'],smoke=smoke))
    print(scene,'max coefficient error',error,flush=True)
data=dict(purpose='ROS goal-message integration and exact offline replay; not a performance benchmark',
    cpu=2,address_space_limit_bytes=4*1024**3,trajectory_optimization=False,h3=rows,
    binaries={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (binary,repo/'install/gcopter/lib/gcopter/direct_corridor_demo')})
Path('docs/validation/h3_volume/rviz.json').write_text(json.dumps(data,indent=2)+'\n')
