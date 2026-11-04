#!/usr/bin/env python3
"""Paired union volume and independent boundary geometry audit for E1."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
dependencies=ROOT/".scratch/native_splines/audit_dependencies/usr/lib/python3/dist-packages"
if dependencies.exists():sys.path.insert(0,str(dependencies))
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
from scipy.spatial import cKDTree

def geometry_audit(path,entry,tree):
    raw=np.loadtxt(path,delimiter=",",skiprows=1)
    degree=int(raw[0,0]);pieces=raw.reshape(-1,degree+1,12)
    controls=pieces[:,:,3:];h=pieces[:,0,2]-pieces[:,0,1]
    seed=np.loadtxt(ROOT/entry["seed"]);length=np.linalg.norm(np.diff(seed,axis=0),axis=1).sum()
    def basis(d,t):return np.array([math.comb(d,j)*t**j*(1-t)**(d-j) for j in range(d+1)]).T
    def evaluate(c,t):return np.einsum("tj,sjk->stk",basis(c.shape[1]-1,t),c)
    def world(y,angles):
        shape=y[:,:,3:].reshape(*y.shape[:2],2,3).swapaxes(-1,-2)
        return y[:,:,:3,None]+shape@angles
    angles=np.arange(48)*2*np.pi/48;disk=np.array([np.cos(angles),np.sin(angles)])
    t=np.linspace(0,1,65)
    first=degree*np.diff(controls,axis=1)/h[:,None,None]
    second=(degree-1)*np.diff(first,axis=1)/h[:,None,None]
    velocity=world(evaluate(first,t),disk).swapaxes(-1,-2)
    acceleration=world(evaluate(second,t),disk).swapaxes(-1,-2)
    speed=np.linalg.norm(velocity,axis=-1)
    curvature=np.linalg.norm(np.cross(velocity,acceleration),axis=-1)/speed**3
    # Angle-averaged squared longitudinal second derivative, measured using
    # reference arc s=L*u. It equals E_center + E_shape/2, not surface curvature.
    q=np.array([.5-.5/math.sqrt(3),.5+.5/math.sqrt(3)])
    value=evaluate(second,q)
    density=np.sum(value[:,:,:3]**2,axis=-1)+.5*np.sum(value[:,:,3:]**2,axis=-1)
    boundary_energy=float(np.sum(h*density.mean(axis=1))/length**3)
    jump=0.;coeff=controls.copy()
    for order in range(3):
        jump=max(jump,float(np.linalg.norm(coeff[:-1,-1]-coeff[1:,0],axis=1).max(initial=0)))
        if order<2:coeff=(degree-order)*np.diff(coeff,axis=1)/(h*length)[:,None,None]
    sample=evaluate(controls,np.linspace(0,1,17))
    points=np.concatenate([world(sample,disk*r).swapaxes(-1,-2).reshape(-1,3) for r in (0.,.5,1.)])
    nearest=tree.query(points,workers=1)[0]
    obstacle=max(0.,float((math.sqrt(3)*.25/2+.25-nearest).max()))
    bounds=max(0.,float((np.asarray(entry["bounds_lower"])+.25-points).max()),
                  float((points+.25-np.asarray(entry["bounds_upper"])).max()))
    assert max(obstacle,bounds)<1e-6,(entry["case"],path,obstacle,bounds)
    return dict(boundary_energy=boundary_energy,boundary_curvature_max=float(curvature.max()),
                boundary_curvature_p99=float(np.quantile(curvature,.99)),
                minimum_boundary_parameter_speed=float(speed.min()),c2_error=jump,
                sample_points=len(points),obstacle_penetration_m=obstacle,world_violation_m=bounds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development",default=".scratch/native_splines/e1/final_development")
    parser.add_argument("--expanded",default=".scratch/native_splines/e1/final_expanded")
    parser.add_argument("--output",default=".scratch/native_splines/e1/evaluation")
    parser.add_argument("--samples",type=int,default=262144)
    parser.add_argument("--cases",default="all")
    parser.add_argument("--volume-binary",default=".scratch/native_splines/e1_baseline/native_volume_refine")
    args=parser.parse_args()
    os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    inputs=json.loads((ROOT/".scratch/native_splines/e1/inputs/manifest.json").read_text())["cases"]
    if args.cases!="all":inputs=[r for r in inputs if r["case"] in set(map(int,args.cases.split(",")))]
    sources=[json.loads((ROOT/d/"summary.json").read_text()) for d in (args.development,args.expanded)]
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    results=[];joint=[]
    binary=ROOT/args.volume_binary
    for entry in inputs:
        case=entry["case"];source=sources[case>=7]
        base_name="frozen" if "frozen" in source else "triangle"
        baseline=next(r for r in source[base_name] if r["case"]==case)
        candidate=next(r for r in source["hybrid"] if r["case"]==case)
        row=dict(case=case,family=entry["family"],group=entry["group"],
                 baseline_status=baseline.get("status"),candidate_status=candidate.get("status"))
        successful=lambda r: bool(r.get("runs")) and all(s=="SUCCESS" for s in r["status"])
        if successful(baseline) and successful(candidate):
            target=out/f"case_{case:02d}";target.mkdir(exist_ok=True)
            for name,record in [("C3-before",baseline),("C3-E1",candidate)]:
                shutil.copy2(ROOT/record["output"]/"C3-OPT-native.csv",target/f"{name}.csv")
            tree=cKDTree(np.loadtxt(ROOT/entry["map"]).reshape(-1,3))
            row["geometry"]={name:geometry_audit(target/f"{name}.csv",entry,tree) for name in ("C3-before","C3-E1")}
            row["identical"]=baseline["geometry_sha256"]==candidate["geometry_sha256"]
            command=[str(binary),str(ROOT/entry["seed"]),str(target),str(args.samples),"C3-before,C3-E1"]
            started=time.monotonic()
            with (target/"volume.log").open("w") as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            row["volume"]=json.loads((target/"volume_refined.jsonl").read_text())
            row["measurement_s"]=time.monotonic()-started
            row["volume_command"]=command
            assert row["volume"]["unresolved"]==0
            joint.append(entry|dict(output=str(target.relative_to(ROOT))))
        results.append(row)
        (out/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        (out/"manifest.json").write_text(json.dumps(dict(cases=joint,samples_per_shift=args.samples,
                    shifts=8,cpu=2,address_space_limit_bytes=4*1024**3,
                    volume_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+"\n")
        v=row.get("volume",{})
        print(case,"volume",v.get("paired_delta"),"CI",v.get("paired_ci95"),"s",round(row.get("measurement_s",0),2),flush=True)


if __name__=="__main__":main()
