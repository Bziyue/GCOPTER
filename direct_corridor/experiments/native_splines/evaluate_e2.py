#!/usr/bin/env python3
"""Paired union volumes and boundary audits for E2 certificate replacements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from evaluate_e1 import ROOT,geometry_audit,np,cKDTree


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations",nargs="+",required=True)
    parser.add_argument("--configurations",default="elevate05,elevate10,elevate25,split05,split10,split25")
    parser.add_argument("--baseline",default="baseline")
    parser.add_argument("--output",required=True)
    parser.add_argument("--samples",type=int,default=65536)
    parser.add_argument("--cases",default="all")
    args=parser.parse_args()
    os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    inputs=json.loads((ROOT/".scratch/native_splines/e1/inputs/manifest.json").read_text())["cases"]
    names=[args.baseline,*args.configurations.split(",")]
    records={name:{} for name in names}
    for directory in args.generations:
        source=json.loads((ROOT/directory/"summary.json").read_text())
        for name in names:
            for row in source.get(name,[]):records[name][row["case"]]=row
    selected=set(records[args.baseline])
    if args.cases!="all":selected&=set(map(int,args.cases.split(",")))
    inputs=[row for row in inputs if row["case"] in selected]
    binary=ROOT/".scratch/native_splines/e2_baseline/native_volume_refine"
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    results=[]
    manifest=dict(generations=args.generations,configurations=names,samples_per_shift=args.samples,shifts=8,
                  cpu=2,address_space_limit_bytes=4*1024**3,
                  volume_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for entry in inputs:
        case=entry["case"];target=out/f"case_{case:02d}";target.mkdir(exist_ok=True)
        row=dict(case=case,family=entry["family"],group=entry["group"],configurations={})
        successful=[];tree=None
        for name in names:
            record=records[name].get(case)
            if record is None:continue
            result=dict(status=record.get("status"),median_ms=record.get("median_ms"))
            row["configurations"][name]=result
            if record.get("runs") and all(s=="SUCCESS" for s in record["status"]):
                if tree is None:tree=cKDTree(np.loadtxt(ROOT/entry["map"]).reshape(-1,3))
                geometry=ROOT/record["output"]/"C3-OPT-native.csv"
                shutil.copy2(geometry,target/f"{name}.csv")
                result["geometry"]=geometry_audit(geometry,entry,tree)
                result["geometry_sha256"]=hashlib.sha256(geometry.read_bytes()).hexdigest()
                successful.append(name)
        if args.samples and successful and successful[0]==args.baseline and len(successful)>1:
            command=[str(binary),str(ROOT/entry["seed"]),str(target),str(args.samples),",".join(successful)]
            start=time.monotonic()
            with (target/"volume.log").open("w") as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            row["volume_command"]=command;row["measurement_s"]=time.monotonic()-start
            for line in (target/"volume_refined.jsonl").read_text().splitlines():
                volume=json.loads(line);assert volume["unresolved"]==0
                row["configurations"][volume["model"]]["volume"]=volume
        results.append(row);(out/"results.json").write_text(json.dumps(results,indent=2)+"\n")
        print(case,{name:round(100*r["volume"]["paired_delta"]/r["volume"]["baseline"],4)
                    for name,r in row["configurations"].items() if "volume" in r},flush=True)


if __name__=="__main__":main()
