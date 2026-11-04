#!/usr/bin/env python3
"""Serial E2 certificate-budget comparisons on the E1 frozen inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--cases",default="all")
    parser.add_argument("--configurations",default="baseline,elevate05,elevate10,elevate25,split05,split10,split25")
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--cpu",type=int,default=2)
    parser.add_argument("--blocks",type=int,default=1)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
    inputs=json.loads((ROOT/".scratch/native_splines/e1/inputs/manifest.json").read_text())["cases"]
    if args.cases!="all":
        selected=set(map(int,args.cases.split(",")));inputs=[r for r in inputs if r["case"] in selected]
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    names=args.configurations.split(",")
    configurations={"baseline":{},"frozen":{}}
    for kind,mode in (("elevate",1),("split",2)):
        for budget in (5,10,25,50,100):
            configurations[f"{kind}{budget:02d}"]={"NATIVE_CERTIFICATE_REFINEMENT":str(mode),"NATIVE_CERTIFICATE_FRACTION":str(budget/100)}
    assert all(name in configurations for name in names)
    binaries={name:ROOT/(".scratch/native_splines/e2_baseline/native_spline_benchmark" if name=="frozen"
                        else ".scratch/native_splines/build/native_spline_benchmark") for name in names}
    manifest=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  configurations=configurations,repeats=args.repeats,blocks=args.blocks,cpu=args.cpu,address_space_limit_bytes=4*1024**3,
                  binaries={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in binaries.items()},
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).parent.rglob("*") if p.suffix in (".cpp",".hpp",".py")},cases=[])
    summary={name:[] for name in names}
    for row in inputs:
        case=row["case"]
        for kind in ("seed","map"):
            assert hashlib.sha256((ROOT/row[kind]).read_bytes()).hexdigest()==row[kind+"_sha256"]
        for block in range(args.blocks):
            order=names if (case+block)%2==0 else names[::-1]
            order=order[block%len(order):]+order[:block%len(order)]
            for name in order:
                target=out/name/f"case_{case:02d}"
                if args.blocks>1:target=target/f"block_{block:02d}"
                target.mkdir(parents=True,exist_ok=True)
                env={k:v for k,v in os.environ.items() if not k.startswith("NATIVE_")}
                env.update(NATIVE_CANDIDATES="1",NATIVE_PRESET="balanced",NATIVE_GENERATION_ONLY="1",NATIVE_METHOD_COUNT="1")
                env.update(NATIVE_SKIP_H5="1",NATIVE_REACH_BOUND="0")
                env.update(configurations[name])
                command=[str(binaries[name]),str(ROOT/row["seed"]),str(ROOT/row["map"]),str(target),
                         *map(str,row["bounds_lower"]+row["bounds_upper"]),str(args.repeats),"0"]
                start=time.monotonic()
                with (target/"process.log").open("w") as log:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                record=row|dict(configuration=name,block=block,command=command,returncode=result.returncode,
                                elapsed_s=time.monotonic()-start,output=str(target.relative_to(ROOT)))
                manifest["cases"].append(record)
                (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
                rows=[json.loads(line) for line in (target/"rows.jsonl").read_text().splitlines()]
                runs=[r for r in rows if r.get("kind")=="generation" and r.get("model")=="C3-OPT-native" and r["repeat"]>=0]
                value=dict(case=case,family=row["family"],group=row["group"],returncode=result.returncode,
                           output=record["output"],runs=runs,
                           warmup=[r for r in rows if r.get("kind")=="generation" and r.get("model")=="C3-OPT-native" and r["repeat"]==-1],
                           audit=[r for r in rows if r.get("kind")=="fresh_map_certificate" and r.get("model")=="C3-OPT-native"],
                           resources=[r for r in rows if r.get("kind") in ("input","resource")])
                if runs:
                    value["median_ms"]=statistics.median(r["total_ms"] for r in runs)
                    value["status"]=[r["status"] for r in runs]
                    geometry=target/"C3-OPT-native.csv"
                    if all(r["status"]=="SUCCESS" for r in runs):
                        value["geometry_sha256"]=hashlib.sha256(geometry.read_bytes()).hexdigest()
                value["block_medians_ms"]=[value.get("median_ms")]
                if block:
                    previous=summary[name][-1];assert previous["case"]==case
                    assert previous.get("geometry_sha256")==value.get("geometry_sha256")
                    for key in ("runs","warmup","audit","resources","status","block_medians_ms"):previous[key].extend(value.get(key,[]))
                    previous["median_ms"]=statistics.median(r["total_ms"] for r in previous["runs"])
                else:summary[name].append(value)
                (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
                print(case,block,name,value.get("status",["PROCESS_FAILED"])[0],round(value.get("median_ms",0),3),
                      "ms refined planes",runs[0].get("refined_planes",0) if runs else 0,flush=True)


if __name__=="__main__":main()
