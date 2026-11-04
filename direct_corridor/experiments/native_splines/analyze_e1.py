#!/usr/bin/env python3
"""Archive E1 evidence and draw comparisons without rerunning generation."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import tarfile

for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
os.sched_setaffinity(0,{2})
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/".scratch/native_splines/e1"
OUT=ROOT/"docs/validation/native_splines_e1"


def relative(a,b):
    return 100*(b/a-1) if abs(a)>1e-15 else (0. if a==b else None)


def archive_files(path,files):
    # Preserve every original path while storing identical coefficients/covers
    # only once. Tar hard links restore the complete directory tree on replay.
    seen={}
    with tarfile.open(path,"w:gz",compresslevel=6) as archive:
        for p in files:
            name=str(p.relative_to(ROOT));key=(p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
            if key in seen:
                info=archive.gettarinfo(str(p),arcname=name)
                info.type=tarfile.LNKTYPE;info.linkname=seen[key];info.size=0
                archive.addfile(info)
            else:
                archive.add(p,arcname=name);seen[key]=name


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((WORK/"inputs/manifest.json").read_text())
    measurements={k:json.loads((WORK/k/"summary.json").read_text()) for k in ("final_development","final_expanded")}
    evaluation={r["case"]:r for r in json.loads((WORK/"evaluation/results.json").read_text())}
    refined=json.loads((WORK/"evaluation_refined/results.json").read_text())
    evaluation.update({r["case"]:r for r in refined})
    rows=[]
    for entry in inputs["cases"]:
        i=entry["case"];data=measurements["final_development" if i<7 else "final_expanded"]
        old=next(r for r in data["frozen" if i<7 else "triangle"] if r["case"]==i)
        new=next(r for r in data["hybrid"] if r["case"]==i)
        row=entry|dict(baseline_ms=old["median_ms"],e1_ms=new["median_ms"],
                       time_change_pct=relative(old["median_ms"],new["median_ms"]),
                       baseline_p95_ms=float(np.quantile([r["total_ms"] for r in old["runs"]],.95)),
                       e1_p95_ms=float(np.quantile([r["total_ms"] for r in new["runs"]],.95)),
                       baseline_status=old["status"],e1_status=new["status"],
                       baseline_sample=old["runs"][0],e1_sample=new["runs"][0],
                       baseline_audit=old["audit"],e1_audit=new["audit"],
                       baseline_hash=old.get("geometry_sha256"),e1_hash=new.get("geometry_sha256"),
                       evaluation=evaluation[i])
        row["paired_success"]=all(s=="SUCCESS" for s in old["status"]+new["status"])
        for name,value in [("baseline",old),("e1",new)]:
            row[name+"_gradient_check"]=value["warmup"][0]
            row[name+"_rss_kib"]=next(r["peak_rss_kib"] for r in value["resources"] if "peak_rss_kib" in r)
            row[name+"_stages"]={k:statistics.median(r[k] for r in value["runs"])
                                  for k in ("query_ms","separation_ms","setup_ms","pruning_ms","assembly_ms",
                                            "solve_ms","line_search_ms","verification_ms")}
        if row["paired_success"]:
            for record in old["audit"]+new["audit"]:
                assert record["status"]=="SUCCESS" and record["sampled_penetration_m"]==0
            v=evaluation[i]["volume"];g=evaluation[i]["geometry"]
            row.update(volume_change_pct=100*v["paired_delta"]/v["baseline"],
                       volume_ci95_pct=100*v["paired_ci95"]/v["baseline"],
                       boundary_energy_change_pct=relative(g["C3-before"]["boundary_energy"],g["C3-E1"]["boundary_energy"]),
                       boundary_curvature_p99_change_pct=relative(g["C3-before"]["boundary_curvature_p99"],g["C3-E1"]["boundary_curvature_p99"]),
                       boundary_curvature_max_change_pct=relative(g["C3-before"]["boundary_curvature_max"],g["C3-E1"]["boundary_curvature_max"]),
                       curvature_change_pct=relative(old["runs"][0]["max_curvature"],new["runs"][0]["max_curvature"]),
                       row_change_pct=relative(old["runs"][0]["active_constraints"],new["runs"][0]["active_constraints"]))
            for name,label in [("C3-before","baseline"),("C3-E1","e1")]:
                sample=row[label+"_sample"]
                reference=sample["center_energy"]+.5*sample["shape_energy"]
                assert abs(reference-g[name]["boundary_energy"])<1e-8*(1+reference)
        rows.append(row)
    valid=[r for r in rows if r["paired_success"]]
    groups={}
    for name,selected in [("all",valid),("development",valid[:7])]+[(g,[r for r in valid if r["group"]==g]) for g in ("historical","independent")]:
        groups[name]=dict(cases=len(selected),baseline_sum_ms=sum(r["baseline_ms"] for r in selected),
                          e1_sum_ms=sum(r["e1_ms"] for r in selected),
                          median_time_change_pct=statistics.median(r["time_change_pct"] for r in selected),
                          median_row_change_pct=statistics.median(r["row_change_pct"] for r in selected),
                          volume_up=sum(r["volume_change_pct"]>r["volume_ci95_pct"] for r in selected),
                          volume_down=sum(r["volume_change_pct"] < -r["volume_ci95_pct"] for r in selected),
                          median_volume_change_pct=statistics.median(r["volume_change_pct"] for r in selected),
                          median_boundary_energy_change_pct=statistics.median(r["boundary_energy_change_pct"] for r in selected))
        groups[name]["volume_unresolved_or_identical"]=len(selected)-groups[name]["volume_up"]-groups[name]["volume_down"]
    result=dict(reference_commit="f920f42f732aa317e908df37caed901399b21116",default_reach_bound=0,
                candidate_reach_bound=2,selection="keep E1 opt-in: no consistent time/volume/smoothness improvement",
                groups=groups,cases=rows,refined_cases=[r["case"] for r in refined],
                input_maps=len({r["map_sha256"] for r in rows}),
                max_gradient_error=max(r[k+"_gradient_check"]["gradient_error"] for r in valid for k in ("baseline","e1")),
                max_hessian_error=max(r[k+"_gradient_check"]["hessian_error"] for r in valid for k in ("baseline","e1")),
                peak_generation_rss_mib=max(r[k+"_rss_kib"] for r in rows for k in ("baseline","e1"))/1024,
                max_c2_error=max(g["c2_error"] for r in valid for g in r["evaluation"]["geometry"].values()))
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    fields=["case","group","family","paired_success","baseline_ms","e1_ms","baseline_p95_ms","e1_p95_ms",
            "time_change_pct","volume_change_pct","volume_ci95_pct","boundary_energy_change_pct",
            "boundary_curvature_p99_change_pct","boundary_curvature_max_change_pct","curvature_change_pct","row_change_pct"]
    with (OUT/"case_metrics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    shutil.copy2(WORK/"inputs/manifest.json",OUT/"input_manifest.json")
    shutil.copy2(WORK/"review.json",OUT/"review.json")
    shutil.copy2(ROOT/".scratch/native_splines/build/Testing/Temporary/LastTest.log",OUT/"tests.log")

    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    dev=valid[:7];x=np.arange(7)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    axes[0].bar(x,[-r["time_change_pct"] for r in dev],color="#397aaa")
    axes[0].set(ylabel="Generation time reduction [%]",title="Positive = faster")
    axes[1].errorbar(x,[r["volume_change_pct"] for r in dev],yerr=[r["volume_ci95_pct"] for r in dev],fmt="o",color="#a54548",capsize=3)
    axes[1].set(ylabel="Union volume change [%]",title="Paired QMC, 95% intervals")
    axes[2].bar(x,[r["boundary_energy_change_pct"] for r in dev],color="#bd7c39")
    axes[2].set(ylabel="Boundary bending energy change [%]",title="Negative = smoother by this metric")
    for ax in axes:ax.axhline(0,color="gray",lw=.8);ax.set_xticks(x);ax.set_xlabel("Frozen case");ax.grid(axis="y",alpha=.2)
    fig.suptitle("E1 exact reach: fewer support rows do not ensure better final geometry")
    fig.tight_layout();fig.savefig(OUT/"development-comparison.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for family in dict.fromkeys(r["family"] for r in valid):
        a=[r for r in valid if r["family"]==family]
        axes[0].scatter([r["volume_change_pct"] for r in a],[-r["time_change_pct"] for r in a],label=family,s=25,alpha=.8)
        axes[1].scatter([r["volume_change_pct"] for r in a],[r["boundary_energy_change_pct"] for r in a],s=25,alpha=.8)
    for ax in axes:ax.axhline(0,color="gray",lw=.8);ax.axvline(0,color="gray",lw=.8);ax.grid(alpha=.2);ax.set_xlabel("Union volume change [%]")
    axes[0].set_ylabel("Generation time reduction [%]");axes[1].set_ylabel("Boundary energy change [%]")
    axes[0].legend(fontsize=7,frameon=False)
    fig.suptitle("65 successful pairs; full suite: 67 cases on 16 maps, including 2 failure controls")
    fig.tight_layout();fig.savefig(OUT/"expanded-comparison.png",dpi=180);plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    for number,case in enumerate((51,54,57,60)):
        row=rows[case];ax=fig.add_subplot(2,2,number+1,projection="3d")
        cloud=np.loadtxt(ROOT/row["map"]);seed=np.loadtxt(ROOT/row["seed"])
        cloud=cloud[::max(1,len(cloud)//1600)]
        ax.scatter(*cloud.T,s=.5,c="#81919b",alpha=.13,rasterized=True)
        raw=np.loadtxt(WORK/f"evaluation/case_{case:02d}/C3-E1.csv",delimiter=",",skiprows=1).reshape(-1,4,12)
        t=np.linspace(0,1,3);basis=np.array([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3]).T
        y=np.einsum("tj,sjk->stk",basis,raw[:,:,3:]).reshape(-1,9)
        angle=np.linspace(0,2*np.pi,25);disk=np.array([np.cos(angle),np.sin(angle)])
        shape=y[:,3:].reshape(-1,2,3).swapaxes(-1,-2)
        surface=y[:,:3,None]+shape@disk
        ax.plot_surface(surface[:,0],surface[:,1],surface[:,2],color="#cf554c",alpha=.42,linewidth=0,rcount=100,ccount=25)
        ax.plot(*seed.T,color="#2a536a",lw=1)
        ax.set_title(f"{row['family']} (case {case})");ax.set_xlabel("x [m]");ax.set_ylabel("y [m]");ax.set_zlabel("z [m]")
        ax.set_box_aspect(np.asarray(row["bounds_upper"])-row["bounds_lower"]);ax.view_init(27,-60)
    fig.suptitle("Additional frozen environments: E1 corridors and seed paths")
    fig.tight_layout();fig.savefig(OUT/"additional-environments.png",dpi=170);plt.close(fig)

    # Preserve complete time sequences, coefficients, covers and failed runs.
    archive_files(OUT/"inputs.tar.gz",[p for p in sorted((WORK/"inputs").glob("*")) if p.is_file()])
    files=[]
    for name in ("pilot","development","expanded","final_development","final_expanded","evaluation","evaluation_refined"):
        files.extend(p for p in sorted((WORK/name).rglob("*")) if p.is_file())
    files.extend([ROOT/".scratch/native_splines/e1_baseline/manifest.json",WORK/"review.json"])
    archive_files(OUT/"measurements.tar.gz",files)
    manifest=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.rglob("*") if p.suffix in (".cpp",".hpp",".py",".txt")},
                  compiler=subprocess.check_output(["c++","--version"],text=True).splitlines()[0],
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))


if __name__=="__main__":main()
