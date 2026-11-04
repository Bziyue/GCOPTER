#!/usr/bin/env python3
"""Summarize the E2 budget scan and serial, interleaved regression runs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from analyze_e1 import ROOT,archive_files,relative,np,plt

WORK=ROOT/".scratch/native_splines/e2"
OUT=ROOT/"docs/validation/native_splines_e2"
NAMES=("elevate05","split05")


def median(values):
    values=[v for v in values if v is not None]
    return statistics.median(values) if values else None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((ROOT/".scratch/native_splines/e1/inputs/manifest.json").read_text())
    runs={name:{} for name in ("frozen","baseline",*NAMES)}
    for directory in ("paired_development","paired_expanded"):
        data=json.loads((WORK/directory/"summary.json").read_text())
        for name,records in data.items():runs[name].update({r["case"]:r for r in records})
    evaluation={r["case"]:r for r in json.loads((WORK/"evaluation/results.json").read_text())}
    rows=[];peak=0;gradient=0;hessian=0;continuity=0
    for entry in inputs["cases"]:
        case=entry["case"];base=runs["baseline"][case]
        for name in NAMES:
            candidate=runs[name][case]
            row=dict(case=case,family=entry["family"],group=entry["group"],configuration=name,
                     baseline_ms=base["median_ms"],candidate_ms=candidate["median_ms"],
                     baseline_cpu_ms=median([r["own_cpu_ms"] for r in base["runs"]]),
                     candidate_cpu_ms=median([r["own_cpu_ms"] for r in candidate["runs"]]),
                     baseline_p95_ms=float(np.quantile([r["total_ms"] for r in base["runs"]],.95)),
                     candidate_p95_ms=float(np.quantile([r["total_ms"] for r in candidate["runs"]],.95)),
                     baseline_status=base["status"],candidate_status=candidate["status"])
            row["time_change_pct"]=relative(row["baseline_ms"],row["candidate_ms"])
            row["cpu_change_pct"]=relative(row["baseline_cpu_ms"],row["candidate_cpu_ms"])
            row["paired_success"]=all(s=="SUCCESS" for s in base["status"]+candidate["status"])
            row["selection_ms"]=median([r["support_selection_ms"] for r in candidate["runs"]])
            row["candidate_sample"]=candidate["runs"][0];row["baseline_sample"]=base["runs"][0]
            for record in (base,candidate):
                peak=max(peak,*[r["peak_rss_kib"]/1024 for r in record["resources"] if "peak_rss_kib" in r])
                for check in record["warmup"]:
                    gradient=max(gradient,check["gradient_error"]);hessian=max(hessian,check["hessian_error"])
            if row["paired_success"]:
                for audit in base["audit"]+candidate["audit"]:
                    assert audit["status"]=="SUCCESS" and audit["sampled_penetration_m"]==0
                ev=evaluation[case]["configurations"];v=ev[name]["volume"]
                row["volume_change_pct"]=100*v["paired_delta"]/v["baseline"]
                row["volume_ci95_pct"]=100*v["paired_ci95"]/v["baseline"]
                row["baseline_volume_m3"]=v["baseline"];row["candidate_volume_m3"]=v["volume"]
                for metric in ("boundary_energy","boundary_curvature_p99","boundary_curvature_max"):
                    row[metric+"_change_pct"]=relative(ev["baseline"]["geometry"][metric],ev[name]["geometry"][metric])
                row["row_change_pct"]=relative(base["runs"][0]["active_constraints"],candidate["runs"][0]["active_constraints"])
                continuity=max(continuity,ev[name]["geometry"]["c2_error"],ev["baseline"]["geometry"]["c2_error"])
            rows.append(row)
    groups={}
    for name in NAMES:
        groups[name]={}
        for group in ("all","development","historical","independent"):
            selected=[r for r in rows if r["configuration"]==name and r["paired_success"]
                      and (group=="all" or (r["case"]<7 if group=="development" else r["group"]==group))]
            total=lambda k:sum(r[k] for r in selected)
            groups[name][group]=dict(cases=len(selected),baseline_sum_ms=total("baseline_ms"),candidate_sum_ms=total("candidate_ms"),
                sum_time_change_pct=relative(total("baseline_ms"),total("candidate_ms")),
                sum_cpu_change_pct=relative(total("baseline_cpu_ms"),total("candidate_cpu_ms")),
                median_time_change_pct=median([r["time_change_pct"] for r in selected]),
                median_volume_change_pct=median([r["volume_change_pct"] for r in selected]),
                median_boundary_energy_change_pct=median([r["boundary_energy_change_pct"] for r in selected]),
                median_row_change_pct=median([r["row_change_pct"] for r in selected]),
                volume_up=sum(r["volume_change_pct"]>r["volume_ci95_pct"] for r in selected),
                volume_down=sum(r["volume_change_pct"] < -r["volume_ci95_pct"] for r in selected),
                volume_unresolved=sum(abs(r["volume_change_pct"])<=r["volume_ci95_pct"] for r in selected))
    # Large-budget runs use the same final geometry as the pilot; keep their
    # paired QMC uncertainty instead of interpreting sub-error changes.
    scan=json.loads((WORK/"final_development/summary.json").read_text())
    pilot=json.loads((WORK/"pilot/summary.json").read_text())
    ev_scan=json.loads((WORK/"pilot_evaluation/results.json").read_text())
    budget=[]
    for name in scan:
        if name in ("baseline","frozen"):continue
        assert all(a["geometry_sha256"]==b["geometry_sha256"] for a,b in zip(scan[name],pilot[name]))
        values=[r["configurations"][name] for r in ev_scan]
        budget.append(dict(configuration=name,
            time_change_pct=relative(sum(r["median_ms"] for r in scan["baseline"]),sum(r["median_ms"] for r in scan[name])),
            median_volume_change_pct=median([100*r["volume"]["paired_delta"]/r["volume"]["baseline"] for r in values]),
            median_boundary_energy_change_pct=median([relative(r["configurations"]["baseline"]["geometry"]["boundary_energy"],r["configurations"][name]["geometry"]["boundary_energy"]) for r in ev_scan])))
    for case,record in runs["frozen"].items():assert record["geometry_sha256"]==runs["baseline"][case]["geometry_sha256"]
    diagnostic=json.loads((WORK/"reference_audit/summary.json").read_text())
    corrected_gradient=0;corrected_hessian=0;unchanged=0
    reference_gradient=0;reference_hessian=0;minimum_step=1;validation_peak=0
    for name,records in diagnostic.items():
        for record in records:
            reference=runs[name][record["case"]]
            assert record["status"][0]==reference["status"][0]
            assert record.get("geometry_sha256")==reference.get("geometry_sha256")
            unchanged+=1
            validation_peak=max(validation_peak,*[r["peak_rss_kib"]/1024 for r in record["resources"] if "peak_rss_kib" in r])
            for check in record["warmup"]:
                corrected_gradient=max(corrected_gradient,check["gradient_error"])
                corrected_hessian=max(corrected_hessian,check["hessian_error"])
                reference_gradient=max(reference_gradient,check["gradient_reference_error"])
                reference_hessian=max(reference_hessian,check["hessian_reference_error"])
                minimum_step=min(minimum_step,check["derivative_min_step"])
    result=dict(reference_commit="5c6f860",default_certificate_refinement=0,groups=groups,cases=rows,budget_scan=budget,
                input_cases=len(inputs["cases"]),input_maps=len({r["map_sha256"] for r in inputs["cases"]}),
                peak_generation_rss_mib=peak,peak_validation_rss_mib=validation_peak,max_c2_error=continuity,
                original_fixed_step_gradient_error=gradient,original_fixed_step_hessian_error=hessian,
                max_gradient_error=corrected_gradient,max_hessian_error=corrected_hessian,
                max_gradient_reference_error=reference_gradient,max_hessian_reference_error=reference_hessian,
                derivative_min_step=minimum_step,
                diagnostic_unchanged_configurations=unchanged,
                frozen_geometry_identical_cases=len(runs["frozen"]))
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    fields=["case","group","family","configuration","paired_success","baseline_ms","candidate_ms","time_change_pct",
            "baseline_cpu_ms","candidate_cpu_ms","cpu_change_pct","baseline_p95_ms","candidate_p95_ms",
            "volume_change_pct","volume_ci95_pct","boundary_energy_change_pct","boundary_curvature_p99_change_pct",
            "boundary_curvature_max_change_pct","row_change_pct","selection_ms"]
    with (OUT/"case_metrics.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    shutil.copy2(ROOT/".scratch/native_splines/e1/inputs/manifest.json",OUT/"input_manifest.json")
    shutil.copy2(ROOT/".scratch/native_splines/build/Testing/Temporary/LastTest.log",OUT/"tests.log")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colors={"elevate05":"#2878b5","split05":"#d77930"}
    labels={"elevate05":"Degree elevation, 5%","split05":"Bisection, 5%"}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8));x=np.arange(7)
    for i,name in enumerate(NAMES):
        data=[r for r in rows if r["configuration"]==name and r["case"]<7]
        for ax,key in zip(axes,("time_change_pct","volume_change_pct","boundary_energy_change_pct")):
            ax.bar(x+(i-.5)*.35,[r[key] for r in data],.35,label=labels[name],color=colors[name],
                   yerr=[r["volume_ci95_pct"] for r in data] if key=="volume_change_pct" else None,capsize=2)
    for ax,title in zip(axes,("Generation time: lower is better","Union volume: higher is better","Boundary energy: lower is smoother")):
        ax.set(title=title,xlabel="Frozen case",ylabel="Change [%]",xticks=x);ax.axhline(0,color="0.4",lw=.6)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/"development.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in NAMES:
        data=[r for r in rows if r["configuration"]==name and r["paired_success"]]
        for ax,key in zip(axes,("time_change_pct","boundary_energy_change_pct")):
            usable=[r for r in data if r[key] is not None]
            ax.scatter([r[key] for r in usable],[r["volume_change_pct"] for r in usable],s=20,alpha=.7,color=colors[name],label=labels[name])
    for ax in axes:ax.axhline(0,color="0.5",lw=.7);ax.axvline(0,color="0.5",lw=.7);ax.set_ylabel("Union volume change [%]")
    axes[0].set_xlabel("Generation time change [%]");axes[1].set_xlabel("Boundary energy change [%]");axes[0].legend(fontsize=8)
    fig.suptitle("65 successful frozen cases; 2 expected failures retained in raw results")
    fig.tight_layout();fig.savefig(OUT/"regression.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    t=np.linspace(0,1,301);ax.plot(t,.5+3.3*t*(1-t),color="#242b35",lw=2.5,label="Actual cubic boundary")
    ax.plot(np.arange(4)/3,[.5,1.6,1.6,.5],"o--",color="#c34742",label="Cubic control hull")
    ax.plot(np.arange(6)/5,[.5,1.16,1.49,1.49,1.16,.5],"s--",color="#2878b5",label="Exact degree-five elevation")
    ax.plot(np.arange(7)/6,[.5,1.05,1.325,1.325,1.325,1.05,.5],"^--",color="#d77930",label="One bisection")
    ax.axhline(1.51,color="#6a6773",ls=":",label="Same separating plane")
    ax.set(xlabel="Local parameter",ylabel="Support in a fixed direction [m]",ylim=(.43,1.72))
    ax.legend(fontsize=8,loc="lower center");fig.tight_layout();fig.savefig(OUT/"certificate_example.png",dpi=160);plt.close(fig)
    files=sorted(p for folder in ("pilot","pilot_evaluation","final_development","paired_development","paired_expanded","evaluation","gradient_pilot","derivative_audit","reference_audit")
                 for p in (WORK/folder).rglob("*") if p.is_file())
    archive_files(OUT/"measurements.tar.gz",files)
    source=Path(__file__).parent
    source_files=sorted(p for p in source.rglob("*") if p.suffix in (".cpp",".hpp",".py") or p.name in ("CMakeLists.txt","README.md"))
    archive_files(OUT/"source_snapshot.tar.gz",source_files)
    manifest=dict(reference_commit="5c6f860",input_archive="../native_splines_e1/inputs.tar.gz",
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"groups":groups,"budget_scan":budget,"peak_mib":peak},indent=2))


if __name__=="__main__":main()
