#!/usr/bin/env python3
"""Archive E3 timings and prove exact output equality against the frozen binary."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
from analyze_e1 import ROOT,archive_files,np,plt

WORK=ROOT/".scratch/native_splines/e3"
OUT=ROOT/"docs/validation/native_splines_e3"
STAGES=("initialization_ms","cover_ms","query_ms","separation_ms","setup_ms","initial_problem_ms",
        "pruning_ms","assembly_ms","solve_ms","line_search_ms","jacobian_check_ms","verification_ms")


def verify_audit(record,reference):
    status=reference["status"][0]
    assert all(s==status for s in record["status"])
    assert record["returncode"]==reference["returncode"]
    for key in ("geometry_sha256","support_sha256"):
        assert record.get(key)==reference.get(key),(record["case"],key)
    if status!="SUCCESS":return 0
    for run in record["runs"]:
        assert run["support_pass"] and run["jacobian_pass"]
        for key in ("newton_steps","regularity_repairs","regularity_backtracks","backtracks","termination"):
            assert run[key]==reference["runs"][0][key],(record["case"],key)
    assert record["audit"]
    for audit in record["audit"]:
        assert audit["status"]=="SUCCESS" and audit["sampled_penetration_m"]==0
    for file,key in (("C3-OPT-native.csv","geometry_sha256"),("C3-OPT-native_support_planes.csv","support_sha256")):
        assert hashlib.sha256((ROOT/record["output"]/file).read_bytes()).hexdigest()==reference[key]
    return 2


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((ROOT/".scratch/native_splines/e1/inputs/manifest.json").read_text())
    runs={}
    for directory in ("development","expanded"):
        source=json.loads((WORK/directory/"summary.json").read_text())
        for name,records in source.items():runs.setdefault(name,{}).update({r["case"]:r for r in records})
    rows=[];peak=0.;gradient=0.;hessian=0.;c2=0.;block_hash_checks=0
    for entry in inputs["cases"]:
        case=entry["case"];baseline=runs["frozen"][case];legacy=runs["legacy"][case];candidate=runs["all"][case]
        status=baseline["status"][0]
        assert all(s==status for record in (baseline,legacy,candidate) for s in record["status"])
        for record in (legacy,candidate):
            assert record.get("geometry_sha256")==baseline.get("geometry_sha256")
            assert record.get("support_sha256")==baseline.get("support_sha256")
        row=entry|dict(baseline_status=status,candidate_status=candidate["status"][0],success=status=="SUCCESS",
                       baseline_ms=baseline["median_ms"],legacy_ms=legacy["median_ms"],candidate_ms=candidate["median_ms"],
                       geometry_sha256=baseline.get("geometry_sha256"),support_sha256=baseline.get("support_sha256"))
        for label,record in (("baseline",baseline),("legacy",legacy),("candidate",candidate)):
            row[label+"_cpu_ms"]=statistics.median(r["own_cpu_ms"] for r in record["runs"])
            row[label+"_p95_ms"]=float(np.quantile([r["total_ms"] for r in record["runs"]],.95))
            row[label+"_sample"]=record["runs"][0]
            row[label+"_stages"]={k:statistics.median(r.get(k,0.) for r in record["runs"]) for k in STAGES}
            row[label+"_index_ms"]=statistics.median(r["index_ms"] for r in record["resources"] if "index_ms" in r)
            row[label+"_rss_mib"]=max(r["peak_rss_kib"]/1024 for r in record["resources"] if "peak_rss_kib" in r)
            peak=max(peak,row[label+"_rss_mib"])
            for check in record["warmup"]:
                gradient=max(gradient,check["gradient_reference_error"]);hessian=max(hessian,check["hessian_reference_error"])
            if row["success"]:
                for result in record["runs"]:
                    assert result["support_pass"] and result["jacobian_pass"]
                    for key in ("newton_steps","regularity_repairs","regularity_backtracks","backtracks","termination"):
                        assert result[key]==baseline["runs"][0][key],(case,label,key)
                for audit in record["audit"]:assert audit["status"]=="SUCCESS" and audit["sampled_penetration_m"]==0
                c2=max(c2,record["runs"][0]["c2_error"])
                geometry=ROOT/record["output"]/"C3-OPT-native.csv"
                assert hashlib.sha256(geometry.read_bytes()).hexdigest()==row["geometry_sha256"]
        row["time_reduction_pct"]=100*(1-row["candidate_ms"]/row["baseline_ms"])
        row["cpu_reduction_pct"]=100*(1-row["candidate_cpu_ms"]/row["baseline_cpu_ms"])
        row["cold_baseline_ms"]=row["baseline_ms"]+row["baseline_index_ms"]
        row["cold_candidate_ms"]=row["candidate_ms"]+row["candidate_index_ms"]
        rows.append(row)
    # Audit every exported block, not just the first one used by the summary.
    for directory in ("development","expanded"):
        source=json.loads((WORK/directory/"manifest.json").read_text())
        for record in source["cases"]:
            reference=runs["frozen"][record["case"]]
            for file,key in (("C3-OPT-native.csv","geometry_sha256"),("C3-OPT-native_support_planes.csv","support_sha256")):
                p=ROOT/record["output"]/file
                if key in reference:assert hashlib.sha256(p.read_bytes()).hexdigest()==reference[key];block_hash_checks+=1
    default=json.loads((WORK/"default_audit/summary.json").read_text())["default"]
    assert {r["case"] for r in default}=={r["case"] for r in inputs["cases"]}
    default_hash_checks=sum(verify_audit(record,runs["frozen"][record["case"]]) for record in default)
    combinations=json.loads((WORK/"e2_audit/summary.json").read_text())
    e2_checks={}
    for mode in ("elevate","split"):
        reference={r["case"]:r for r in combinations["frozen_"+mode]}
        candidate={r["case"]:r for r in combinations[mode+"_cached"]}
        assert candidate.keys()==reference.keys()
        hashes=sum(verify_audit(reference[case],reference[case])+verify_audit(record,reference[case])
                   for case,record in candidate.items())
        e2_checks[mode]=dict(cases=len(candidate),file_hash_checks=hashes,exact=True)
    audits=dict(default_preset=dict(cases=len(default),file_hash_checks=default_hash_checks,exact=True),
                e2_combinations=e2_checks)
    groups={}
    for name,predicate in (("all",lambda r:True),("success",lambda r:r["success"]),
                           ("development",lambda r:r["case"]<7),("historical",lambda r:r["group"]=="historical"),
                           ("independent",lambda r:r["group"]=="independent"),
                           ("expected_failure",lambda r:not r["success"])):
        selected=[r for r in rows if predicate(r)];total=lambda key:sum(r[key] for r in selected)
        groups[name]=dict(cases=len(selected),baseline_sum_ms=total("baseline_ms"),candidate_sum_ms=total("candidate_ms"),
                          time_reduction_pct=100*(1-total("candidate_ms")/total("baseline_ms")),
                          cpu_reduction_pct=100*(1-total("candidate_cpu_ms")/total("baseline_cpu_ms")),
                          cold_time_reduction_pct=100*(1-total("cold_candidate_ms")/total("cold_baseline_ms")),
                          median_case_reduction_pct=statistics.median(r["time_reduction_pct"] for r in selected),
                          faster_cases=sum(r["candidate_ms"]<r["baseline_ms"] for r in selected),
                          stages={label:{k:sum(r[label+"_stages"][k] for r in selected) for k in STAGES} for label in ("legacy","candidate")})
    ablations=[]
    for name in runs:
        selected=[runs[name][i] for i in range(7)]
        total=sum(r["median_ms"] for r in selected)
        ablations.append(dict(configuration=name,total_ms=total,
                              reduction_pct=100*(1-total/groups["development"]["baseline_sum_ms"]),
                              setup_ms=sum(statistics.median(v.get("setup_ms",0) for v in r["runs"]) for r in selected)))
    result=dict(reference_commit="3a34a75",input_cases=len(rows),input_maps=len({r["map_sha256"] for r in rows}),
                groups=groups,ablations=ablations,cases=rows,block_hash_checks=block_hash_checks,audits=audits,
                peak_generation_rss_mib=peak,max_gradient_reference_error=gradient,max_hessian_reference_error=hessian,
                max_c2_error=c2,geometry_comparison="byte-identical cubic coefficients and original support planes",
                volume_and_boundary_change=0.,volume_sampling="not repeated: exactly equal geometry")
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    fields=["case","family","group","success","baseline_status","candidate_status","baseline_ms","candidate_ms",
            "time_reduction_pct","baseline_cpu_ms","candidate_cpu_ms","cpu_reduction_pct","baseline_p95_ms","candidate_p95_ms",
            "cold_baseline_ms","cold_candidate_ms","geometry_sha256","support_sha256"]
    with (OUT/"case_metrics.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    shutil.copy2(ROOT/".scratch/native_splines/e1/inputs/manifest.json",OUT/"input_manifest.json")
    shutil.copy2(ROOT/".scratch/native_splines/build/Testing/Temporary/LastTest.log",OUT/"tests.log")
    shutil.copy2(WORK/"environment.json",OUT/"environment.json")
    shutil.copy2(ROOT/".scratch/native_splines/e3_baseline/manifest.json",OUT/"baseline_manifest.json")
    for name in ("ros_build.log","ros_install.log"):
        if (WORK/name).exists():shutil.copy2(WORK/name,OUT/name)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    dev=rows[:7];x=np.arange(7)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(x-.18,[r["baseline_ms"] for r in dev],.36,label="Frozen C3",color="#8a9bad")
    axes[0].bar(x+.18,[r["candidate_ms"] for r in dev],.36,label="E3 cache",color="#2878b5")
    axes[0].set(xlabel="Frozen case",ylabel="Median generation [ms]",xticks=x);axes[0].legend()
    axes[1].bar(x,[r["time_reduction_pct"] for r in dev],color="#2878b5")
    axes[1].set(xlabel="Frozen case",ylabel="Time reduction [%]",xticks=x)
    fig.suptitle("Same cubic coefficients and support planes, byte for byte")
    fig.tight_layout();fig.savefig(OUT/"development.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    colors={"development":"#2878b5","historical":"#7c63ae","independent":"#d77930","failure_control":"#a34343"}
    names={"development":"Development","historical":"Historical","independent":"Additional","failure_control":"Expected failure"}
    for group in ("development","historical","independent","failure_control"):
        selected=[r for r in rows if ("development" if r["case"]<7 else r["group"])==group]
        if not selected:continue
        axes[0].scatter([r["baseline_ms"] for r in selected],[r["time_reduction_pct"] for r in selected],s=25,label=names[group],color=colors[group])
    axes[0].axhline(0,color="0.5",lw=.6);axes[0].set(xlabel="Frozen generation [ms]",ylabel="Time reduction [%]");axes[0].legend(fontsize=8,loc="lower right")
    labels=("Preparation","Pruning","Cover","Newton assembly","Linear solve")
    keys=("setup_ms","pruning_ms","cover_ms","assembly_ms","solve_ms")
    for label,color,offset in (("legacy","#8a9bad",-.18),("candidate","#2878b5",.18)):
        axes[1].bar(np.arange(5)+offset,[groups["success"]["stages"][label][k] for k in keys],.36,label=label,color=color)
    axes[1].set(xticks=np.arange(5),xticklabels=labels,ylabel="Sum of per-case medians [ms]")
    axes[1].tick_params(axis="x",rotation=20);axes[1].legend(fontsize=8)
    fig.suptitle("67 frozen inputs; pruning is a subset of preparation")
    fig.tight_layout();fig.savefig(OUT/"regression.png",dpi=160);plt.close(fig)
    folders=[p for p in WORK.iterdir() if p.is_dir()]
    files=sorted(p for folder in folders for p in folder.rglob("*") if p.is_file())
    archive_files(OUT/"measurements.tar.gz",files)
    source=Path(__file__).parent
    source_files=sorted(p for p in source.rglob("*") if p.suffix in (".cpp",".hpp",".py") or p.name in ("CMakeLists.txt","README.md"))
    archive_files(OUT/"source_snapshot.tar.gz",source_files)
    manifest=dict(reference_commit="3a34a75",input_archive="../native_splines_e1/inputs.tar.gz",
                  sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"groups":groups,"ablations":ablations,"peak_mib":peak},indent=2))


if __name__=="__main__":main()
