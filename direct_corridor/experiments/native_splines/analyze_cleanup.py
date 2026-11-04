#!/usr/bin/env python3
"""Verify the frozen cleanup comparison and archive its complete evidence."""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import tarfile

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
os.sched_setaffinity(0,{2})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'.scratch/native_splines/cleanup'
OUT=ROOT/'docs/validation/native_splines_cleanup'
TRACE=('newton_steps','regularity_repairs','regularity_backtracks','backtracks','termination','predictor_steps','predictor_rejections','volume_steps')
STAGES=('cover_ms','query_ms','separation_ms','setup_ms','assembly_ms','solve_ms','line_search_ms','jacobian_check_ms','volume_ms')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(record,reference):
    assert record['returncode']==reference['returncode']==0
    assert all(s==reference['status'][0] for s in record['status'])
    for key in ('geometry_sha256','support_sha256'):assert record.get(key)==reference.get(key),(record['case'],key)
    if reference['status'][0]!='SUCCESS':return 0
    for run in record['runs']:
        assert run['support_pass'] and run['jacobian_pass']
        for key in TRACE:assert run[key]==reference['runs'][0][key],(record['case'],key)
    assert record['audit']
    for audit in record['audit']:assert audit['status']=='SUCCESS' and audit['sampled_penetration_m']==0
    for name,key in (('C3-OPT-native.csv','geometry_sha256'),('C3-OPT-native_support_planes.csv','support_sha256')):
        assert digest(ROOT/record['output']/name)==reference[key]
    return 2


def archive(path,files):
    seen={}
    with tarfile.open(path,'w:gz',compresslevel=6) as stream:
        for p in files:
            name=str(p.relative_to(ROOT));key=(p.stat().st_size,digest(p))
            if key in seen:
                info=stream.gettarinfo(str(p),arcname=name);info.type=tarfile.LNKTYPE;info.linkname=seen[key];info.size=0
                stream.addfile(info)
            else:stream.add(p,arcname=name);seen[key]=name


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((ROOT/'.scratch/native_splines/e1/inputs/manifest.json').read_text())
    runs={};files_checked=0
    for folder in ('development','expanded'):
        for name,records in json.loads((WORK/folder/'summary.json').read_text()).items():
            runs.setdefault(name,{}).update({r['case']:r for r in records})
    cases=[];peak=0.;gradient=0.;hessian=0.
    for source in inputs['cases']:
        case=source['case'];baseline=runs['frozen'][case];candidate=runs['clean'][case]
        files_checked+=verify(candidate,baseline)+verify(baseline,baseline)
        row=source|dict(status=baseline['status'][0],geometry_sha256=baseline.get('geometry_sha256'),support_sha256=baseline.get('support_sha256'))
        for name,record in (('baseline',baseline),('clean',candidate)):
            row[name+'_ms']=record['median_ms']
            row[name+'_cpu_ms']=statistics.median(r['own_cpu_ms'] for r in record['runs'])
            row[name+'_p95_ms']=float(np.quantile([r['total_ms'] for r in record['runs']],.95))
            row[name+'_index_ms']=statistics.median(r['index_ms'] for r in record['resources'] if 'index_ms' in r)
            row[name+'_rss_mib']=max(r['peak_rss_kib']/1024 for r in record['resources'] if 'peak_rss_kib' in r)
            row[name+'_trace']={key:record['runs'][0][key] for key in TRACE}
            peak=max(peak,row[name+'_rss_mib'])
            if name=='clean':
                for warmup in record['warmup']:
                    gradient=max(gradient,warmup['gradient_reference_error']);hessian=max(hessian,warmup['hessian_reference_error'])
                assert all(r['assembly_ms']==0 and r['query_ms']==0 for r in record['runs'])
        row['time_reduction_pct']=100*(1-row['clean_ms']/row['baseline_ms']);cases.append(row)
    # Each process/block export, including repeat-block hard links, is checked.
    block_checks=0
    for folder in ('development','expanded'):
        for r in json.loads((WORK/folder/'manifest.json').read_text())['cases']:
            reference=runs['frozen'][r['case']]
            if reference['status'][0]!='SUCCESS':continue
            for name,key in (('C3-OPT-native.csv','geometry_sha256'),('C3-OPT-native_support_planes.csv','support_sha256')):
                assert digest(ROOT/r['output']/name)==reference[key];block_checks+=1
    groups={}
    for name,predicate in (('all',lambda r:True),('success',lambda r:r['status']=='SUCCESS'),('development',lambda r:r['case']<7),
                           ('historical',lambda r:r['group']=='historical'),('additional',lambda r:r['group']=='independent'),
                           ('expected_failure',lambda r:r['status']!='SUCCESS')):
        subset=[r for r in cases if predicate(r)]
        total=lambda field:sum(r[field] for r in subset)
        groups[name]=dict(cases=len(subset),baseline_sum_ms=total('baseline_ms'),clean_sum_ms=total('clean_ms'),
            time_reduction_pct=100*(1-total('clean_ms')/total('baseline_ms')),
            cpu_reduction_pct=100*(1-total('clean_cpu_ms')/total('baseline_cpu_ms')),
            cold_time_reduction_pct=100*(1-(total('clean_ms')+total('clean_index_ms'))/(total('baseline_ms')+total('baseline_index_ms'))),
            faster_cases=sum(r['clean_ms']<r['baseline_ms'] for r in subset))
    audit=json.loads((WORK/'presets/summary.json').read_text());presets={}
    for name in ('fast','volume'):
        old={r['case']:r for r in audit['frozen_'+name]}
        checks=sum(verify(r,old[r['case']])+verify(old[r['case']],old[r['case']]) for r in audit[name])
        presets[name]=dict(cases=len(old),file_checks=checks,exact=True)
    old={r['case']:r for r in audit['clean']}
    for record in audit['profiled']:verify(record,old[record['case']])
    profiles=dict(cases=len(old),exact=True,clean_sum_ms=sum(r['median_ms'] for r in audit['clean']),
                  profiled_sum_ms=sum(r['median_ms'] for r in audit['profiled']),
                  stages={key:sum(statistics.median(t[key] for t in r['runs']) for r in audit['profiled']) for key in STAGES})
    result=dict(reference_commit='01c8a5b',inputs=len(cases),maps=len({r['map_sha256'] for r in cases}),groups=groups,cases=cases,
                presets=presets,profiling=profiles,block_hash_checks=block_checks,summary_hash_checks=files_checked,
                peak_rss_mib=peak,max_gradient_reference_error=gradient,max_hessian_reference_error=hessian,
                geometric_change=0,volume_change=0,boundary_smoothness_change=0)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['case','family','group','status','baseline_ms','clean_ms','time_reduction_pct','baseline_cpu_ms','clean_cpu_ms','baseline_p95_ms','clean_p95_ms','baseline_rss_mib','clean_rss_mib','geometry_sha256','support_sha256']
    with (OUT/'case_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
    for name in ('build.log','tests.log','ros_build.log','trajectory_build.log','ros_install.log'):
        if (WORK/name).exists():shutil.copy2(WORK/name,OUT/name)
    shutil.copy2(ROOT/'.scratch/native_splines/e1/inputs/manifest.json',OUT/'input_manifest.json')
    shutil.copy2(ROOT/'.scratch/native_splines/cleanup_baseline/manifest.json',OUT/'baseline_manifest.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));x=np.arange(7)
    axes[0].bar(x-.18,[r['baseline_ms'] for r in cases[:7]],.36,label='E3',color='#8a9bad')
    axes[0].bar(x+.18,[r['clean_ms'] for r in cases[:7]],.36,label='Clean C3',color='#2878b5')
    axes[0].set(xlabel='Frozen case',ylabel='Median generation [ms]',xticks=x);axes[0].legend()
    colors={'historical':'#7c63ae','independent':'#d77930','failure_control':'#a34343'}
    for group in ('development','historical','independent','failure_control'):
        rows=[r for r in cases if ('development' if r['case']<7 else r['group'])==group]
        axes[1].scatter([r['baseline_ms'] for r in rows],[r['time_reduction_pct'] for r in rows],s=24,label=group.replace('_',' '),color=colors.get(group,'#2878b5'))
    axes[1].axhline(0,color='.5',lw=.7);axes[1].set(xlabel='E3 generation [ms]',ylabel='Time reduction [%]');axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('Same cubic coefficients, support planes and Newton decisions')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    folders=('development','expanded','presets')
    archive(OUT/'measurements.tar.gz',sorted(p for name in folders for p in (WORK/name).rglob('*') if p.is_file()))
    source=Path(__file__).parent
    files=sorted(p for p in source.rglob('*') if p.suffix in ('.cpp','.hpp','.py') or p.name in ('CMakeLists.txt','README.md'))
    files+=list(map(lambda name:ROOT/name,('gcopter/CMakeLists.txt','gcopter/src/direct_corridor_demo.cpp','gcopter/experiments/trajectory/benchmark.cpp')))
    archive(OUT/'source_snapshot.tar.gz',files)
    metadata=dict(reference_commit='01c8a5b',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  input_archive='../native_splines_e1/inputs.tar.gz',sources={str(p.relative_to(ROOT)):digest(p) for p in files},
                  artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('groups','presets','profiling','peak_rss_mib','max_gradient_reference_error','max_hessian_reference_error')},indent=2))


if __name__=='__main__':main()
