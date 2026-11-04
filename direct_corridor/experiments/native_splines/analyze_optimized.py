#!/usr/bin/env python3
"""Archive the frozen second-round comparisons and render their report figures."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics as stats
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.scratch/native_splines'
OUT = ROOT / 'docs/validation/native_splines_optimized'
OUT.mkdir(parents=True, exist_ok=True)
PROFILES = ['fast', 'balanced', 'volume', 'native_arch']
MODELS = ['H5-S', 'C3-fast', 'C3-balanced', 'C3-volume', 'C3-OPT-project-round1']
COLOR = {'H5-S':'#bf4438', 'C3-fast':'#23966b', 'C3-balanced':'#127da7',
         'C3-volume':'#df962d', 'C3-OPT-project-round1':'#815faf', 'round1':'#6e7883'}

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def archive(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == '.csv':
        with source.open('rb') as a, gzip.open(str(target)+'.gz','wb') as b:
            shutil.copyfileobj(a,b)
    else:
        shutil.copy2(source,target)

def median(rs, key):
    return stats.median(r[key] for r in rs)

generation = {}
for profile in PROFILES:
    source = WORK/'round2'/profile
    archive(source/'manifest.json',OUT/'generation'/profile/'manifest.json')
    generation[profile] = []
    for case in range(7):
        path=source/f'case_{case:02d}'
        rr=rows(path/'rows.jsonl')
        runs=[r for r in rr if r['kind']=='generation' and r['model']=='C3-OPT-native' and r['repeat']>=0]
        h5=[r for r in rr if r['kind']=='generation' and r['model']=='H5-S' and r['repeat']>=0]
        assert len(runs)==7 and all(r['status']=='SUCCESS' and r['support_pass'] and r['jacobian_pass'] for r in runs)
        fresh=next(r for r in rr if r['kind']=='fresh_map_certificate' and r['model']=='C3-OPT-native')
        assert fresh['status']=='SUCCESS' and fresh['sampled_penetration_m']<=1e-6
        initial=next(r for r in rr if r['kind']=='generation' and r['model']=='C3-OPT-native' and r['repeat']==-1)
        generation[profile].append({'case':case,'median_ms':median(runs,'total_ms'),
          'p95_ms':float(np.percentile([r['total_ms'] for r in runs],95)),
          'h5_ms':median(h5,'total_ms'),'sample':runs[0], 'derivative_check':initial,
          'runs':runs,'stages':{k:median(runs,k) for k in ['initialization_ms','cover_ms','setup_ms','assembly_ms','solve_ms','line_search_ms','jacobian_check_ms','verification_ms']},
          'index_ms':next(r['index_ms'] for r in rr if r['kind']=='input'),
          'peak_rss_kib':next(r['peak_rss_kib'] for r in rr if r['kind']=='resource')})
        for f in path.glob('*'):
            if f.suffix in {'.jsonl','.csv'}:archive(f,OUT/'generation'/profile/path.name/f.name)

volume=[]
joint=WORK/'round2/joint'
archive(joint/'manifest.json',OUT/'geometry'/'manifest.json')
for case in range(7):
    path=joint/f'case_{case:02d}'
    rr=rows(path/'volume_refined.jsonl')
    assert all(r['unresolved']==0 and r['samples']==8*262144 for r in rr)
    volume.append({r['model']:r for r in rr})
    for f in path.glob('*'):archive(f,OUT/'geometry'/path.name/f.name)

trajectory={}
for kind in ['trajectory','trajectory_quintic']:
    source=WORK/'round2'/kind
    archive(source/'manifest.json',OUT/kind/'manifest.json')
    trajectory[kind]={}
    for model_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        values=[]
        for case_dir in sorted(model_dir.glob('case_*')):
            rr=rows(case_dir/'runs.jsonl')
            runs=[r for r in rr if r['method']=='hermite' and r['repeat']>=0]
            values.append({'case':int(case_dir.name[-2:]),'median_ms':median(runs,'optimize_ms'),
              'sample':runs[0], 'gradient_error':max(r['gradient_relative_error'] for r in rr),
              'runs':runs,'peak_rss_kib':int((case_dir/'peak_rss_kib.txt').read_text())})
            for f in case_dir.iterdir():
                if f.name=='firi_trajectory.csv' and model_dir.name!='H5-S':continue
                if f.suffix in {'.jsonl','.csv','.txt'}:archive(f,OUT/kind/model_dir.name/case_dir.name/f.name)
        trajectory[kind][model_dir.name]=values
for case in range(7):
    hashes={data[case]['sample']['initial_hash'] for arms in trajectory.values() for data in arms.values()}
    assert len(hashes)==1

# Exploratory stages retain their own source/binary identities, not a claim
# that different implementation stages formed one controlled parameter grid.
sweeps={}
for name in ['sweep','solver_sweep','progress_sweep','final_sweep']:
    sweeps[name]={}
    for folder in sorted((WORK/name).iterdir()):
        if not folder.is_dir():continue
        archive(folder/'manifest.json',OUT/'exploration'/name/folder.name/'manifest.json')
        rr=[]
        for f in sorted(folder.glob('case_*/rows.jsonl')):
            records=rows(f);rr.append({'case':int(f.parent.name[-2:]),'rows':records})
            archive(f,OUT/'exploration'/name/folder.name/f.parent.name/f.name)
        sweeps[name][folder.name]=rr
for name in ['assembly_probe','grouped_probe','warm_probe','polar_probe']:
    folder=WORK/name
    archive(folder/'manifest.json',OUT/'exploration'/name/'manifest.json')
    for f in folder.glob('case_*/rows.jsonl'):archive(f,OUT/'exploration'/name/f.parent.name/f.name)

old_times=[];assembly_delta=0;arch_delta=0
for case in range(7):
    path=WORK/'final'/f'case_{case:02d}'
    rr=rows(path/'rows.jsonl')
    old_times.append(median([r for r in rr if r['kind']=='generation' and r['model']=='C3-OPT-native' and r['repeat']>=0],'total_ms'))
    original=np.loadtxt(path/'C3-OPT-native.csv',delimiter=',',skiprows=1)
    grouped=np.loadtxt(WORK/'grouped_probe'/path.name/'C3-OPT-native.csv',delimiter=',',skiprows=1)
    assembly_delta=max(assembly_delta,float(np.max(np.abs(original-grouped))))
    portable=np.loadtxt(WORK/'round2/balanced'/path.name/'C3-OPT-native.csv',delimiter=',',skiprows=1)
    accelerated=np.loadtxt(WORK/'round2/native_arch'/path.name/'C3-OPT-native.csv',delimiter=',',skiprows=1)
    arch_delta=max(arch_delta,float(np.max(np.abs(portable-accelerated))))

summary={'generation':generation,'volume':volume,'trajectory':trajectory,
         'old_native_generation_ms':old_times,'assembly_only_max_coefficient_delta':assembly_delta,
         'native_arch_max_coefficient_delta':arch_delta,'exploration_configurations':sum(map(len,sweeps.values()))}
(OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'exploration.json').write_text(json.dumps(sweeps,indent=2)+'\n')

# Preserve the uncommitted first-round generator sources alongside its data.
with tarfile.open(OUT/'round1_generator_sources.tar.gz','w:gz') as tar:
    for p in (WORK/'round2_baseline/source').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            tar.add(p,arcname=str(Path('direct_corridor/experiments/native_splines')/p.relative_to(WORK/'round2_baseline/source')))
    p=ROOT/'direct_corridor/include/gcopter/direct_corridor.hpp'
    tar.add(p,arcname=str(p.relative_to(ROOT)))

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
x=np.arange(7)
fig,axes=plt.subplots(1,3,figsize=(17,4.4),layout='constrained')
axes[0].plot(x,[r['h5_ms'] for r in generation['balanced']], 'o-',color=COLOR['H5-S'],label='H5-S')
axes[0].plot(x,old_times,'o--',color=COLOR['round1'],label='C3 native: round 1')
axes[0].plot(x,[r['median_ms'] for r in generation['balanced']],'o-',color=COLOR['C3-balanced'],label='C3 balanced')
axes[0].set(title='Complete hot generation',ylabel='Time (ms)',xlabel='Frozen case',yscale='log')
for name,label in [('C3-OPT-native-round1','C3 native: round 1'),('C3-OPT-project-round1','C3 projected: reference'),('C3-balanced','C3 balanced'),('C3-volume','C3 volume')]:
    color=COLOR.get(name,COLOR['round1']);y=[100*v[name]['paired_delta']/v[name]['baseline'] for v in volume]
    axes[1].errorbar(x,y,yerr=[100*v[name]['paired_ci95']/v[name]['baseline'] for v in volume],marker='o',color=color,label=label)
axes[1].axhline(0,color=COLOR['H5-S'],lw=1,ls='--');axes[1].set(title='Shared-domain union volume',ylabel='Change from H5-S (%)',xlabel='Frozen case')
for name in ['H5-S','C3-balanced']:
    axes[2].plot(x,[r['median_ms'] for r in trajectory['trajectory'][name]],'o-',color=COLOR[name],label=name)
axes[2].plot(x,[r['median_ms'] for r in trajectory['trajectory_quintic']['C3-balanced']],'o--',color=COLOR['round1'],label='Same C3, elevated query')
axes[2].set(title='Same initial trajectory',ylabel='Optimization (ms)',xlabel='Frozen case')
for ax in axes:ax.set_xticks(x);ax.grid(alpha=.15);ax.legend(fontsize=9)
fig.savefig(OUT/'performance.png',dpi=170);plt.close(fig)

manifest=json.loads((joint/'manifest.json').read_text())
# Keep the small, deduplicated input snapshots with the measurements so a new
# checkout can rerun a preset without the original RViz recording directory.
with tarfile.open(OUT/'frozen_inputs.tar.gz','w:gz') as tar:
    saved=set()
    for entry in manifest['cases']:
        for kind in ['seed','map']:
            path=ROOT/entry[kind]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry[kind+'_sha256']
            if path not in saved:
                tar.add(path,arcname=str(path.relative_to(ROOT)));saved.add(path)

def draw_geometry(ax,case,model,centerline=True):
    entry=manifest['cases'][case];seed=np.loadtxt(ROOT/entry['seed']);obstacles=np.loadtxt(ROOT/entry['map'])
    lo=seed[:,:2].min(axis=0)-3;hi=seed[:,:2].max(axis=0)+3
    z=np.median(seed[:,2]);mask=np.abs(obstacles[:,2]-z)<.14
    ax.scatter(obstacles[mask,0],obstacles[mask,1],s=3,c='#788896',alpha=.24,lw=0,rasterized=True)
    data=np.loadtxt(joint/f'case_{case:02d}'/(model+'.csv'),delimiter=',',skiprows=1)
    degree=int(data[0,0]);controls=data[:,3:].reshape(-1,degree+1,9)
    import math
    t=np.linspace(0,1,9,endpoint=False)
    basis=np.array([math.comb(degree,j)*t**j*(1-t)**(degree-j) for j in range(degree+1)]).T
    y=(basis@controls).reshape(-1,9);y=np.vstack([y,controls[-1,-1]])
    center=y[:,:3];m=y[:,3:].reshape(-1,2,3).swapaxes(1,2)
    theta=np.arange(32)*2*np.pi/32;circle=np.array([np.cos(theta),np.sin(theta)])
    rings=(center[:,:,None]+m@circle)[:,:2,:].transpose(0,2,1)
    size=np.maximum(1,((hi-lo)/max(hi-lo)*1100).astype(int))
    im=Image.new('L',tuple(size),0);draw=ImageDraw.Draw(im)
    pixels=(rings-lo)/(hi-lo)*(size-1)
    for i in range(len(pixels)-1):
        for j in range(32):
            k=(j+1)%32
            polygon=[tuple(v) for v in [pixels[i,j],pixels[i,k],pixels[i+1,k],pixels[i+1,j]]]
            draw.polygon(polygon,fill=255)
    mask=np.asarray(im)/255.;from matplotlib.colors import to_rgb
    rgba=np.zeros((*mask.shape,4));rgba[:,:,:3]=to_rgb(COLOR[model]);rgba[:,:,3]=mask*.23
    ax.imshow(rgba,extent=(*[lo[0],hi[0]],lo[1],hi[1]),origin='lower',interpolation='bilinear',zorder=2)
    ax.contour(np.linspace(lo[0],hi[0],size[0]),np.linspace(lo[1],hi[1],size[1]),mask,levels=[.5],colors=[COLOR[model]],linewidths=.45,zorder=3)
    ax.plot(seed[:,0],seed[:,1],color='#987637',ls='--',lw=.7,alpha=.8,zorder=4)
    if centerline:
        ax.plot(center[:,0],center[:,1],color=COLOR[model],lw=.9,zorder=4)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x (m)',ylabel='y (m)',aspect='equal')

fig,axes=plt.subplots(2,3,figsize=(13,10),layout='constrained')
for row,case in enumerate([1,3]):
    for column,model in enumerate(['H5-S','C3-balanced','C3-OPT-project-round1']):
        draw_geometry(axes[row,column],case,model)
        axes[row,column].set_title(f'Case {case}: '+{'C3-OPT-project-round1':'C3 projection reference'}.get(model,model))
fig.savefig(OUT/'corridors.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
for ax,case in zip(axes,[1,3]):
    draw_geometry(ax,case,'C3-balanced',centerline=False)
    for model in ['H5-S','C3-balanced']:
        trajectory_file=WORK/'round2/trajectory'/model/f'case_{case:02d}'/'hermite_trajectory.csv'
        a=np.genfromtxt(trajectory_file,delimiter=',',names=True)
        ax.plot(a['x'],a['y'],color=COLOR[model],lw=1.4,label=model+' trajectory',zorder=5)
    ax.set_title(f'Case {case}: common optimizer');ax.legend(fontsize=9)
fig.savefig(OUT/'trajectories.png',dpi=170);plt.close(fig)

lines=['| 场景 | H5-S / ms | 原生 C3 首轮 / ms | C3 平衡 / ms | 平衡体积变化 | 体积 / 投影参照 |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
for case in range(7):
    r=generation['balanced'][case];v=volume[case]['C3-balanced'];ref=volume[case]['C3-OPT-project-round1']
    lines.append(f"| {case} | {r['h5_ms']:.2f} | {old_times[case]:.2f} | {r['median_ms']:.2f} | {100*v['paired_delta']/v['baseline']:+.2f}% ± {100*v['paired_ci95']/v['baseline']:.2f}% | {100*v['volume']/ref['volume']:.2f}% |")
(OUT/'generation_table.md').write_text('\n'.join(lines)+'\n')
lines=['| 方案 | 生成中位数 / ms | 优化中位数 / ms | 采样验收 | 目标值 / H5-S（中位数） |',
       '| --- | ---: | ---: | ---: | ---: |']
for model in MODELS:
    ts=trajectory['trajectory'][model]
    generation_median=stats.median(r['h5_ms'] for r in generation['balanced']) if model=='H5-S' else stats.median(r['median_ms'] for r in generation[model[3:]]) if model in ['C3-fast','C3-balanced','C3-volume'] else None
    ratio=stats.median(ts[i]['sample']['objective']/trajectory['trajectory']['H5-S'][i]['sample']['objective'] for i in range(7))
    generation_text=f'{generation_median:.1f}' if generation_median else '冻结几何参照'
    lines.append(f"| {model} | {generation_text} | {stats.median(r['median_ms'] for r in ts):.1f} | {sum(r['sample']['sampled_feasible'] for r in ts)}/7 | {ratio:.5f} |")
(OUT/'trajectory_table.md').write_text('\n'.join(lines)+'\n')
print('Archived results and figures:',OUT)
print('Max assembly coefficient delta:',assembly_delta,'; CPU-build delta:',arch_delta)
