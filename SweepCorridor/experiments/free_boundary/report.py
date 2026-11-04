#!/usr/bin/env python3
"""Summarize paired runs and draw frozen geometries; never regenerate paths."""
import argparse,csv,itertools,json,os,resource,statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--development',default='.scratch/free_boundary/final_development')
p.add_argument('--regression',default='.scratch/free_boundary/final_regression')
p.add_argument('--output',default='docs/validation/free_boundary')
p.add_argument('--firi',default='.scratch/free_boundary/firi/firi_planes.csv')
a=p.parse_args();os.chdir(Path(__file__).resolve().parents[3])
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.font_manager import FontProperties
from geometry_metrics import bending

out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
datasets={name:json.loads((Path(root)/'summary.json').read_text()) for name,root in [('development',a.development),('regression',a.regression)]}
rows=[r for data in datasets.values() for r in data]
base={r['case']:r for r in rows if r['mode']=='baseline'}
candidate={r['case']:r for r in rows if r['mode']=='free_boundary'}
inputs={r['case']:r for r in json.loads(Path('docs/validation/inflation_round1/inputs.json').read_text())['cases']}

def controls(path):return np.loadtxt(path,delimiter=',',skiprows=1).reshape(-1,4,12)
quality={};geometric={}
for r in rows:
    if r['success']:
        key=f"{r['case']}:{r['mode']}"
        geometric[key]=bending(Path(r['output'])/'core.csv')
        quality[key]=geometric[key]['longitudinal']
(out/'intrinsic_bending.json').write_text(json.dumps(quality,indent=2)+'\n')
(out/'geometric_metrics.json').write_text(json.dumps(geometric,indent=2)+'\n')

def gain(r,key='union'):return 100*r[key]['paired_delta']/r[key]['baseline']
aggregate={}
with (out/'comparison.csv').open('w') as file:
    writer=csv.writer(file);writer.writerow(['split','case','family','success','baseline_ms','candidate_ms','time_delta_percent','baseline_union_m3','candidate_union_m3','union_delta_percent','paired_ci95_m3','core_delta_percent','baseline_bending','candidate_bending','baseline_intrinsic_bending','candidate_intrinsic_bending','c2_error','max_rss_kib'])
    for name,data in datasets.items():
        good=[r for r in data if r['mode']=='free_boundary' and r['success'] and base[r['case']]['success']]
        changes={
            'time':[100*(r['median_ms']/base[r['case']]['median_ms']-1) for r in good],
            'volume':[gain(r) for r in good],
            'core_volume':[gain(r,'core_union') for r in good],
            'bending':[100*(r['boundary_bending']/base[r['case']]['boundary_bending']-1) for r in good if base[r['case']]['boundary_bending']>1e-6],
            'intrinsic_bending':[100*(quality[f"{r['case']}:free_boundary"]/quality[f"{r['case']}:baseline"]-1) for r in good if quality[f"{r['case']}:baseline"]>1e-6],
            'surface_bending':[100*(geometric[f"{r['case']}:free_boundary"]['surface']/geometric[f"{r['case']}:baseline"]['surface']-1) for r in good]}
        aggregate[name]={'success':len(good),'count':len(data)//2,'positive_volume':sum(gain(r)>0 for r in good),
                         **{k:{'min':min(v),'median':statistics.median(v),'max':max(v)} for k,v in changes.items()}}
        for r in data:
            if r['mode']!='free_boundary':continue
            b=base[r['case']];key=str(r['case'])
            writer.writerow([name,r['case'],r['family'],r['success'],b['median_ms'],r['median_ms'],
                100*(r['median_ms']/b['median_ms']-1),r.get('union',{}).get('baseline',''),r.get('union',{}).get('candidate',''),
                gain(r) if 'union' in r else '',r.get('union',{}).get('paired_ci95',''),gain(r,'core_union') if 'core_union' in r else '',
                b.get('boundary_bending',''),r.get('boundary_bending',''),quality.get(key+':baseline',''),quality.get(key+':free_boundary',''),r.get('c2_error',''),r['max_rss_kib']])
(out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n');print(json.dumps(aggregate,indent=2),flush=True)

font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})

def hull(points):
    points=sorted(set(map(tuple,np.round(points,9))))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return np.array(lower[:-1]+upper[:-1])

def firi_outlines(path):
    data=np.loadtxt(path,delimiter=',',skiprows=1);lines=[]
    for i in np.unique(data[:,0]):
        h=data[data[:,0]==i,1:];vertices=[]
        for ids in itertools.combinations(range(len(h)),3):
            selected=h[list(ids)];mat=selected[:,:3]
            if abs(np.linalg.det(mat))<1e-9:continue
            point=np.linalg.solve(mat,-selected[:,3])
            if np.all(h[:,:3]@point+h[:,3]<=1e-7):vertices.append(point[:2])
        poly=hull(vertices)
        if len(poly):lines.append(np.vstack([poly,poly[0]]))
    return lines

def surface(path):
    cp=controls(path);t=np.linspace(0,1,9);w=np.array([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3]).T
    y=np.concatenate([w@p[:,3:] for p in cp]);centers=y[:,:3];m=y[:,3:].reshape(-1,2,3).transpose(0,2,1)
    angle=np.linspace(0,2*np.pi,33);circle=np.array([np.cos(angle),np.sin(angle)])
    rings=centers[:,:,None]+m@circle
    quads=np.stack([rings[:-1,:2,:-1].transpose(0,2,1),rings[1:,:2,:-1].transpose(0,2,1),rings[1:,:2,1:].transpose(0,2,1),rings[:-1,:2,1:].transpose(0,2,1)],axis=2).reshape(-1,4,2)
    return quads,centers

for case,name in [(67,'same_input_corridors'),(60,'maze_regression')]:
    info=inputs[case];seed=np.loadtxt(info['seed']);cloud=np.loadtxt(info['map']);lo=seed[:,:2].min(axis=0)-4;hi=seed[:,:2].max(axis=0)+4
    wire=firi_outlines(Path(a.firi)) if case==67 and Path(a.firi).exists() else []
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,r,title,color,line in zip(axes,[base[case],candidate[case]],['原 C3：端点固定','新 C3：统一自由边界'],['#c2d7e6','#efb7ba'],['#366983','#a82f42']):
        quads,centers=surface(Path(r['output'])/'corridor.csv')
        ax.scatter(cloud[::16,0],cloud[::16,1],s=2,color='#d7dee6',linewidths=0,rasterized=True)
        ax.add_collection(PolyCollection(quads,facecolors=color,edgecolors='none',rasterized=True))
        if wire:ax.add_collection(LineCollection(wire,colors='#3463a0',linewidths=.65,alpha=.9))
        ax.plot(centers[:,0],centers[:,1],color=line,lw=1)
        ax.plot(seed[:,0],seed[:,1],color='#a77d35',lw=.7,ls='--')
        ax.scatter(seed[[0,-1],0],seed[[0,-1],1],color='#182f43',s=27,zorder=5)
        for point,label in zip(seed[[0,-1]],['S','G']):ax.annotate(label,point[:2],xytext=(5,4),textcoords='offset points')
        volume=candidate[case]['union']['candidate' if r['mode']=='free_boundary' else 'baseline']
        ax.set_title(f"{title}\n{r['median_ms']:.2f} ms  ·  {volume:.2f} m³")
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x / m',ylabel='y / m');ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('同输入走廊俯视投影'+('：蓝线为 FIRI' if wire else '：此前出现体积回退的连续迷宫转弯'),fontsize=14)
    fig.text(.5,.02,'S / G 为原始点击位置；虚线为种子路径。图形是俯视投影，体积由独立三维并集采样计算。',ha='center')
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/(name+'.png'),dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.5))
good=[candidate[i] for i in sorted(candidate) if candidate[i]['success']]
ids=[r['case'] for r in good];vol=[gain(r) for r in good];times=[100*(r['median_ms']/base[r['case']]['median_ms']-1) for r in good]
axes[0].bar(ids,vol,color=['#b74756' if v<0 else '#28816f' for v in vol],width=.8)
axes[1].scatter(times,vol,c=['#b74756' if v<0 else '#28816f' for v in vol],s=22)
axes[0].set(xlabel='冻结案例编号',ylabel='并集体积变化 / %',title='全部 66 条共同成功路径')
axes[1].set(xlabel='生成耗时变化 / %',ylabel='并集体积变化 / %',title='速度与体积的取舍')
for ax in axes:ax.axhline(0,color='#687789',lw=.8);ax.grid(alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(out/'paired_metrics.png',dpi=160);plt.close(fig)
