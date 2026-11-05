#!/usr/bin/env python3
"""Independent curvature integration, paired tables, plots and raw-run archive."""
import argparse,csv,gzip,hashlib,json,os,resource,statistics as st,subprocess,sys,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.scratch/h3_adaptive');a=p.parse_args()
repo=Path(__file__).resolve().parents[3];os.chdir(repo)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(repo/'SweepCorridor/experiments/free_boundary'))
from geometry_metrics import bending
root=Path(a.root);out=Path('docs/validation/h3_adaptive');out.mkdir(parents=True,exist_ok=True)
sets={name:dict(summary=json.loads((root/name/'summary.json').read_text()),manifest=json.loads((root/name/'manifest.json').read_text()))
      for name in ('final_development','final_regression','final_ablation','final_profile')}
# The screenshot case is close to C3. Increase this preselected detailed case
# to 8 x 262144 common samples rather than ranking a sub-percent difference.
detail={r['mode']:r for r in sets['final_development']['summary'] if r['case']==68}
seed=next(r['seed'] for r in sets['final_development']['manifest']['cases'] if r['case']==68)
for mode in ('h3_joint_baseline','h3_new'):
    r=detail[mode]
    for key,base,file in [('union','c3','corridor.csv'),('core_union','c3','core.csv')]+([('union_vs_joint','h3_joint_baseline','corridor.csv')] if mode=='h3_new' else []):
        if r[key]['samples']>=8*262144:continue
        value=json.loads(subprocess.check_output([str(repo/'.scratch/h3/measure'),str(Path(detail[base]['output'])/file),str(Path(r['output'])/file),seed,'262144'],text=True,timeout=180))
        assert value['unresolved']==0;r[key]=value
sets['final_development']['manifest']['refined_volume_samples_per_shift']={'68':262144}
for name in ('summary','manifest'):(root/'final_development'/f'{name}.json').write_text(json.dumps(sets['final_development'][name],indent=2)+'\n')
rows=sets['final_development']['summary']+sets['final_regression']['summary']
lookup={(r['case'],r['mode']):r for r in rows};modes=('c3','h3_joint_baseline','h3_new')
cache_path=root/'bending.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
for i,r in enumerate(rows):
    if not r['success']:continue
    file=Path(r['output'])/'core.csv';key=hashlib.sha256(file.read_bytes()).hexdigest()
    if key not in cache:
        data=bending(file,48,96);coarse=bending(file,24,48)
        data['order']=48;data['change_from_24']=abs(data['surface']/coarse['surface']-1)
        if data['change_from_24']>.01:
            fine=bending(file,96,192);fine['order']=96;fine['change_from_48']=abs(fine['surface']/data['surface']-1)
            fine['change_from_24']=data['change_from_24'];data=fine
        cache[key]=data;cache_path.write_text(json.dumps(cache,indent=2)+'\n')
    r['bending']=cache[key]
    if i%15==0:print('Curvature',i,'/',len(rows),flush=True)
def stats(values):return dict(minimum=min(values),median=st.median(values),maximum=max(values))
def aggregate(mode):
    items=[r for r in rows if r['mode']==mode];ok=[r for r in items if r['success']]
    result=dict(cases=len(items),success=len(ok),failed=[r['case'] for r in items if not r['success']],
        max_rss_mib=max(r['max_rss_kib'] for r in items)/1024,max_c1_error=max(r['c1_error'] for r in ok),
        terminations=sorted({run['termination'] for r in items for run in r['runs']}))
    for base in ('c3','h3_joint_baseline'):
        if base==mode:continue
        pairs=[(r,lookup[r['case'],base]) for r in ok if lookup[r['case'],base]['success']]
        d=dict(time_ratio=stats([r['median_ms']/b['median_ms'] for r,b in pairs]),
            all_input_time_ratio=stats([r['median_ms']/lookup[r['case'],base]['median_ms'] for r in items]),
            surface_ratio=stats([r['bending']['surface']/b['bending']['surface'] for r,b in pairs]),
            faster=sum(r['median_ms']<b['median_ms'] for r,b in pairs),
            smoother=sum(r['bending']['surface']<b['bending']['surface'] for r,b in pairs))
        key='union' if base=='c3' else 'union_vs_joint'
        if all(key in r for r,b in pairs):
            d['union_ratio']=stats([r[key]['candidate']/r[key]['baseline'] for r,b in pairs])
            d['larger']=sum(r[key]['paired_delta']>0 for r,b in pairs)
            d['larger_beyond_qmc_interval']=sum(r[key]['paired_delta']>r[key]['paired_ci95'] for r,b in pairs)
        if base=='c3' and all('core_union' in r for r,b in pairs):
            d['core_union_ratio']=stats([r['core_union']['candidate']/r['core_union']['baseline'] for r,b in pairs])
        result['vs_'+base]=d
    return result
aggregate_data={m:aggregate(m) for m in modes}
(out/'aggregate.json').write_text(json.dumps(aggregate_data,indent=2)+'\n')
with (out/'paired.csv').open('w') as f:
    writer=csv.writer(f,lineterminator='\n');writer.writerow(['case','family','mode','success','median_ms','p95_ms','union_m3','union_ratio_c3','gain_joint_pct','gain_ci95_pct','surface_energy','surface_ratio_joint','c1_error'])
    for r in rows:
        values=[r['case'],r['family'],r['mode'],r['success'],r['median_ms'],r['p95_ms']]
        if r['success']:
            b=lookup[r['case'],'h3_joint_baseline'];v=r.get('union');old=r.get('union_vs_joint')
            volume=v['candidate'] if v else lookup[r['case'],'h3_new']['union']['baseline']
            values += [volume,v['candidate']/v['baseline'] if v else 1,
                100*old['paired_delta']/old['baseline'] if old else '',100*old['paired_ci95']/old['baseline'] if old else '',
                r['bending']['surface'],r['bending']['surface']/b['bending']['surface'],r['c1_error']]
        writer.writerow(values)
# The local quadrature cache must change no output coefficients. Rebuilt fd7
# must also reproduce the frozen executable used for the initial screen.
screen=json.loads((root/'screen_share/summary.json').read_text()) if (root/'screen_share/summary.json').exists() else []
checks=[]
for old in screen:
    mode='h3_new' if old['mode']=='h3_new_share99' else old['mode'];r=lookup[old['case'],mode]
    assert old['success']==r['success']
    if old['success']:
        equal=old['geometry_sha256']==r['geometry_sha256'];assert equal,(old['case'],mode)
        checks.append(dict(case=r['case'],mode=mode,byte_identical=equal))
(out/'code_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
colors={'c3':'#4778b8','h3_joint_baseline':'#85939b','h3_new':'#cc3e50'}
labels={'c3':'C3','h3_joint_baseline':'H3 shared jets','h3_new':'H3 local refinement'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17})
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for mode in modes[1:]:
    rr=[r for r in rows if r['mode']==mode and r['success']]
    axes[0].scatter([100*r['median_ms']/lookup[r['case'],'c3']['median_ms'] for r in rr],
        [100*r['union']['candidate']/r['union']['baseline'] for r in rr],s=23,alpha=.7,c=colors[mode],label=labels[mode])
    axes[1].scatter([100*r['bending']['surface']/lookup[r['case'],'c3']['bending']['surface'] for r in rr],
        [100*r['union']['candidate']/r['union']['baseline'] for r in rr],s=23,alpha=.7,c=colors[mode],label=labels[mode])
for ax in axes:
    ax.axvline(100,color='#999',lw=.8,ls='--');ax.axhline(100,color='#999',lw=.8,ls='--')
    ax.set_ylabel('Union volume / C3 (%)');ax.legend(loc='lower left')
axes[0].set_xlabel('Generation time / C3 (%)');axes[1].set_xlabel('Core surface bending energy / C3 (%)')
fig.suptitle('69 common successes / 71 frozen inputs · generation only')
fig.savefig(out/'tradeoff.png',dpi=170);plt.close(fig)
def sample(file,per_span=8,angles=40):
    data=np.loadtxt(file,delimiter=',',skiprows=1).reshape(-1,4,12);cp=data[:,:,3:]
    t=np.linspace(0,1,per_span,endpoint=False);r=1-t;b=np.array([r**3,3*r*r*t,3*r*t*t,t**3]).T
    y=np.einsum('kj,pjd->pkd',b,cp).reshape(-1,9);y=np.vstack([y,cp[-1,-1]])
    theta=np.linspace(0,2*np.pi,angles);surface=y[:,:3,None]+y[:,3:6,None]*np.cos(theta)+y[:,6:9,None]*np.sin(theta)
    area=np.pi*np.linalg.norm(np.cross(y[:,3:6],y[:,6:9]),axis=1)
    u=(data[:,0,1,None]+(data[:,0,2]-data[:,0,1])[:,None]*t).ravel();return surface,y[:,:3],np.r_[u,1.],area
inputs={r['case']:r for r in json.loads(Path('docs/validation/h3_volume/inputs.json').read_text())['cases']}
for case in (68,2,37):
    seed=np.loadtxt(inputs[case]['seed']);cloud=np.loadtxt(inputs[case]['map'])
    lo=seed.min(axis=0)-4;hi=seed.max(axis=0)+4
    cloud=cloud[((cloud>=lo)&(cloud<=hi)).all(axis=1)][::2]
    cloud=cloud[::max(1,len(cloud)//4000)]
    fig=plt.figure(figsize=(15,4.7),layout='constrained')
    for k,mode in enumerate(modes):
        r=lookup[case,mode];ax=fig.add_subplot(1,3,k+1,projection='3d');s,c,u,area=sample(Path(r['output'])/'corridor.csv')
        ax.scatter(*cloud.T,s=.4,color='#bac7d2',alpha=.09,rasterized=True)
        ax.plot_surface(s[:,0,:],s[:,1,:],s[:,2,:],rstride=3,cstride=2,linewidth=0,alpha=.65,color=colors[mode],shade=True)
        ax.plot(*seed.T,color='#e2ab32',lw=1.2);ax.plot(*c.T,color=colors[mode],lw=.9)
        ax.scatter(*seed[[0,-1]].T,s=25,c=['#219580','#b83a49'],depthshade=False)
        volume=r['union']['candidate'] if mode!='c3' else lookup[case,'h3_new']['union']['baseline']
        ax.set_title(f"{labels[mode]}\n{r['median_ms']:.2f} ms | {volume:.1f} m³",pad=8)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(0,5))
        ax.view_init(elev=65,azim=-105);ax.set_proj_type('ortho')
        ax.set_box_aspect((hi[0]-lo[0],hi[1]-lo[1],5),zoom=.95);ax.set_axis_off()
    fig.suptitle(f'Case {case} · identical map, seed and safety model · exported coefficients')
    fig.savefig(out/f'case{case}_geometry.png',dpi=170,bbox_inches='tight',pad_inches=.2);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
for mode in modes:
    _,c,u,area=sample(Path(lookup[68,mode]['output'])/'corridor.csv',14)
    axes[0].plot(u,area,c=colors[mode],label=labels[mode],lw=1.4)
    axes[1].plot(c[:,0],c[:,1],c=colors[mode],lw=1.8,label=labels[mode])
seed=np.loadtxt(inputs[68]['seed']);goal=seed[-1]
axes[1].scatter(*goal[:2],c='#e2ab32',s=45,zorder=5,label='Input goal')
axes[1].set(xlim=(goal[0]-3.5,goal[0]+3.5),ylim=(goal[1]-3.5,goal[1]+3.5),xlabel='x / m',ylabel='y / m',title='Case 68 · free end centerline');axes[1].set_aspect('equal')
axes[0].set(xlabel='Seed parameter u',ylabel='Section area / m²',title='Case 68 · complete corridor');axes[0].legend()
fig.savefig(out/'case68_end.png',dpi=170);plt.close(fig)
studies={}
for path in sorted(root.glob('*/summary.json')):
    studies[path.parent.name]=dict(summary=json.loads(path.read_text()))
    for name in ('manifest','quality'):
        if (path.parent/(name+'.json')).exists():studies[path.parent.name][name]=json.loads((path.parent/(name+'.json')).read_text())
studies.update(sets)
payload=dict(studies=studies,tests={path.name:path.read_text() for path in root.glob('*tests.log')},aggregate=aggregate_data)
with gzip.open(out/'results.json.gz','wt',encoding='utf8') as f:json.dump(payload,f,ensure_ascii=False)
with tarfile.open(out/'geometry.tar.gz','w:gz') as tar:
    for r in rows:
        if not r['success'] or r['case'] not in (0,1,2,12,37,57,60,67,68,69,70):continue
        for name in ('corridor.csv','core.csv','support_planes.csv'):
            tar.add(Path(r['output'])/name,arcname=f"case_{r['case']:02d}/{r['mode']}/{name}")
artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='artifacts.json'}
(out/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print(json.dumps(aggregate_data,indent=2),flush=True)
