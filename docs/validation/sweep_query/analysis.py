from pathlib import Path
import json,statistics as st,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('docs/validation/sweep_query')
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
generation=[];query=[];geometry=[]
for folder in sorted((root/'generation').glob('[0-9]*'),key=lambda p:int(p.name)):
 rs=rows(folder/'rows.jsonl');case=int(folder.name);item={'case':case}
 for name in ['C3-OPT-native','H5-S']:
  r=[x for x in rs if x.get('model')==name and x.get('repeat',-1)>=0]
  item[name]={'status':r[0]['status'],'median_wall_ms':st.median(x['wall_ms'] for x in r),'wall_ms':[x['wall_ms'] for x in r]}
  d=3 if name=='C3-OPT-native' else 5
  old=Path(f'.scratch/sweep_corridor/final_regression/d{d}/after/{case}/{name}.csv')
  new=folder/(name+'.csv')
  if old.exists():
   geometry.append(dict(case=case,degree=d,reference=str(old),sha256=digest(new),byte_equal=new.read_bytes()==old.read_bytes()))
 for d in [3,5]:
  for mode in ['full_hessian','first_gradient','inverse_unique','inverse_repeated']:
   q=[x for x in rs if x.get('degree')==d and x.get('mode')==mode and x['repeat']>=0]
   if q:query.append(dict(case=case,degree=d,mode=mode,median_ns=st.median(x['ms']*1e6/x['count'] for x in q),relative_gradient_error=max(x['relative_gradient_error'] for x in q)))
 generation.append(item)
trajectory=[];profile=[]
for stage,destination in [('trajectory',trajectory),('profile',profile)]:
 for folder in sorted((root/stage).glob('[0-9]*'),key=lambda p:int(p.name)):
  rs=rows(folder/'runs.jsonl');item={'case':int(folder.name)}
  assert len(set(r['initial_hash'] for r in rs))==1
  item['initial_hash']=rs[0]['initial_hash']
  for name in ['C3','H5']:
   r=[x for x in rs if x['method']==name and x['repeat']>=0]
   warm=[x for x in rs if x['method']==name and x['repeat']==-1][0]
   keys=['optimize_ms','objective','jerk_energy','duration_s','length_m','max_corridor_error_m','max_penetration_m','max_plane_error_m','inverse_failures','iterations','evaluations','objective_eval_ms','tube_query_ms','integral_callback_ms','dynamics_callback_ms','coefficient_callback_ms']
   item[name]={k:st.median(x[k] for x in r) for k in keys}
   item[name].update(status=r[0]['status'],feasible=all(x['sampled_feasible'] for x in r),gradient_error=warm['gradient_relative_error'],optimize_samples_ms=[x['optimize_ms'] for x in r])
  destination.append(item)
old=json.loads(Path('docs/validation/native_splines_round5/results.json').read_text())['corridor_volume'];volumes=[]
for i in range(7):
 checks=[]
 for m in ['C3-OPT-native','H5-S']:
  a=Path(f'.scratch/native_splines/round5/corridor_final/pruned_predict3_volume1/case_{i:02d}/{m}.csv');b=root/'generation'/str(i)/(m+'.csv')
  checks.append(a.read_bytes()==b.read_bytes())
  assert checks[-1]
 volumes.append(dict(case=i,C3=old[i]['C3-selected'],H5=old[i]['H5-S'],exact_geometry_match=all(checks)))
paired=[x for x in generation if all(x[m]['status']=='SUCCESS' for m in ['C3-OPT-native','H5-S'])]
meds=lambda m:sum(x[m]['median_wall_ms'] for x in paired)
peak=max(int(x) for p in root.glob('*/*/process.log') for x in re.findall(r'Maximum resident set size \(kbytes\): (\d+)',p.read_text()))
result=dict(generation=generation,query=query,trajectory=trajectory,profile=profile,geometry=geometry,volume=volumes,summary=dict(paired_success=len(paired),C3_faster=sum(x['C3-OPT-native']['median_wall_ms']<x['H5-S']['median_wall_ms'] for x in paired),C3_sum_ms=meds('C3-OPT-native'),H5_sum_ms=meds('H5-S'),peak_rss_kib=peak))
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['summary'])
for label,values in [('development',[x for x in trajectory if x['case']<7]),('all',trajectory)]:
 print(label,'sum_ms',*[sum(x[m]['optimize_ms'] for x in values) for m in ['C3','H5']], 'feasible',*[sum(x[m]['feasible'] for x in values) for m in ['C3','H5']])
 print('ranges_speed',min(1-x['C3']['optimize_ms']/x['H5']['optimize_ms'] for x in values),max(1-x['C3']['optimize_ms']/x['H5']['optimize_ms'] for x in values))
print('query_ns',[(d,m,st.median(x['median_ns'] for x in query if x['degree']==d and x['mode']==m)) for d in [3,5] for m in ['full_hessian','first_gradient','inverse_unique','inverse_repeated']])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
ratio=[x['C3-OPT-native']['median_wall_ms']/x['H5-S']['median_wall_ms'] for x in paired]
ax[0].bar([x['case'] for x in paired],ratio,color=['#d6734e' if x>1 else '#337c9b' for x in ratio]);ax[0].axhline(1,color='#555',ls='--');ax[0].set(title='Generation: C3 / H5',xlabel='Frozen case ID',ylabel='Time ratio (lower is faster)')
dev=[x for x in trajectory if x['case']<7]
for j,m in enumerate(['C3','H5']):ax[1].bar([x['case']+(j-.5)*.35 for x in dev],[x[m]['optimize_ms'] for x in dev],width=.35,label=m,color=['#337c9b','#d6734e'][j])
ax[1].set(title='Common trajectory backend',xlabel='Development case ID',ylabel='Optimization time (ms)');ax[1].legend()
ax[2].bar(range(7),[100*(x['C3']['volume']/x['H5']['volume']-1) for x in volumes],color='#337c9b');ax[2].set(title='Union volume: C3 vs H5',xlabel='Development case ID',ylabel='Increase (%)')
fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for a,modes,title in [(ax[0],['full_hessian','first_gradient'],'Fixed-u differential'),(ax[1],['inverse_unique','inverse_repeated'],'Implicit inverse + gradient')]:
 for j,d in enumerate([3,5]):a.bar([i+(j-.5)*.35 for i in range(2)],[st.median(x['median_ns'] for x in query if x['degree']==d and x['mode']==m) for m in modes],width=.35,label=f'Degree {d}',color=['#337c9b','#d6734e'][j])
 a.set_xticks(range(2),[m.replace('_',' ') for m in modes]);a.set(title=title,ylabel='Nanoseconds / query');a.legend()
fig.savefig(root/'queries.png',dpi=160);plt.close(fig)
