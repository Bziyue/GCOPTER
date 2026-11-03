#!/usr/bin/env python3
"""Collect frozen-input evidence and plot the shape-spline comparison."""
from pathlib import Path
import hashlib,json,statistics,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parents[3];S=R/'.scratch/shape_spline';OUT=R/'docs/validation/shape_spline';OUT.mkdir(parents=True,exist_ok=True);FIG=R/'docs/figures';FIG.mkdir(exist_ok=True)
modes=['baseline','hermite-free','cubic-c2'];labels=['Perlin-A1','Perlin-B1','Maze-1','Perlin-A2','Perlin-B2','Maze-2'];colors=['#8b99a9','#8b55b5','#237da6']
gen=json.loads((S/'generation/results.json').read_text());audits=json.loads((S/'audit/results.json').read_text());summary={'generation':gen,'trajectory':{},'global_membership_audit':audits,'cases':[]}
summary['final_geometry_replay']=json.loads((S/'geometry_replay_equivalence.json').read_text())
assert all(x['identical'] for x in summary['final_geometry_replay'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for mode in modes:
 d=S/'trajectory'/mode;manifest=json.loads((d/'manifest.json').read_text());cases=[]
 for c in manifest['cases']:
  i=c['case'];p=d/f'case_{i:02}';rows=[json.loads(x) for x in (p/'runs.jsonl').read_text().splitlines()];record={**c,'runs':rows}
  if c['returncode']!=0:
   assert mode=='cubic-c2' and i==1
   log=(p/'stderr.log').read_text();record['gradient_failure_log']=log;record['failed_gradient_relative_error']=float(re.search(r'gradient relative error ([\d.e+-]+)',log).group(1));p=S/'diagnostic_cubic_case01/case_00';record['diagnostic_runs']=[json.loads(x) for x in (p/'runs.jsonl').read_text().splitlines()];record['diagnostic_manifest']=json.loads((S/'diagnostic_cubic_case01/manifest.json').read_text());assert record['diagnostic_manifest']['cases'][0]['request']==c['request']
  record['trajectory_sha256']=sha(p/'hermite_trajectory.csv');record['initial_csv_sha256']=sha(p/'initial_waypoints.csv');cases.append(record)
 summary['trajectory'][mode]={'manifest':manifest,'cases':cases}
for i,c in enumerate(gen['cases']):
 rec={'case':i,'name':labels[i],'volume':{v['mode']:v for v in c['volume']},'generation':{},'trajectory':{}}
 assert c['volume'][0]['baseline_volume_m3']==c['volume'][1]['baseline_volume_m3']
 hashes=set();csv_hashes=set()
 for mode in modes:
  rows=[x for x in c['generation'] if x['mode']==mode and x['repeat']>=0];rec['generation'][mode]={**rows[0],'median_generation_ms':statistics.median(x['generation_ms'] for x in rows)}
  t=summary['trajectory'][mode]['cases'][i];rows=t.get('diagnostic_runs',t['runs']);timed=[x for x in rows if x['method']=='hermite' and x['repeat']>=0];assert len(timed)==3;rec['trajectory'][mode]={**timed[0],'median_optimize_ms':statistics.median(x['optimize_ms'] for x in timed),'gradient_validated':t['returncode']==0};hashes.update(x['initial_hash'] for x in rows);csv_hashes.add(t['initial_csv_sha256']);assert t['map_sha256']==c['map_sha256'] and t['seed_sha256']==c['seed_sha256']
 assert len(hashes)==len(csv_hashes)==1
 rec['initial_hash']=hashes.pop();rec['initial_csv_sha256']=csv_hashes.pop();summary['cases'].append(rec)
summary['source_sha256']={str(p.relative_to(R)):sha(p) for p in [*sorted((R/'direct_corridor/experiments/shape_spline').glob('*.cpp')),*sorted((R/'direct_corridor/experiments/shape_spline').glob('*.hpp')),R/'direct_corridor/src/hermite_optimizer.hpp',R/'direct_corridor/include/gcopter/direct_corridor.hpp',R/'gcopter/experiments/trajectory/benchmark.cpp']}
(OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copyfile(S/'baseline_equivalence.json',OUT/'baseline_equivalence.json')
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
x=np.arange(6);fig,axs=plt.subplots(1,3,figsize=(15,4.8));titles=['走廊并集体积变化','完整走廊生成耗时','同初值轨迹优化耗时']
for k,mode in enumerate(modes):
 off=(k-1)*.25
 if mode!='baseline':
  delta=[100*r['volume'][mode]['paired_delta_m3']/r['volume'][mode]['baseline_volume_m3'] for r in summary['cases']];err=[100*r['volume'][mode]['paired_ci95_m3']/r['volume'][mode]['baseline_volume_m3'] for r in summary['cases']];axs[0].bar(x+(k-1.5)*.32,delta,.32,color=colors[k],yerr=err,capsize=2,label=['基线','五次释放导数','C² 三次'][k])
 for ax,key,field in [(axs[1],'generation','median_generation_ms'),(axs[2],'trajectory','median_optimize_ms')]:ax.bar(x+off,[r[key][mode][field] for r in summary['cases']],.25,color=colors[k],label=['基线','五次释放导数','C² 三次'][k])
for ax,title in zip(axs,titles):ax.set_title(title);ax.set_xticks(x,labels,rotation=30,ha='right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
axs[0].axhline(0,c='#555',lw=.8);axs[0].set_ylabel('相对基线 / %（配对 95% 区间）');axs[1].set_ylabel('中位数 / ms');axs[2].set_ylabel('中位数 / ms');axs[1].legend(fontsize=9,frameon=False)
fig.suptitle('截面形状参数化：五次增大体积，三次更平滑；轨迹收益尚未形成',fontsize=14);fig.text(.5,.01,'C² 三次的 Perlin-B1 为初始梯度检查失败后的诊断补跑；较短计算时间也可能来自提前停止。',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(FIG/'shape_spline_comparison.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(13,8.2))
for col,i in enumerate([0,5]):
 for mode,color in zip(modes,colors):
  a=np.loadtxt(S/'generation'/f'case_{i:02}'/f'{mode}_sections.csv',delimiter=',',skiprows=1);sv=np.linalg.svd(np.stack([a[:,4:7],a[:,7:10]],axis=2),compute_uv=False);axs[0,col].plot(a[:,0],np.pi*sv[:,0]*sv[:,1],c=color,lw=1.6,label=['基线','五次释放导数','C² 三次'][modes.index(mode)])
  t=np.loadtxt(S/'trajectory'/mode/f'case_{i:02}'/'hermite_trajectory.csv',delimiter=',',skiprows=1);axs[1,col].plot(t[:,1],t[:,2],c=color,lw=1.6)
 axs[0,col].set(title=labels[i]+' · 截面面积',xlabel='走廊进度 u',ylabel='面积 / m²');axs[0,col].legend(frameon=False,fontsize=9)
 axs[1,col].set(title=labels[i]+' · 最终轨迹俯视',xlabel='x / m',ylabel='y / m');axs[1,col].set_aspect('equal',adjustable='datalim')
 for ax in axs[:,col]:ax.grid(alpha=.15)
fig.suptitle('同一中心线与节点：形状变化对面积和轨迹的影响',fontsize=14);fig.tight_layout();fig.savefig(FIG/'shape_spline_geometry.png',dpi=170);plt.close(fig)
# Tables for the technical report.
rows=['| 路径 | 基线体积 m³ | 五次体积 m³（变化） | 三次体积 m³（变化） | 生成耗时 ms：基线 / 五次 / 三次 |','|---|---:|---:|---:|---:|']
for r in summary['cases']:
 h=r['volume']['hermite-free'];c=r['volume']['cubic-c2'];tim=' / '.join(f"{r['generation'][m]['median_generation_ms']:.1f}" for m in modes);rows.append(f"| {r['name']} | {h['baseline_volume_m3']:.2f} | {h['volume_m3']:.2f}（{100*h['paired_delta_m3']/h['baseline_volume_m3']:+.2f}%） | {c['volume_m3']:.2f}（{100*c['paired_delta_m3']/c['baseline_volume_m3']:+.2f}%） | {tim} |")
rows+=['','| 路径 | 优化耗时 ms | 飞行时长 s | jerk 能量 | 选根位置残差 mm | 采样通过 |','|---|---:|---:|---:|---:|---|']
for r in summary['cases']:
 vals=[]
 for key,scale,digits in [('median_optimize_ms',1,1),('duration_s',1,2),('jerk_energy',1,2),('max_corridor_error_m',1000,2)]:vals.append(' / '.join(f"{r['trajectory'][m][key]*scale:.{digits}f}" for m in modes))
 passed=' / '.join('是' if r['trajectory'][m]['sampled_feasible'] else '否' for m in modes);rows.append('| '+r['name']+(' *' if r['case']==1 else '')+' | '+' | '.join(vals+[passed])+' |')
(S/'tables.md').write_text('\n'.join(rows)+'\n')
for mode in modes[1:]:
 ratios=[r['generation'][mode]['shape_acceleration_energy']/r['generation'][mode]['baseline_shape_acceleration_energy'] for r in summary['cases']];print(mode,'energy ratio',min(ratios),max(ratios),'generation ms',min(r['generation'][mode]['median_generation_ms'] for r in summary['cases']),max(r['generation'][mode]['median_generation_ms'] for r in summary['cases']))
print((S/'tables.md').read_text())
print('max RSS KiB',max(c['peak_rss_kib'] for c in gen['cases']),max(c['peak_rss_kib'] for m in modes for c in summary['trajectory'][m]['manifest']['cases']))
