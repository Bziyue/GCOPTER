#!/usr/bin/env python3
"""Serial offline geometry metrics, figures, and a reproducible result archive."""
import argparse,gzip,hashlib,json,os,resource,statistics as st,sys,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.scratch/h3');args=p.parse_args()
repo=Path(__file__).resolve().parents[3];os.chdir(repo)
os.sched_setaffinity(0,{2});resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(repo/'SweepCorridor/experiments/free_boundary'))
from geometry_metrics import bending
root=Path(args.root);out=Path('docs/validation/h3');out.mkdir(parents=True,exist_ok=True)
inputs=json.loads(Path('docs/validation/free_boundary/inputs.json').read_text());cases={c['case']:c for c in inputs['cases']}
sets={name:dict(summary=json.loads((root/name/'summary.json').read_text()),manifest=json.loads((root/name/'manifest.json').read_text())) for name in ('final_development','final_regression')}
rows=sum([v['summary'] for v in sets.values()],[])
cache_path=root/'bending.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
convergence=[]
for i,r in enumerate(rows):
    if not r['success']:continue
    file=Path(r['output'])/'core.csv';key=hashlib.sha256(file.read_bytes()).hexdigest()
    if key not in cache:
        data=bending(file,48,96);data['order']=48
        if r['case'] in (0,1,2,3,4,5,6,67) or r['case']>=51:
            coarse=bending(file,24,48);change=max(abs(data[k]-coarse[k])/max(abs(data[k]),1e-10) for k in ('surface','longitudinal'))
            data['change_from_24']=change
            if change>.02:
                fine=bending(file,96,192);fine['order']=96;fine['change_from_48']=max(abs(fine[k]-data[k])/max(abs(fine[k]),1e-10) for k in ('surface','longitudinal'));fine['change_from_24']=change;data=fine
        cache[key]=data;cache_path.write_text(json.dumps(cache,indent=2)+'\n')
    r['bending']=cache[key]
    if 'change_from_24' in r['bending']:convergence.append(dict(case=r['case'],mode=r['mode'],**r['bending']))
    if i%15==0:print('Curvature',i,'/',len(rows),flush=True)
base={r['case']:r for r in rows if r['mode']=='c3'}

def aggregate(data,mode):
    items=[r for r in data if r['mode']==mode];ok=[r for r in items if r['success'] and base[r['case']]['success']]
    def stats(values):return dict(median=st.median(values),minimum=min(values),maximum=max(values)) if values else None
    volume=[r['union']['candidate']/r['union']['baseline'] for r in ok if 'union' in r]
    return dict(cases=len(items),success=sum(r['success'] for r in items),failed=[r['case'] for r in items if not r['success']],
        time_ratio=stats([r['median_ms']/base[r['case']]['median_ms'] for r in ok]),
        all_input_time_ratio=stats([r['median_ms']/base[r['case']]['median_ms'] for r in items]),union_ratio=stats(volume),
        surface_ratio=stats([r['bending']['surface']/base[r['case']]['bending']['surface'] for r in ok]),
        longitudinal_ratio=stats([r['bending']['longitudinal']/base[r['case']]['bending']['longitudinal'] for r in ok if base[r['case']]['bending']['longitudinal']>1e-8]),
        max_rss_mib=max(r['max_rss_kib'] for r in items)/1024,
        max_c1_error=max(r['c1_error'] for r in ok),max_c2_jump=max(r['c2_error'] for r in ok),
        terminations=sorted(set(t['termination'] for r in items for t in r['runs'])))
summary={name:{mode:aggregate(data['summary'],mode) for mode in sorted(set(r['mode'] for r in data['summary']))} for name,data in sets.items()}
summary['all']={m:aggregate(rows,m) for m in ('c3','h3_jets','h3_mid')}
(out/'aggregate.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(out/'curvature_resolution.json').write_text(json.dumps(convergence,indent=2)+'\n')

colors={'c3':'#45566a','h3':'#cb8140','h3_jets':'#168677','h3_mid':'#4779bf'}
labels={'c3':'C3 baseline','h3':'H3 fixed jets','h3_jets':'H3 updated jets','h3_mid':'H3 early jets'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
for mode in ('h3_jets','h3_mid'):
    a=[r for r in rows if r['mode']==mode and r['success']]
    ax.scatter([100*r['median_ms']/base[r['case']]['median_ms'] for r in a],[100*r['union']['candidate']/r['union']['baseline'] for r in a],s=30,alpha=.7,c=colors[mode],label=labels[mode])
ax.axvline(100,color='#888',ls='--',lw=1);ax.axhline(100,color='#888',ls='--',lw=1)
ax.set(xlabel='Generation time / C3 (%) — lower is faster',ylabel='Union volume / C3 (%) — higher is larger',title='66 common successes · same frozen inputs');ax.legend(loc='lower left')
fig.savefig(out/'tradeoff.png',dpi=170);plt.close(fig)

# Actual exported Bezier sweeps, sampled for display. Metrics use independent QMC.
def sample(file,per_span=5,angles=36):
    data=np.loadtxt(file,delimiter=',',skiprows=1).reshape(-1,4,12);cp=data[:,:,3:]
    t=np.linspace(0,1,per_span,endpoint=False);r=1-t;b=np.array([r**3,3*r*r*t,3*r*t*t,t**3]).T
    y=np.einsum('kj,pjd->pkd',b,cp).reshape(-1,9);y=np.vstack([y,cp[-1,-1]])
    theta=np.linspace(0,2*np.pi,angles);surface=y[:,:3,None]+y[:,3:6,None]*np.cos(theta)+y[:,6:9,None]*np.sin(theta)
    area=np.pi*np.linalg.norm(np.cross(y[:,3:6],y[:,6:9]),axis=1)
    u=(data[:,0,1,None]+(data[:,0,2]-data[:,0,1])[:,None]*t).ravel();u=np.r_[u,1.]
    return surface,y[:,:3],u,area
case=67;lookup={(r['case'],r['mode']):r for r in rows};seed=np.loadtxt(cases[case]['seed'])
fig=plt.figure(figsize=(14,5),layout='constrained')
for k,mode in enumerate(('c3','h3','h3_jets')):
    ax=fig.add_subplot(1,3,k+1,projection='3d');r=lookup[(case,mode)];s,center,u,area=sample(Path(r['output'])/'corridor.csv')
    ax.plot_surface(s[:,0,:],s[:,1,:],s[:,2,:],rstride=2,cstride=2,linewidth=0,alpha=.55,color=colors[mode],shade=True)
    ax.plot(*seed.T,color='#e6aa45',lw=1.2);ax.plot(*center.T,color=colors[mode],lw=.7)
    ax.set_title(labels[mode]+f"\n{r['median_ms']:.2f} ms",pad=18);ax.view_init(elev=63,azim=-105)
    ax.set_box_aspect((np.ptp(seed[:,0])+5,np.ptp(seed[:,1])+5,6));ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_zlim(0,5);ax.set_zticks([0,5])
    ax.set_xlim(seed[:,0].min()-3,seed[:,0].max()+3);ax.set_ylim(seed[:,1].min()-3,seed[:,1].max()+3)
fig.savefig(out/'case67_geometry.png',dpi=170,bbox_inches='tight',pad_inches=.25);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
for mode in ('c3','h3','h3_jets','h3_mid'):
    r=lookup[(case,mode)];s,c,u,a=sample(Path(r['output'])/'core.csv',12)
    axes[0].plot(u,a,c=colors[mode],label=labels[mode],lw=1.1)
    axes[1].plot(u,a,c=colors[mode],lw=1.1)
axes[0].set(title='Case 67 · cross-section area between the original anchors',ylabel='Area / m²');axes[0].legend(ncol=2)
axes[1].set(xlim=(.25,.47),xlabel='Core parameter u',ylabel='Area / m²',title='Detail of the same curves')
fig.savefig(out/'case67_area.png',dpi=170);plt.close(fig)

ablation_rows=json.loads((root/'assembly_ablation/summary.json').read_text());assembly=[]
for case in (0,2,67):
    a={r['mode']:r for r in ablation_rows if r['case']==case};w,h=a['h3_world'],a['h3_jets']
    x=np.loadtxt(Path(w['output'])/'corridor.csv',skiprows=1,delimiter=',');y=np.loadtxt(Path(h['output'])/'corridor.csv',skiprows=1,delimiter=',')
    assembly.append(dict(case=case,world_ms=w['median_ms'],direct_ms=h['median_ms'],speed_reduction=1-h['median_ms']/w['median_ms'],max_coefficient_delta=float(abs(x-y).max()),union_delta=h['union']['candidate']-w['union']['candidate']))
(root/'assembly_ablation/comparison.json').write_text(json.dumps(assembly,indent=2)+'\n')
assembly_table='\n'.join(f"| {r['case']} | {r['world_ms']:.2f} | {r['direct_ms']:.2f} | {100*r['speed_reduction']:.1f}% | {r['max_coefficient_delta']:.2e} |" for r in assembly)
profiles=json.loads((root/'profile/summary.json').read_text())
lines=[]
for r in profiles:
    if r['case']!=67:continue
    med=lambda key:st.median(t[key] for t in r['runs'])
    lines.append('| '+labels[r['mode']]+' | '+' | '.join(f'{med(k):.2f}' for k in ('cover_ms','setup_ms','assembly_ms','solve_ms','line_ms'))+f" | {med('jet_ms') if r['mode']!='c3' else 0:.2f} |")

m=summary['all']['h3_jets'];n=summary['all']['h3_mid'];dev=sets['final_development']['summary']
with (out/'paired.csv').open('w') as f:
    f.write('case,family,mode,success,time_ms,time_ratio,union_ratio,surface_ratio,c1_error,c2_jump\n')
    for r in rows:
        b=base[r['case']];f.write(f"{r['case']},{r['family']},{r['mode']},{r['success']},{r['median_ms']},{r['median_ms']/b['median_ms']},")
        if r['success']:f.write(f"{r['union']['candidate']/r['union']['baseline'] if 'union' in r else 1},{r['bending']['surface']/b['bending']['surface']},{r['c1_error']},{r['c2_error']}")
        else:f.write(',,,')
        f.write('\n')
table=[]
for case in (0,1,2,3,4,5,6,67):
    b=base[case];a=lookup[(case,'h3_jets')];z=lookup[(case,'h3_mid')]
    table.append(f"| {case} / {b['family']} | {b['median_ms']:.2f} | {a['median_ms']:.2f} | {z['median_ms']:.2f} | {100*(a['union']['candidate']/a['union']['baseline']-1):+.2f}% | {100*(a['bending']['surface']/b['bending']['surface']-1):+.1f}% |")
text=f'''# H3 三次 Hermite 走廊实验

2026-11-05 · 分支 `experiment/h3-local-hermite` · C3 基线 `7d4fc603607f131a1d282247c3e523c11da5d06e`。

本轮得到一个专门利用两节点结构的 C¹ 三次 Hermite 生成器。68 个冻结输入中，C3、H3 默认候选和提前更新候选均有 66 个成功，两个失败输入一致。对 66 个共同成功输入，默认 H3 的配对耗时比中位数为 **{m['time_ratio']['median']:.3f}**，即减少 **{100*(1-m['time_ratio']['median']):.1f}%**；并集体积比中位数为 **{m['union_ratio']['median']:.3f}**，即减少 **{100*(1-m['union_ratio']['median']):.1f}%**。曲面弯曲能量比中位数为 **{m['surface_ratio']['median']:.3f}**。

目前 H3 提供了明确的生成速度收益，但尚未在体积上追平 C3。实验代码保留在独立目录，主库和 RViz2 默认仍使用 C3。

![相同输入的时间与并集体积取舍](validation/h3/tradeoff.png)

图中每个点对应一组相同地图、种子路径和安全设置。失败输入仍保留在原始统计中；图中展示的是共同成功的 66 组。空旷直线、竖直线的速度收益更大，不能用它们代替复杂地图的结果。

## 实现和数学结构

所有方法使用同一个几何模型：

$$F(u,z)=c(u)+M(u)z,\\qquad \\|z\\|_2\\le 1.$$

H3 的节点值写成 $Y_i=[c_i,\\operatorname{{vec}}(M_i)]\\in\\mathbb R^9$。每个节点优化三个世界中心偏移与三个对称形状变量。共享世界一阶导数 $V_i$ 在一轮节点优化中固定：

$$B_0=Y_i,\\quad B_1=Y_i+\\frac{{h_i}}{{3}}V_i,\\quad
B_2=Y_{{i+1}}-\\frac{{h_i}}{{3}}V_{{i+1}},\\quad B_3=Y_{{i+1}}.$$

这不是把 H5 的次数常量改为 3。实现针对下面的结构进行了处理。

| 结构 | 当前原生 C3 | 本次 H3 |
|---|---|---|
| 单段依赖的节点值 | 4 个、24 维 | 2 个、12 维 |
| 普通支撑控制截面 | 多节点组合 | 未细分时只关联一个 6 维节点 |
| Hessian 局部结构 | 块半带宽 3 | 6×6 块三对角 |
| 导数 Bézier 控制量 | 三组可变 | 两组固定，只保留中间一组可变进度锥 |
| 端点反求的局部梯度 | 24 维 | 12 维 |
| 体积积分 | 五点 Gauss，八次精确 | 同样精确，预缓存权重及固定导数偏置 |
| 连续性 | C² | 共享一阶导数保证 C¹ |
| 全截面 Jacobian 证书次数 | 8 | 8 |

从归档 H5 `archive/h5-sweep-corridor@bd6d81f` 借用了节点分组、块三对角 Schur 消元、尺度平衡和共享导数提议的思路。保留当前 C3 的自由端面、原起终点的隐式包含约束、真实扫掠体积目标和全截面 Jacobian 接受条件。

单节点支撑行直接组装六维梯度与 Hessian，避免先组装九维世界 Hessian 再回传；真正经过细分的行仍只关联两个节点。线性系统因子在体积秩一修正和自由端面两个额外右端项之间复用。H3 与 C3 都输出三次 Bézier，单次查询与证书的多项式次数相同，不能把生成器提速归因于查询降阶。

固定分离平面和共享导数时，支撑锥、形状正定域及二次正则的梯度与 Hessian 是解析计算。与基线一致，真实体积的非凸 Hessian 使用正秩一部分，隐式锚点使用 Gauss–Newton 曲率；没有声称整个问题是凸问题。

## 共享导数如何更新

直接固定初始化导数会在截面扩大后产生明显起伏。默认候选在每轮膨胀后，对当前九维节点值求自然三次的导数提议。九个通道共用一套三对角分解，然后逐节点限制更新幅度：

1. 一个节点导数只改变左段的 $B_2$ 和右段的 $B_1$，因此只扫描这两段已有的支持平面。
2. 对原有细分区间求精确的二次锥线搜索可行步长。
3. 重新认证这两段的全截面 Jacobian，并更新原起终点的截面反求。
4. 接受降低这两段弯曲能量的更新。进入下一轮节点优化后，再将导数固定。

导数并非一边随节点变量变化、一边忽略其梯度；它们属于明确的外层更新。初值来自一个未优化的小截面参考的精确基变换，没有隐藏一次完整 C3 走廊生成，全部初始化成本计入 H3。

共享支持冗余证明也做了一个必要扩展：H3 控制截面带有固定形状偏置 $D_{{\\mathrm{{ref}}}}$，残差方向 $r$ 的可达域上界必须增加 $\\|D_{{\\mathrm{{ref}}}}^T r\\|$。测试中构造了忽略该项会错误删除约束的反例。原生 C3 通过编译期分支保持原计算路径。

## 相同输入的结果

开发集八例，每种方法预热一次、正式重复七次。下表的时间为单例中位数；体积和曲面能量变化均为默认更新导数 H3 相对 C3。

| 输入 | C3 / ms | H3 更新导数 / ms | H3 提前更新 / ms | 并集体积变化 | 曲面弯曲能量变化 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

曲面弯曲能量使用侧壁上的 $\\int(k_1^2+k_2^2)\\,dA$，越低代表该指标下边界起伏越少。另行保存边界母线的几何曲率积分、截面面积和 C² 二阶导数跳变量。积分采用几何度量，而不是仅用参数二阶导数判断平滑。

![同一路径下的实际输出曲面](validation/h3/case67_geometry.png)

上图黄色线为同一条种子路径，曲面直接取自导出的 Bézier 系数。中间的固定导数版本虽快，但边界起伏明显；右侧为外层更新导数后的版本。

![截面面积及局部放大](validation/h3/case67_area.png)

面积曲线只比较两个原始锚点之间的区域，横轴为该区域重新归一化后的参数 $u$。总体并集体积另外包含自由端部扩张，二者分别存储。

其余 60 例每种方法预热一次、正式重复三次，包括 Perlin、迷宫、S 弯、变半径通道、螺旋、迷宫急转弯、空旷直线及竖直线。全部 68 例的成功率均为 **66/68**；回折尖点 `65` 和窄口 `66` 三种方法均报告初始化失败。

| 66 个共同成功输入的配对指标 | H3 更新导数 | H3 提前更新 |
|---|---:|---:|
| 耗时比中位数 | {m['time_ratio']['median']:.3f} | {n['time_ratio']['median']:.3f} |
| 耗时比范围 | {m['time_ratio']['minimum']:.3f}–{m['time_ratio']['maximum']:.3f} | {n['time_ratio']['minimum']:.3f}–{n['time_ratio']['maximum']:.3f} |
| 并集体积比中位数 | {m['union_ratio']['median']:.3f} | {n['union_ratio']['median']:.3f} |
| 并集体积比范围 | {m['union_ratio']['minimum']:.3f}–{m['union_ratio']['maximum']:.3f} | {n['union_ratio']['minimum']:.3f}–{n['union_ratio']['maximum']:.3f} |
| 曲面弯曲能量比中位数 | {m['surface_ratio']['median']:.3f} | {n['surface_ratio']['median']:.3f} |
| 最大 C¹ 接缝误差 | {m['max_c1_error']:.2e} | {n['max_c1_error']:.2e} |
| 生成进程峰值 RSS / MiB | {m['max_rss_mib']:.1f} | {n['max_rss_mib']:.1f} |

默认 H3 在 56/66 个输入上的曲面能量低于 C3，其余 10 例略高，最大增幅约 16%。这项改进不是逐例支配。空旷直线的基线母线弯曲能量接近零，因此该指标的比值会被极小分母放大，原始绝对值也一起保存。

也保存包含失败输入的配对耗时统计。数据中多条路径属于同一地图，因此这些汇总是当前测试集的描述统计，没有把它们当作 68 个独立地图来推断泛化置信区间。

## 阶段耗时和消融

以下为输入 67 单独开启阶段计时后的三次中位数。总速度排名使用关闭阶段计时的运行；线搜索列已经包含 Jacobian 认证时间，不能再把证书时间相加。

| 方法 | 覆盖 / ms | 构造 / ms | 组装 / ms | 求解 / ms | 线搜索 / ms | 导数更新 / ms |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

初测另外比较了三种预算方案：把每轮初始 Newton 从 8 步增加到 12 步、三轮较短膨胀、平滑后追加三步再膨胀。增加预算可在部分路径收回少量体积，但耗时增长更明显；三轮较短膨胀还能降低边界能量，却未改善体积。原始结果保存在 `pilot_budget` 与 `pilot_polish`，没有据此替换默认候选。

单节点六维组装还单独与九维回传版本比较，其他算法与预算保持一致。两种二进制可以通过 `--targets benchmark,world` 分别构建。三条路径的 QMC 并集体积完全相同，完整生成耗时再下降约 10%，系数最大差异低于 $6\\times10^{{-10}}$。

| 输入 | 九维回传 / ms | 六维直接组装 / ms | 完整生成耗时减少 | 最大系数差 |
|---|---:|---:|---:|---:|
{assembly_table}

该优化已包含在最终结果中，原始数据位于 `assembly_ablation`。

## 安全、数值验证与复现

所有方法使用 0.25 m 体素的外接闭球、0.20 m 机器人半径、0.05 m 安全距离、0.002 m 证书余量、3.0 m 局部扩张范围、0.40 m 初始间距及相同冻结路径。连续认证标准没有改变。每次正式运行在计时结束后独立复核全部支持平面、全截面 Jacobian、C¹ 接缝及原起终点包含关系。

并集体积使用八组随机平移，每组 65,536 个同序列 QMC 样本，同时给出配对差值误差区间；全部反求未判定计数为零。输出同时记录解析扫掠体积，后者有重叠时会重复计数，不作为并集体积替代。曲率积分采用 48 阶纵向积分与 96 个周向样本；开发集及人工场景的 24/48 阶对比最大相对变化约 0.020%，记录保存在 `curvature_resolution.json`。

数值测试包括非均匀节点的精确重构、C¹ 接缝、自然导数完整接受时的 C² 恢复、单节点与细分支撑行的梯度/Hessian、解析体积梯度、体积线搜索多项式、隐式反求梯度、块求解的稠密参考、自由端面 KKT、近墙端部及固定偏置冗余界反例。有限差分误差约为 $10^{{-10}}$。详细测试输出与原始终止状态在结果归档中。主库测试通过，当前 C3 在开发集八例的完整几何、核心区间几何及支持平面与冻结提交逐字节一致。

C¹ 并不妨碍节点空间中的解析 Hessian。关于空间查询，接缝处的一阶梯度连续，涉及 $u$ 的二阶几何导数通常有跳变；没有把这种走廊当作 C² 发布。本轮没有运行轨迹优化，也不推断其后端耗时或质量。

全部构建与数值实验串行执行，固定 CPU 2、数值库线程数为 1、地址空间上限 4 GiB。冷热计时分别保留：表格比较复用 BVH 的生成热路径，原始行记录一次 BVH 构建时间。H3 默认仍使用 28 个节点 Newton 步的预算，多数结果的终止状态为 `iteration_limit`；成功状态表示得到已认证走廊。

复现命令见 [实验目录 README](../SweepCorridor/experiments/h3/README.md)。本目录的 `paired.csv`、`aggregate.json`、`results.json.gz`、`geometry.tar.gz` 和 `artifacts.json` 分别保存逐例配对数据、汇总、原始运行及测试、代表性几何、源码与文件哈希。

## 后续方向

下一步优先解决导数优化与节点膨胀脱节带来的体积损失。当前自然导数目标只考虑弯曲能量，并不主动增加体积；安全裁剪后再固定导数，也限制了体积优化的方向。可以试验中心导数、形状导数各一个局部幅度变量，在保持两节点结构的前提下，把体积梯度直接传播到这两个自由度，再比较增加块尺寸后的实际成本。

其次应检查体积差较大的 Perlin 路径：区分自由端部扩张不足、当前分离面方向与 Bézier 支撑凸包的保守性。用局部认证缺口选择细分区间，而不是统一增加节点或统一增加迭代。只有在这些诊断证明有收益后，才考虑把 H3 接入默认演示。
'''
Path('docs/H3三次Hermite走廊实验.md').write_text(text)
# Archive raw observations, including unsuccessful and unattractive variants.
experiments=dict(sets)
for name in ('pilot_fixed','pilot_jets','pilot_mid','pilot_budget','pilot_polish','kernel','assembly_ablation','development','profile'):
    directory=root/name
    if (directory/'summary.json').exists():experiments[name]={k:json.loads((directory/(k+'.json')).read_text()) for k in ('summary','manifest')}
experiments['assembly_ablation']['comparison']=assembly
logs={p.name:p.read_text() for p in root.glob('*.log') if p.name in ('tests.log','library_test.log','c3_unchanged.log')}
with gzip.open(out/'results.json.gz','wt',encoding='utf-8') as f:json.dump(dict(experiments=experiments,validation=logs,aggregate=summary),f,ensure_ascii=False)
(out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
with tarfile.open(out/'geometry.tar.gz','w:gz') as archive:
    for case in (0,1,2,4,6,51,57,60,63,64,67):
        for mode in ('c3','h3','h3_jets','h3_mid'):
            r=lookup.get((case,mode))
            if not r or not r['success']:continue
            for name in ('corridor.csv','core.csv','support_planes.csv','rows.jsonl'):archive.add(Path(r['output'])/name,arcname=f'case_{case:02d}/{mode}/{name}')
        archive.add(cases[case]['seed'],arcname=f'case_{case:02d}/seed.xyz')
artifacts=dict(baseline='7d4fc603607f131a1d282247c3e523c11da5d06e',
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('SweepCorridor/include','SweepCorridor/experiments/h3') for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.hpp','.cpp','.py','.md')},
    binaries={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'benchmark',root/'baseline_benchmark',root/'tests',root/'measure',root/'world')},
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='artifacts.json'})
(out/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print(json.dumps(summary['all'],indent=2),flush=True)
