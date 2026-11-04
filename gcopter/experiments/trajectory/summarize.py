#!/usr/bin/env python3
"""Store compact evidence and generate the comparison report (no map copies)."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('runs',type=Path)
p.add_argument('evidence',type=Path)
p.add_argument('report',type=Path)
a=p.parse_args()
a.evidence.mkdir(parents=True,exist_ok=True)
names=['Perlin-A1','Perlin-B1','Maze-1','Perlin-A2','Perlin-B2','Maze-2']
profiles={}
for profile in ['final_fast','final_dense']:
    root=a.runs/profile
    manifest=json.loads((root/'manifest.json').read_text())
    cases=[]
    for item in manifest['cases']:
        path=root/f"case_{item['case']:02d}"
        records=[json.loads(line) for line in (path/'runs.jsonl').read_text().splitlines()]
        if item['returncode']!=0 or len(records)!=2*(manifest['parameters']['repeats']+1):
            raise ValueError(f'incomplete run: {path}')
        assert len({r['initial_hash'] for r in records})==1
        cases.append(dict(**item,runs=records))
    profiles[profile]=dict(parameters=manifest['parameters'],cases=cases)
    (a.evidence/(profile+'.json')).write_text(json.dumps(profiles[profile],indent=2)+'\n')

def pair(profile,case):
    rows=profiles[profile]['cases'][case]['runs']
    result={}
    for method in ['firi','C3']:
        records=[r for r in rows if r['repeat']>=0 and r['method']==method]
        r=records[0].copy()
        r['median_ms']=statistics.median(x['optimize_ms'] for x in records)
        r['min_ms']=min(x['optimize_ms'] for x in records)
        r['max_ms']=max(x['optimize_ms'] for x in records)
        r['cost_per_evaluation_us']=1000*r['median_ms']/(r['evaluations']-1)
        result[method]=r
    return result

lines=['# 相同初值的 FIRI / 可微走廊轨迹优化对照','',
'实验使用六组已有 RViz 冻结地图与种子路径，来自同一份 Perlin 地图的四条路径与一份 Maze 地图的两条路径。历史 Random 标签按地图 SHA256 更正为 Perlin-A，批量 Perlin 请求记为 Perlin-B。两边从同一份五次最小 jerk 样条及时间分配开始，用同一后端优化。当前可微走廊指原生 C3 椭圆扫掠。','',
'**这是一套新接入的公共后端对照，不是原生 GCOPTER/FIRI 完整管线的性能排名。** 原生后端会根据多面体重建初值；本实验为了保持初值一致，使用统一的空间变量及软走廊约束。可微走廊的局部截面反求、FIRI 并集的活动约束切换都会影响优化。详细公式、容差与复现命令见 [实验说明](../gcopter/experiments/trajectory/README.md)。','',
'![高预算耗时与走廊违约](figures/trajectory_comparison.png)','','## 设置与可信范围','',
'- AMD Ryzen 7 9700X，GCC 15.2 Release；串行、单线程，无并行跑图。编译地址空间上限 2 GiB；每个运行进程 512 MiB。',
'- 轨迹按种子弧长约 1 m 分段，初速分配 2 m/s，起终点速度/加速度为零。每次从相同初值重新开始，并核对优化向量哈希。',
'- 每条方法先预热一次，再正式运行三次，交替顺序；表中为正式优化时间中位数。走廊生成时间单独记录一次，不混入轨迹优化。',
'- 小预算：16 个积分间隔/段、800 次迭代上限、位置惩罚 1e5。高预算：128 个积分间隔/段、3200 次迭代上限、位置惩罚 1e6。两档同时改变了积分、预算和权重，属于整体配置对照，不是单因素消融。',
'- 通过要求：稠密检查中的走廊违约 ≤5 mm，原始膨胀障碍穿透 ≤1e-6 m，动力学限制允许 1% 容差，且截面反求无失败。审计采样至少为优化的 4 倍，时间间隔不超过 5 ms。',
'- “通过”只表示上述采样/容差要求，不表示连续时间安全证书或优化收敛。失败轨迹的耗时仍列出，不当作成功解耗时。','']

for profile,title in [('final_fast','小预算'),('final_dense','高预算')]:
    lines += [f'## {title}结果','', '| 输入 | FIRI 优化 ms | 可微优化 ms | FIRI 违约 mm | 可微违约 mm | FIRI / 可微采样通过 |', '|---|---:|---:|---:|---:|---|']
    for i,name in enumerate(names):
        r=pair(profile,i); f,h=r['firi'],r['C3']
        lines.append(f"| {name} | {f['median_ms']:.1f} | {h['median_ms']:.1f} | {f['max_corridor_error_m']*1000:.2f} | {h['max_corridor_error_m']*1000:.2f} | {'通过' if f['sampled_feasible'] else '未通过'} / {'通过' if h['sampled_feasible'] else '未通过'} |")
    counts={m:sum(pair(profile,i)[m]['sampled_feasible'] for i in range(6)) for m in ['firi','C3']}
    lines += ['',f"严格 5 mm 口径下：FIRI {counts['firi']}/6、可微 {counts['C3']}/6。重复运行是同一输入的耗时重复，不算额外独立成功案例。",'']

lines += ['## 高预算的轨迹质量与终止','',
'以下展示全部输出，包含不满足走廊容差的轨迹，不能直接从某一项较低得出可行解更优。长度为稠密折线积分，jerk 能量为样条解析积分。','',
'| 输入 | 方法 | 飞行时间 s | 长度 m | jerk 能量 | 迭代数 / 状态 | 优化节点违约 / 稠密违约 mm |',
'|---|---|---:|---:|---:|---|---:|']
for i,name in enumerate(names):
    for m,r in pair('final_dense',i).items():
        lines.append(f"| {name} | {m} | {r['duration_s']:.3f} | {r['length_m']:.3f} | {r['jerk_energy']:.3f} | {r['iterations']} / {r['status']} | {r['optimization_sample_violation_m']*1000:.2f} / {r['max_corridor_error_m']*1000:.2f} |")
lines += ['', '状态 1：相对目标变化停止；-1008：达到迭代上限；-1009：线搜索区间过小。三者均不自动意味着可行或最优。','',
'## 诊断','',
'| 输入 | FIRI 单次梯度评估 μs | 可微单次梯度评估 μs | 初始种子在 FIRI 外的最大距离指标 m | FIRI 相邻不重叠数 |',
'|---|---:|---:|---:|---:|']
for i,name in enumerate(names):
    r=pair('final_dense',i); f,h=r['firi'],r['C3']
    lines.append(f"| {name} | {f['cost_per_evaluation_us']:.1f} | {h['cost_per_evaluation_us']:.1f} | {f['firi_seed_outside_m']:.3f} | {f['firi_disconnected_neighbors']} |")
allruns=[r for profile in profiles.values() for c in profile['cases'] for r in c['runs']]
maxgradient=max(r['gradient_relative_error'] for r in allruns)
peak=max(c['peak_rss_kib'] for p in profiles.values() for c in p['cases'])/1024
collisions=sum(r['max_penetration_m']>1e-6 for r in allruns if r['repeat']==0)
lines += ['',
f'- 完整目标对初始优化向量的中心有限差分检查，两档、六输入、两方法最大相对误差 {maxgradient:.3g}。这只检查测试点附近的梯度，不证明全局光滑。',
f'- 每个进程最大实测 RSS {peak:.1f} MiB；CPU/wall 上限及每次终止状态见原始 JSON。',
f'- 两档 24 条不同配置/输入/方法输出中，原始膨胀障碍采样检查出现穿透的有 {collisions} 条；走廊违规仍独立计为失败，不能以“碰撞采样没发现问题”代替走廊约束。',
'- 表中的“初始种子在 FIRI 外”按原始折线采样测量，不是初始样条碰撞量。它说明 shortcut/侵蚀之后原种子不一定仍被完整覆盖；相同初值不是“两边都可行的初值”。',
'- 更高采样密度仍有节点间违约，证明仅增加积分点不能代替连续约束。',
'- 可微几何在当前接入方式下没有自动变成更快的轨迹优化：局部截面求根与隐函数梯度是额外开销，多根与活动分支切换还可能影响线搜索。这个结论针对本适配器，不能归结为 C3 曲线本身慢。','',
'空旷直线控制用例已通过：两边初值一致、采样可行，飞行时间、长度、jerk 能量在 1e-4 相对容差内一致。新实验目标与复用公共生成函数后的原体积 benchmark 均已编译通过。','','## 下一步应优先验证','',
'1. 改进约束满足方式：用审计发现的最大违约位置增补约束/积分点，或做段级连续证书；先获得两边满足同一容差的解，再比较质量和收敛速度。',
'2. 为可微走廊显式引入单调进度变量 u(t) 或局部坐标，研究有序、单射条件；避免每个样条采样点都做全局范围截面反求，并减少分支切换。',
'3. 另做原生管线对照：允许 FIRI 使用 PolytopeSpatialMap 与自己的初值，并同时记录其初值变化。它回答“实际系统谁更好”，但与本次“严格相同初值”是不同实验。','',
'## 证据','',
'[小预算原始结果](validation/trajectory_comparison/final_fast.json) · [高预算原始结果](validation/trajectory_comparison/final_dense.json)。包含输入 SHA256、二进制 SHA256、参数、初值哈希、全部预热/正式计时、动力学极值、几何违规和终止状态。','',
'稠密轨迹 CSV、共同初始航点和完整日志位于 `.scratch/trajectory_comparison/final_fast` 与 `final_dense`。冻结地图沿用本地交互快照，不重复加入 Git。']
a.report.parent.mkdir(parents=True,exist_ok=True)
a.report.write_text('\n'.join(lines)+'\n')
