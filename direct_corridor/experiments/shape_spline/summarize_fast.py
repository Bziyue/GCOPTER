#!/usr/bin/env python3
"""Collect the bounded fast-shape experiments without replacing failed trials."""
from pathlib import Path
import hashlib
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / '.scratch/fast_shape'
OUT = ROOT / 'docs/validation/fast_shape'
OUT.mkdir(parents=True, exist_ok=True)
MODES = ['baseline', 'boundary-12', 'continuation']
LABELS = ['原基线', '边界步长＋局部更新', '第二轮延续参数']
NAMES = ['Perlin-A1', 'Perlin-B1', 'Maze-1', 'Perlin-A2', 'Perlin-B2', 'Maze-2']

def read(p):
    return json.loads(p.read_text())

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

generation = read(RUN / 'generation/results.json')
result = {'generation': generation, 'trajectory': {}, 'cases': [], 'pilots': {}}
result['lazy_step_equivalence'] = read(RUN / 'lazy_step_equivalence.json')
assert all(r['identical'] for r in result['lazy_step_equivalence'])
result['before_lazy_step_generation'] = read(RUN / 'before_lazy_step/generation/results.json')
for name in ['pilot', 'coordinate_pilot', 'boundary_pilot', 'coarse_pilot', 'continuation_pilot']:
    result['pilots'][name] = read(RUN / name / 'results.json')
for mode in MODES:
    path = RUN / 'trajectory' / mode
    manifest = read(path / 'manifest.json')
    records = []
    for c in manifest['cases']:
        case = path / f"case_{c['case']:02}"
        record = {**c, 'runs': rows(case / 'runs.jsonl')}
        if c['returncode'] != 0:
            record['failure_log'] = (case / 'stderr.log').read_text()
            case = RUN / 'diagnostic' / mode / f"case_{c['case']:02}" / 'case_00'
            record['diagnostic_manifest'] = read(case.parent / 'manifest.json')
            assert record['diagnostic_manifest']['cases'][0]['request'] == c['request']
            record['diagnostic_runs'] = rows(case / 'runs.jsonl')
        record['csv_sha256'] = {n: sha(case / n) for n in
            ['initial_waypoints.csv', 'hermite_trajectory.csv', 'firi_trajectory.csv']}
        records.append(record)
    result['trajectory'][mode] = {'manifest': manifest, 'cases': records}

for c in generation['cases']:
    i = c['case']
    assert c['returncode'] == 0
    record = {'case': i, 'name': NAMES[i], 'generation': {}, 'trajectory': {},
              'volume': {r['mode']: r for r in c['volume']}}
    assert c['volume'][0]['baseline_volume_m3'] == c['volume'][1]['baseline_volume_m3']
    initials, csvs, firi = set(), set(), set()
    for mode in MODES:
        timed = [r for r in c['generation'] if r['mode'] == mode and r['repeat'] >= 0]
        assert len(timed) == generation['parameters']['repeats']
        record['generation'][mode] = {**timed[0], 'median_ms': statistics.median(r['generation_ms'] for r in timed),
            'min_ms': min(r['generation_ms'] for r in timed), 'max_ms': max(r['generation_ms'] for r in timed)}
        tr = result['trajectory'][mode]['cases'][i]
        assert tr['map_sha256'] == c['map_sha256'] and tr['seed_sha256'] == c['seed_sha256']
        runs = tr.get('diagnostic_runs', tr['runs'])
        ts = [r for r in runs if r['method'] == 'hermite' and r['repeat'] >= 0]
        assert len(ts) == 3
        checked = next(r['gradient_relative_error'] for r in runs if r['method'] == 'hermite' and r['repeat'] == -1)
        record['trajectory'][mode] = {**ts[0], 'median_ms': statistics.median(r['optimize_ms'] for r in ts),
            'gradient_validated': tr['returncode'] == 0, 'checked_gradient_relative_error': checked}
        if tr['returncode'] == 0: assert 0 <= checked < 1e-4
        initials.update(r['initial_hash'] for r in runs)
        csvs.add(tr['csv_sha256']['initial_waypoints.csv'])
        firi.add(tr['csv_sha256']['firi_trajectory.csv'])
    assert len(initials) == len(csvs) == len(firi) == 1
    record['initial_hash'] = initials.pop()
    record['initial_csv_sha256'] = csvs.pop()
    record['firi_csv_sha256'] = firi.pop()
    record['baseline_geometry_unchanged'] = (
        sha(RUN / 'generation' / f'case_{i:02}' / 'baseline_sections.csv') ==
        sha(ROOT / '.scratch/integrated_shape/generation' / f'case_{i:02}' / 'baseline_sections.csv'))
    record['baseline_trajectory_unchanged'] = (
        result['trajectory']['baseline']['cases'][i]['csv_sha256']['hermite_trajectory.csv'] ==
        sha(ROOT / '.scratch/integrated_shape/trajectory/baseline' / f'case_{i:02}' / 'hermite_trajectory.csv'))
    assert record['baseline_geometry_unchanged'] and record['baseline_trajectory_unchanged']
    result['cases'].append(record)

result['membership_audit'] = read(RUN / 'audit/results.json')
sources = list((ROOT / 'direct_corridor/experiments/shape_spline').glob('*.cpp'))
sources += list((ROOT / 'direct_corridor/experiments/shape_spline').glob('*.hpp'))
sources += [ROOT / p for p in ['direct_corridor/src/direct_corridor.cpp', 'direct_corridor/src/hermite_optimizer.hpp',
    'direct_corridor/src/generation_pipeline.hpp', 'gcopter/experiments/trajectory/benchmark.cpp']]
result['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}
(OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')

table = ['| 路径 | 原基线体积 m³ | 边界更新体积变化 | 延续参数体积变化 | 完整生成 ms：基线 / 边界 / 延续 |',
         '|---|---:|---:|---:|---:|']
for r in result['cases']:
    a, b = r['volume']['boundary-12'], r['volume']['continuation']
    table.append(f"| {r['name']} | {a['baseline_volume_m3']:.2f} | {100*a['paired_delta_m3']/a['baseline_volume_m3']:+.2f}% | "
                 f"{100*b['paired_delta_m3']/b['baseline_volume_m3']:+.2f}% | " +
                 ' / '.join(f"{r['generation'][m]['median_ms']:.2f}" for m in MODES) + ' |')
table += ['', '| 路径 | 轨迹优化 ms | 飞行时长 s | jerk 能量 | 局部位置残差 mm | 采样通过 |',
          '|---|---:|---:|---:|---:|---|']
for r in result['cases']:
    fields = []
    for k, factor, digits in [('median_ms', 1, 1), ('duration_s', 1, 2), ('jerk_energy', 1, 2), ('max_corridor_error_m', 1000, 2)]:
        fields.append(' / '.join(f"{r['trajectory'][m][k]*factor:.{digits}f}" for m in MODES))
    fields.append(' / '.join('是' if r['trajectory'][m]['sampled_feasible'] else '否' for m in MODES))
    table.append('| ' + r['name'] + ' | ' + ' | '.join(fields) + ' |')
(RUN / 'tables.md').write_text('\n'.join(table) + '\n')
print('\n'.join(table))

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 10})
colors = ['#8795a5', '#8a57b4', '#198779']
x = np.arange(6)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
for k, mode in enumerate(MODES):
    if k:
        v = [100*r['volume'][mode]['paired_delta_m3']/r['volume'][mode]['baseline_volume_m3'] for r in result['cases']]
        e = [100*r['volume'][mode]['paired_ci95_m3']/r['volume'][mode]['baseline_volume_m3'] for r in result['cases']]
        axes[0].bar(x+(k-1.5)*.32, v, .32, color=colors[k], yerr=e, capsize=2)
    for ax, field in [(axes[1], 'generation'), (axes[2], 'trajectory')]:
        ax.bar(x+(k-1)*.25, [r[field][mode]['median_ms'] for r in result['cases']], .25, color=colors[k], label=LABELS[k])
for ax, title in zip(axes, ['走廊并集体积变化', '完整走廊生成耗时', '同初值轨迹优化耗时']):
    ax.set_title(title); ax.set_xticks(x, NAMES, rotation=30, ha='right')
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
axes[0].axhline(0, color='#555', lw=.8)
axes[0].set_ylabel('相对原基线 / %（配对 95% 区间）')
axes[1].set_ylabel('中位数 / ms'); axes[2].set_ylabel('中位数 / ms')
axes[1].legend(fontsize=8, frameon=False)
fig.suptitle('减少五次形状优化成本：与原基线的同输入对照', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .95])
fig.savefig(ROOT / 'docs/figures/fast_shape_comparison.png', dpi=170)
plt.close(fig)
