#!/usr/bin/env python3
"""Summarize the serial integrated-quintic comparison on six frozen requests."""
from pathlib import Path
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / '.scratch/integrated_shape'
OUT = ROOT / 'docs/validation/integrated_shape'
OUT.mkdir(parents=True, exist_ok=True)
MODES = ['baseline', 'hermite-free', 'integrated-quintic']
LABELS = ['原基线', '离线五次', '集成五次']
NAMES = ['Perlin-A1', 'Perlin-B1', 'Maze-1', 'Perlin-A2', 'Perlin-B2', 'Maze-2']
COLORS = ['#8795a5', '#8a57b4', '#198779']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


generation = read(RUN / 'generation/results.json')
result = {'generation': generation, 'trajectory': {}, 'cases': []}
for mode in MODES:
    directory = RUN / 'trajectory' / mode
    manifest = read(directory / 'manifest.json')
    cases = []
    for record in manifest['cases']:
        i = record['case']
        path = directory / f'case_{i:02}'
        row = {**record, 'runs': lines(path / 'runs.jsonl'),
               'gradient_validated': record['returncode'] == 0}
        if record['returncode'] != 0:
            row['failure_log'] = (path / 'stderr.log').read_text()
            path = RUN / 'diagnostic' / mode / f'case_{i:02}' / 'case_00'
            row['diagnostic_manifest'] = read(path.parent / 'manifest.json')
            assert row['diagnostic_manifest']['cases'][0]['request'] == record['request']
            row['diagnostic_runs'] = lines(path / 'runs.jsonl')
        row['csv_sha256'] = {name: sha(path / name) for name in
                             ['initial_waypoints.csv', 'hermite_trajectory.csv', 'firi_trajectory.csv']}
        cases.append(row)
    result['trajectory'][mode] = {'manifest': manifest, 'cases': cases}

for c in generation['cases']:
    i = c['case']
    assert c['returncode'] == 0 and len(c['volume']) == 2
    assert c['volume'][0]['baseline_volume_m3'] == c['volume'][1]['baseline_volume_m3']
    row = {'case': i, 'name': NAMES[i], 'generation': {},
           'volume': {r['mode']: r for r in c['volume']}, 'trajectory': {}}
    hashes, csv_hashes, firi_hashes = set(), set(), set()
    for mode in MODES:
        runs = [r for r in c['generation'] if r['mode'] == mode and r['repeat'] >= 0]
        assert len(runs) == 3
        row['generation'][mode] = {**runs[0], 'median_generation_ms':
                                   statistics.median(r['generation_ms'] for r in runs)}
        tr = result['trajectory'][mode]['cases'][i]
        assert tr['seed_sha256'] == c['seed_sha256'] and tr['map_sha256'] == c['map_sha256']
        runs = tr.get('diagnostic_runs', tr['runs'])
        timed = [r for r in runs if r['method'] == 'hermite' and r['repeat'] >= 0]
        assert len(timed) == 3
        row['trajectory'][mode] = {**timed[0], 'median_optimize_ms':
                                  statistics.median(r['optimize_ms'] for r in timed),
                                  'gradient_validated': tr['gradient_validated']}
        if tr['gradient_validated']:
            checked = next(r['gradient_relative_error'] for r in runs
                           if r['method'] == 'hermite' and r['repeat'] == -1)
            assert 0 <= checked < 1e-4
            row['trajectory'][mode]['checked_gradient_relative_error'] = checked
        hashes.update(r['initial_hash'] for r in runs)
        csv_hashes.add(tr['csv_sha256']['initial_waypoints.csv'])
        firi_hashes.add(tr['csv_sha256']['firi_trajectory.csv'])
    assert len(hashes) == len(csv_hashes) == len(firi_hashes) == 1
    row['initial_hash'] = hashes.pop()
    row['initial_csv_sha256'] = csv_hashes.pop()
    row['firi_csv_sha256'] = firi_hashes.pop()
    # The center geometry is controlled across all three alternatives.
    center = None
    row['center_identical'] = True
    for mode in MODES:
        data = np.loadtxt(RUN / 'generation' / f'case_{i:02}' / f'{mode}_sections.csv',
                          delimiter=',', skiprows=1)
        if center is None:
            center = data[:, :4]
        else:
            row['center_identical'] &= bool(np.array_equal(center, data[:, :4]))
    assert row['center_identical']
    result['cases'].append(row)

audit_path = RUN / 'audit/results.json'
if audit_path.exists():
    result['global_membership_audit'] = read(audit_path)
previous = ROOT / '.scratch/shape_spline'
equivalence = []
for i in range(6):
    for mode in MODES[:2]:
        pairs = [(RUN / 'generation' / f'case_{i:02}' / f'{mode}_sections.csv',
                  previous / 'generation' / f'case_{i:02}' / f'{mode}_sections.csv')]
        pairs += [(RUN / 'trajectory' / mode / f'case_{i:02}' / name,
                   previous / 'trajectory' / mode / f'case_{i:02}' / name)
                  for name in ['initial_waypoints.csv', 'hermite_trajectory.csv', 'firi_trajectory.csv']]
        for current, old in pairs:
            equivalence.append({'case': i, 'mode': mode, 'file': current.name,
                                'sha256': sha(current), 'identical': sha(current) == sha(old)})
assert all(r['identical'] for r in equivalence)
result['previous_equivalence'] = equivalence
(RUN / 'previous_equivalence.json').write_text(json.dumps(equivalence, indent=2) + '\n')
result['warm_start_pilot'] = read(RUN / 'warm_pilot/results.json')
result['warm_start_pilot']['source_sha256'] = sha(RUN / 'warm_pilot/refit.cpp')
result['warm_start_pilot']['adopted'] = False
result['warm_start_pilot']['settings'] = {'initial_scale': 0.98, 'barrier_parameters': [0.01, 0.001],
                                         'max_iterations_per_stage': 20}
result['experiment_snapshot_commit'] = '57c3cd2'
source = list((ROOT / 'direct_corridor/experiments/shape_spline').glob('*.cpp'))
source += list((ROOT / 'direct_corridor/experiments/shape_spline').glob('*.hpp'))
source += [ROOT / x for x in ['direct_corridor/src/direct_corridor.cpp',
           'direct_corridor/src/hermite_optimizer.hpp', 'direct_corridor/src/generation_pipeline.hpp',
           'gcopter/experiments/trajectory/benchmark.cpp']]
result['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(source)}
(OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')

tables = ['| 路径 | 体积 m³：原基线 / 离线 / 集成 | 集成相对基线 | 生成 ms：原基线 / 离线 / 集成 | 集成相对离线耗时 |',
          '|---|---:|---:|---:|---:|']
for r in result['cases']:
    offline, fused = r['volume']['hermite-free'], r['volume']['integrated-quintic']
    times = [r['generation'][m]['median_generation_ms'] for m in MODES]
    volumes = [fused['baseline_volume_m3'], offline['volume_m3'], fused['volume_m3']]
    tables.append(f"| {r['name']} | " + ' / '.join(f'{v:.2f}' for v in volumes) +
                  f" | {100*fused['paired_delta_m3']/volumes[0]:+.2f}% | " +
                  ' / '.join(f'{t:.1f}' for t in times) + f" | {100*(times[2]/times[1]-1):+.1f}% |")
tables += ['', '| 路径 | 优化耗时 ms | 飞行时长 s | jerk 能量 | 局部查询残差 mm | 采样通过 |',
           '|---|---:|---:|---:|---:|---|']
for r in result['cases']:
    values = []
    for key, factor, digits in [('median_optimize_ms', 1, 1), ('duration_s', 1, 2),
                                ('jerk_energy', 1, 2), ('max_corridor_error_m', 1000, 2)]:
        values.append(' / '.join(f"{r['trajectory'][m][key]*factor:.{digits}f}" for m in MODES))
    values.append(' / '.join('是' if r['trajectory'][m]['sampled_feasible'] else '否' for m in MODES))
    tables.append('| ' + r['name'] + ' | ' + ' | '.join(values) + ' |')
(RUN / 'tables.md').write_text('\n'.join(tables) + '\n')
print('\n'.join(tables))

font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'font.size': 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
x = np.arange(6)
for k, mode in enumerate(MODES):
    if k:
        value = [100*r['volume'][mode]['paired_delta_m3']/r['volume'][mode]['baseline_volume_m3']
                 for r in result['cases']]
        error = [100*r['volume'][mode]['paired_ci95_m3']/r['volume'][mode]['baseline_volume_m3']
                 for r in result['cases']]
        axes[0].bar(x+(k-1.5)*.32, value, .32, color=COLORS[k], yerr=error, capsize=2, label=LABELS[k])
    for ax, field, key in [(axes[1], 'generation', 'median_generation_ms'),
                            (axes[2], 'trajectory', 'median_optimize_ms')]:
        ax.bar(x+(k-1)*.25, [r[field][mode][key] for r in result['cases']], .25,
               color=COLORS[k], label=LABELS[k])
for ax, title in zip(axes, ['走廊并集体积变化', '完整走廊生成耗时', '同初值轨迹优化耗时']):
    ax.set_title(title)
    ax.set_xticks(x, NAMES, rotation=30, ha='right')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('相对原基线 / %（配对 95% 区间）')
axes[1].set_ylabel('中位数 / ms')
axes[2].set_ylabel('中位数 / ms')
axes[1].legend(frameon=False)
fig.suptitle('复用生成器证书的五次非零导数形状优化', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .95])
fig.savefig(ROOT / 'docs/figures/integrated_shape_comparison.png', dpi=170)
plt.close(fig)
