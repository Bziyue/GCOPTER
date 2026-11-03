#!/usr/bin/env python3
"""Serial, memory-bounded paired ellipse/Minkowski corridor experiments."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]
# Argument order shared with minkowski_benchmark. The ellipse reference uses
# its own spacing argument and always two inflation rounds.
FIELDS = ('templates', 'subdivisions', 'steps', 'thinness', 'angle', 'spacing', 'inflation', 'adapt', 'product', 'affine', 'smoothing')
DEFAULT = (4, 0, 12, 0.05, 0, 0.4, 2, 0, 0, 0, 0)


def variant(**changes):
    values = dict(zip(FIELDS, DEFAULT))
    values.update(changes)
    return tuple(values[name] for name in FIELDS)


VARIANTS = {
    'm2_smooth_sparse': variant(templates=2, spacing=.6, adapt=1, smoothing=4),
    'm2_smooth': variant(templates=2, adapt=1, smoothing=4),
    'm2': variant(templates=2),
    'm4': variant(),
    'm4_split1': variant(subdivisions=1),
    'm4_split2': variant(subdivisions=2),
    'm4_fast': variant(steps=8),
    'm4_thick': variant(thinness=0.15),
    'm4_round': variant(thinness=0.3),
    'm4_angle': variant(angle=0.39269908169872414),
    'm4_sparse': variant(spacing=0.6),
    'm4_single': variant(inflation=1),
    'm4_adapt': variant(adapt=1),
    'm4_adapt_fast': variant(steps=8, adapt=1),
    'm4_adapt_sparse': variant(spacing=0.6, adapt=1),
    'm4_adapt_thick': variant(thinness=0.15, adapt=1),
    'm4_product': variant(product=1),
    'm4_product_fast': variant(steps=8, product=1),
    'm4_adapt_product': variant(adapt=1, product=1),
    'm4_needle': variant(thinness=0.01),
    'm4_adapt_needle': variant(thinness=0.01, adapt=1),
    'affine2': variant(templates=2, affine=1),
    'affine4': variant(affine=1),
    'affine2_thick': variant(templates=2, thinness=0.2, affine=1),
    'affine4_sparse': variant(spacing=0.6, affine=1),
    'affine2_round': variant(templates=2, steps=8, thinness=0.4, affine=1),
    'affine2_angle': variant(templates=2, thinness=0.2, angle=0.7853981633974483, affine=1),
    'affine4_fast': variant(steps=8, affine=1),
    'affine4_thick': variant(steps=8, thinness=0.25, affine=1),
    'm4_smooth': variant(adapt=1, smoothing=4),
    'm4_smooth_sparse': variant(spacing=0.6, adapt=1, smoothing=4),
    'm4_smooth_thick': variant(thinness=0.15, adapt=1, smoothing=4),
    'm4_smoother': variant(adapt=1, smoothing=16),
}


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (50, 55))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variants', nargs='+', choices=VARIANTS)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT/'.scratch/minkowski_round6/build/minkowski_benchmark')
    parser.add_argument('--cases', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--samples', type=int, default=32768)
    parser.add_argument('--ellipse-spacing', type=float, default=.4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(args.inputs.read_text())[:args.cases]
    sources = [ROOT/'direct_corridor/src/direct_corridor.cpp',
               ROOT/'direct_corridor/src/hermite_optimizer.hpp',
               ROOT/'gcopter/tools/corridor_volume.hpp', Path(__file__),
               *Path(__file__).parent.glob('minkowski*.*')]
    manifest = dict(binary_sha256=digest(args.binary),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    variants={name: VARIANTS[name] for name in args.variants},
                    runtime_address_space_mib=512, cpu_seconds=50, wall_seconds=60,
                    repeats=args.repeats, shifts=8, samples_per_shift=args.samples,
                    ellipse_spacing=args.ellipse_spacing,
                    baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    records = []
    for name in args.variants:
        for case, raw in enumerate(inputs, 1):
            request = ROOT/raw
            cfg = json.loads((request/'metrics.json').read_text())
            seed = request/'seed.xyz'
            cloud = request.parent/f"map_{cfg['map_version']}.xyz"
            dest = args.output/name/str(case)
            dest.mkdir(parents=True)
            cmd = ['/usr/bin/time', '-f', '%M', '-o', str(dest/'rss_kib.txt'),
                   str(args.binary.resolve()), str(seed), str(cloud), str(dest),
                   *map(str, cfg['bounds_lower']), *map(str, cfg['bounds_upper']),
                   str(cfg['resolution']), str(args.repeats), str(args.samples),
                   *map(str, VARIANTS[name]), str(args.ellipse_spacing)]
            with (dest/'run.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, preexec_fn=limits,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    code = -signal.SIGKILL
            record = dict(variant=name, case=case, request=raw, returncode=code,
                          seed_sha256=digest(seed), map_sha256=digest(cloud))
            if code == 0:
                record.update(json.loads((dest/'metrics.json').read_text()))
                record['max_rss_mib'] = int((dest/'rss_kib.txt').read_text().strip())/1024
                record['speedup_percent'] = 100*(1-record['minkowski_median_ms']/record['ellipse_median_ms'])
                record['volume_gain_percent'] = 100*record['volume_delta_m3']/record['ellipse_union_m3']
                print(f"{name} case {case}: {record['ellipse_median_ms']:.2f} -> "
                      f"{record['minkowski_median_ms']:.2f} ms; volume "
                      f"{record['volume_gain_percent']:+.2f}%; RSS {record['max_rss_mib']:.1f} MiB", flush=True)
            else:
                record['status'] = 'FAILED'
                record['message'] = (dest/'run.log').read_text()[-2000:]
                print(f'{name} case {case}: FAILED {record["message"]}', flush=True)
            records.append(record)
            (args.output/'results.json').write_text(json.dumps(records, indent=2)+'\n')
    if any(r['returncode'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
