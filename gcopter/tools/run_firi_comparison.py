#!/usr/bin/env python3
"""Sequential, memory-limited comparison of saved seed/map snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (25, 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, type=Path,
                        help='one saved request directory; each input runs sequentially')
    parser.add_argument('--binary', type=Path, default=Path('build/gcopter/corridor_firi_benchmark'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=65536)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--policy', choices=['legacy', 'matched-box'], default='legacy')
    parser.add_argument('--range', type=float, default=3.)
    parser.add_argument('--model', choices=['both','points','solid'], default='both')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    prefix = Path('.scratch/ompl/usr/lib/x86_64-linux-gnu').resolve()
    if prefix.is_dir():
        env['LD_LIBRARY_PATH'] = str(prefix)+':'+env.get('LD_LIBRARY_PATH', '')
    records = []
    for case, request in enumerate(args.input, 1):
        cfg = json.loads((request/'metrics.json').read_text())
        cloud = request.parent/f"map_{cfg['map_version']}.xyz"
        seed = request/'seed.xyz'
        for model in (('points', 'solid') if args.model=='both' else (args.model,)):
            dest = args.output/f'{case}_{model}'
            dest.mkdir(parents=True, exist_ok=True)
            cmd = ['/usr/bin/time', '-f', '%M', '-o', str(dest/'peak_rss_kib.txt'),
                   str(args.binary.resolve()), str(seed), str(cloud), str(dest), model,
                   *map(str, cfg['bounds_lower']), *map(str, cfg['bounds_upper']),
                   str(cfg['resolution']), str(args.repeats), str(args.samples),args.policy,str(args.range)]
            with (dest/'run.log').open('w') as log:
                run = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       preexec_fn=limits, start_new_session=True)
                try:
                    code = run.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid, signal.SIGKILL)
                    run.wait()
                    raise
            if code:
                raise RuntimeError(f'{request} {model} failed under resource limits; see {dest}/run.log')
            record = json.loads((dest/'metrics.json').read_text())
            record.update(request=str(request),
                          seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                          map_sha256=hashlib.sha256(cloud.read_bytes()).hexdigest(),
                          max_rss_mib=int((dest/'peak_rss_kib.txt').read_text().strip())/1024,
                          virtual_memory_limit_mib=512)
            records.append(record)
            (args.output/'results.json').write_text(json.dumps(records, indent=2)+'\n')
            print(f"{case} {model}: H {record['hermite_median_ms']:.1f} ms / "
                  f"{record['hermite_union_m3']:.1f} m3; "
                  f"F {record['firi_median_ms']:.1f} ms / {record['firi_union_m3']:.1f} m3; "
                  f"RSS {record['max_rss_mib']:.1f} MiB; connected={record['firi_connected']}", flush=True)


if __name__ == '__main__':
    main()
