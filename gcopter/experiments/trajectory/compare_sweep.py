#!/usr/bin/env python3
"""Serial frozen-input SweepCorridor comparison; no ROS or random path search."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True, help='Frozen case manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--stage', choices=['generation', 'trajectory', 'profile'], required=True)
    p.add_argument('--cases', help='Comma-separated IDs; default: every manifest case')
    p.add_argument('--cpu', type=int, default=2)
    p.add_argument('--memory-gib', type=int, default=4)
    args = p.parse_args()
    cases = json.loads(args.inputs.read_text())['cases']
    if args.cases:
        ids = set(map(int, args.cases.split(',')))
        cases = [c for c in cases if c['case'] in ids]
    # Validate each external snapshot once, before starting the benchmark.
    checked = set()
    for c in cases:
        for kind in ('seed', 'map'):
            if c[kind] not in checked:
                digest = hashlib.sha256(Path(c[kind]).read_bytes()).hexdigest()
                if digest != c[kind + '_sha256']:
                    raise ValueError(f'Input hash mismatch: {c[kind]}')
                checked.add(c[kind])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', SWEEP_C3_ONLY='1')
    for key in ('GCOPTER_PROFILE', 'SWEEP_QUERY_MEASURE'):
        env.pop(key, None)
    if args.stage == 'profile':
        env['GCOPTER_PROFILE'] = '1'
    def limit():
        memory = args.memory_gib * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    stage=args.stage, cpu=args.cpu, memory_gib=args.memory_gib, records=[])
    for c in cases:
        target = args.output / str(c['case'])
        target.mkdir(parents=True, exist_ok=True)
        cmd = [str(args.binary.resolve()), c['seed'], c['map'], str(target),
               *map(str, c['bounds_lower']), *map(str, c['bounds_upper'])]
        if args.stage == 'generation':
            # Query microbenchmarks are separate from generation wall timing.
            if c['case'] < 7:
                env['SWEEP_QUERY_MEASURE'] = '1'
            else:
                env.pop('SWEEP_QUERY_MEASURE', None)
            cmd += ['7', '0']
        else:
            cmd += [str(c['resolution']), '1' if args.stage == 'profile' else '3',
                    '1.0', '100000', '16', '0' if args.stage == 'profile' else '1',
                    '800', 'bezier', '2']
        start = time.monotonic()
        with (target / 'process.log').open('w') as log:
            result = subprocess.run(['/usr/bin/time', '-v', *cmd], stdout=log, stderr=log,
                                    env=env, preexec_fn=limit, timeout=180)
        manifest['records'].append(dict(case=c, command=cmd, exit_code=result.returncode,
                                        elapsed_s=time.monotonic()-start))
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(c['case'], result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f"Case {c['case']} failed; see {target / 'process.log'}")


if __name__ == '__main__':
    main()
