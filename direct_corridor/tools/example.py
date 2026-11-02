#!/usr/bin/env python3
"""Generate a deterministic multi-turn 3-D seed/map, run CLI, optionally audit/plot."""
import argparse
import math
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--binary', required=True)
parser.add_argument('--output', default='.scratch/direct_corridor_example')
parser.add_argument('--plot', action='store_true', help='requires numpy and matplotlib')
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
seed = [(0, 0, 1.8), (3, 0, 2), (3, 3, 2.5), (0, 3, 3), (0, 6, 3), (4, 6, 2.5)]

def segment_distance(p, a, b):
    d = [b[i]-a[i] for i in range(3)]
    t = max(0, min(1, sum((p[i]-a[i])*d[i] for i in range(3))/sum(v*v for v in d)))
    return math.sqrt(sum((p[i]-a[i]-t*d[i])**2 for i in range(3)))

obstacles = []
for i in range(13):
    for j in range(17):
        for k in range(8):
            p = (-1+i*.5, -1+j*.5, .5+k*.5)
            if min(segment_distance(p, a, b) for a, b in zip(seed, seed[1:])) > .95:
                obstacles.append(p)
for filename, values in [('seed.xyz', seed), ('obstacles.xyz', obstacles)]:
    (out/filename).write_text(''.join(' '.join(map(str, p))+'\n' for p in values))
subprocess.run([args.binary, str(out/'seed.xyz'), str(out/'obstacles.xyz'), '.13', '.1',
                '-2', '-2', '0', '6', '8', '5', str(out)], check=True)
if args.plot:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surface = np.loadtxt(out/'surface.csv', delimiter=',', skiprows=1)
    cloud = np.array(obstacles)
    xyz = surface[:, 2:]
    minimum = math.inf
    for start in range(0, len(xyz), 256):
        distances = np.linalg.norm(xyz[start:start+256, None, :]-cloud[None, :, :], axis=2)
        minimum = min(minimum, float(distances.min())-.13-.1-.02)
    assert minimum > -1e-8, f'independent sampled collision: {minimum}'
    (out/'sample_audit.txt').write_text(f'boundary samples={len(xyz)}; obstacle spheres={len(cloud)}; '
                                       f'minimum sampled clearance={minimum:.9f} m\n'
                                       'Sampling is an additional audit, not the continuous certificate.\n')
    coefficients = np.loadtxt(out/'controls.csv', delimiter=',', skiprows=1)
    radii = []
    centers = []
    for u in np.linspace(0, 1, 401):
        ends = coefficients[::6,1]
        span = min(len(ends)-1,int(np.searchsorted(ends,u,side='right')))
        block = coefficients[6*span:6*span+6]
        t = (u-block[0,0])/(block[0,1]-block[0,0])
        basis = np.array([math.comb(5,j)*t**j*(1-t)**(5-j) for j in range(6)])
        c = basis @ block[:,2:]
        centers.append(c[:3])
        radii.append(np.linalg.svd(c[3:].reshape(2, 3).T, compute_uv=False))
    centers, radii = np.array(centers), np.array(radii)
    fig = plt.figure(figsize=(12, 5.5), layout='constrained')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*cloud.T, s=2, c='#64748b', alpha=.1)
    rings = xyz.reshape(401, 64, 3)
    for ring in rings[::5]:
        ring = np.vstack([ring, ring[0]])
        ax.plot(*ring.T, color='#8b5cf6', linewidth=.8, alpha=.65)
    ax.plot(*np.array(seed).T, '--', color='#f59e0b', label='Seed polyline')
    ax.plot(*centers.T, color='#0891b2', label='C2 reference')
    ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title='Direct quintic Hermite corridor')
    ax.set_box_aspect((8, 10, 5))
    ax.view_init(elev=38, azim=-60)
    ax.legend(loc='upper left', fontsize=8)
    ax = fig.add_subplot(122)
    ax.plot(np.linspace(0, 1, 401), radii[:, 0], label='Major semiaxis', color='#8b5cf6')
    ax.plot(np.linspace(0, 1, 401), radii[:, 1], label='Minor semiaxis', color='#0891b2')
    ax.set(xlabel='Normalized path parameter u', ylabel='Section semiaxis [m]', title='Local width adaptation')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out/'corridor.png', dpi=160)
    print((out/'sample_audit.txt').read_text(), out/'corridor.png')
