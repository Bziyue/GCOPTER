#!/usr/bin/env python3
"""Small deterministic geometry experiments; no production corridor is emitted.

Needs NumPy only. Run with BLAS threads=1, 512 MiB and a wall/CPU limit.
Floating polynomial roots are numerical references, not interval proofs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from numpy.polynomial import polynomial as pol


def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def extrema(coeff):
    roots = pol.polyroots(pol.polyder(coeff))
    return np.r_[0., 1., roots.real[(abs(roots.imag) < 1e-8) &
                                    (roots.real > 0) & (roots.real < 1)]]


def p2b(degree):
    return np.array([[math.comb(i, j) / math.comb(degree, j) if j <= i else 0.
                      for j in range(degree+1)] for i in range(degree+1)])


def bernstein_nonnegative(coeff, tolerance=1e-9, depth=0):
    """Numerical subdivision bound with explicit unresolved count (not interval arithmetic)."""
    if min(coeff) >= -tolerance:
        return True, 1, 0
    if coeff[0] < -tolerance or coeff[-1] < -tolerance:
        return False, 1, 0
    if depth == 24:
        return False, 1, 1
    work = np.array(coeff, copy=True)
    left, right = [work[0]], [work[-1]]
    while len(work) > 1:
        work = .5*(work[:-1] + work[1:])
        left.append(work[0]); right.append(work[-1])
    a = bernstein_nonnegative(np.array(left), tolerance, depth+1)
    b = bernstein_nonnegative(np.array(right[::-1]), tolerance, depth+1)
    return a[0] and b[0], a[1]+b[1], a[2]+b[2]


class Section:
    """Fixed orientation ellipses/p=4 balls or smooth ellipse Minkowski sums."""
    def __init__(self, family, normals, angle=0., quadrature=4096):
        self.family = family
        n = normals @ rotation(angle)
        self.normals = normals
        if family.startswith('minkowski'):
            angles = [0., math.pi/2]
            if family == 'minkowski4':
                angles += [math.pi/4, 3*math.pi/4]
            self.q = [rotation(a) @ np.diag([1., .05**2]) @ rotation(a).T for a in angles]
            self.support_matrix = np.stack([np.sqrt(np.sum((n @ q)*n, axis=1))
                                            for q in self.q], axis=1)
            theta = np.arange(quadrature)*2*math.pi/quadrature
            u = np.array([np.cos(theta), np.sin(theta)]).T
            t = np.array([-np.sin(theta), np.cos(theta)]).T
            h = np.stack([np.sqrt(np.sum((u @ q)*u, axis=1)) for q in self.q])
            dh = np.stack([np.sum((u @ q)*t, axis=1)/v for q, v in zip(self.q, h)])
            self.mixed = math.pi/quadrature * (h @ h.T - dh @ dh.T)
            self.dimension = len(angles)
        else:
            self.power = 2. if family == 'ellipse' else 4/3
            self.k = abs(n)**self.power
            self.dimension = 2
            p = 2 if family == 'ellipse' else 4
            self.area_factor = 4*math.gamma(1+1/p)**2/math.gamma(1+2/p)

    def geometry(self, a):
        if self.family.startswith('minkowski'):
            support = self.support_matrix @ a
            jac = self.support_matrix
            hess = np.zeros((len(support), len(a), len(a)))
            v = self.mixed @ a
            area = a @ v
            grad = -2*v/area
            curvature = 4*np.outer(v, v)/area**2 - 2*self.mixed/area
        else:
            q = self.power
            total = self.k @ (a**q)
            support = total**(1/q)
            z = self.k * a**(q-1)
            jac = z * total[:, None]**(1/q-1)
            hess = -(q-1)*np.einsum('ni,nj,n->nij', z, z, total**(1/q-2))
            for j in range(2):
                hess[:, j, j] += (q-1)*self.k[:, j]*a[j]**(q-2)*total**(1/q-1)
            area = self.area_factor*np.prod(a)
            grad, curvature = -1/a, np.diag(1/a**2)
        return support, jac, hess, area, grad, curvature


def section_objective(x, shape, bounds, mu):
    a = x[2:]
    if min(a) <= 0:
        return math.inf, None, None, None
    h, jac, hess, area, g_area, h_area = shape.geometry(a)
    gap = bounds - shape.normals @ x[:2] - h
    if min(gap) <= 0 or area <= 0:
        return math.inf, None, None, None
    j = np.column_stack((shape.normals, jac))
    cost = -math.log(area) - mu*(sum(np.log(gap)) + sum(np.log(a)))
    grad = mu*(j.T @ (1/gap))
    grad[2:] += g_area - mu/a
    hessian = mu*(j.T @ (j/gap[:, None]**2))
    hessian[2:, 2:] += h_area + np.diag(mu/a**2) + mu*np.einsum('n,nij->ij', 1/gap, hess)
    return cost, grad, hessian, area


def solve_section(shape, bounds):
    x = np.r_[0., 0., np.full(shape.dimension, .05*min(bounds)/shape.dimension)]
    iterations = 0
    for mu in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        for _ in range(40):
            iterations += 1
            f, g, h, area = section_objective(x, shape, bounds, mu)
            step = np.linalg.solve(h, -g)
            slope = g @ step
            if -slope < 1e-12:
                break
            alpha = 1.
            for _ in range(40):
                candidate = x+alpha*step
                after = section_objective(candidate, shape, bounds, mu)[0]
                if after <= f + 1e-3*alpha*slope:
                    x = candidate
                    break
                alpha *= .5
            else:
                raise RuntimeError('section line search failed')
        else:
            raise RuntimeError('section iteration cap reached')
    h, _, _, area, _, _ = shape.geometry(x[2:])
    violation = max(shape.normals @ x[:2] + h - bounds)
    if violation > 1e-8:
        raise RuntimeError('section support audit failed')
    return dict(area=float(area), iterations=iterations, max_violation=float(violation), x=x.tolist())


def polygon_area(normals, bounds):
    vertices = []
    for i in range(len(normals)):
        for j in range(i):
            a = normals[[i, j]]
            if abs(np.linalg.det(a)) < 1e-12:
                continue
            v = np.linalg.solve(a, bounds[[i, j]])
            if max(normals @ v - bounds) < 1e-9:
                vertices.append(v)
    v = np.unique(np.round(vertices, 12), axis=0)
    center = np.mean(v, axis=0)
    v = v[np.argsort(np.arctan2(v[:, 1]-center[1], v[:, 0]-center[0]))]
    return .5*abs(np.sum(v[:, 0]*np.roll(v[:, 1], -1) - v[:, 1]*np.roll(v[:, 0], -1)))


def section_experiments():
    n4 = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
    angles = np.arange(8)*math.pi/4
    n8 = np.array([np.cos(angles), np.sin(angles)]).T
    cases = [('square', n4, np.ones(4)), ('rectangle', n4, np.array([3.,3.,.6,.6])),
             ('rotated_rectangle', n4 @ rotation(math.pi/6).T, np.array([3.,3.,.6,.6])),
             ('octagon', n8, np.ones(8)),
             ('cut_corner', np.vstack([n4, [1/2**.5, 1/2**.5]]), np.r_[1.,1.,1.,1.,.75])]
    # Independent finite differences exercise all families before optimization.
    derivative_errors = {}
    for family in ['ellipse', 'superellipse4', 'minkowski2', 'minkowski4']:
        shape = Section(family, n8, .13)
        x = np.r_[.03, -.02, np.full(shape.dimension, .18)]
        _, g, h, _ = section_objective(x, shape, np.ones(8), .02)
        eps = 1e-6
        ng = np.zeros(len(x)); nh = np.zeros_like(h)
        for j in range(len(x)):
            dx = np.eye(len(x))[j]*eps
            a = section_objective(x+dx, shape, np.ones(8), .02)
            b = section_objective(x-dx, shape, np.ones(8), .02)
            ng[j] = (a[0]-b[0])/(2*eps)
            nh[:, j] = (a[1]-b[1])/(2*eps)
        error = max(np.max(abs(g-ng)), np.max(abs(h-nh)))
        if error > 2e-5:
            raise RuntimeError(f'{family} derivatives: {error}')
        derivative_errors[family] = float(error)
    rows = []
    for name, normals, bounds in cases:
        cell_area = polygon_area(normals, bounds)
        for family in ['ellipse', 'superellipse4', 'minkowski2', 'minkowski4']:
            start = time.perf_counter()
            shapes = [Section(family, normals, i*math.pi/12) for i in range(6)]
            setup_ms = (time.perf_counter()-start)*1000
            times = []
            for repeat in range(4):  # one warmup, three measured runs
                start = time.perf_counter()
                candidates = [solve_section(shape, bounds) for shape in shapes]
                elapsed = (time.perf_counter()-start)*1000
                if repeat:
                    times.append(elapsed)
            best = max(candidates, key=lambda r: r['area'])
            rows.append(dict(case=name, family=family, polygon_area=cell_area,
                             fraction=best['area']/cell_area, solve_ms=float(np.median(times)),
                             setup_ms=setup_ms, angle_candidates=6,
                             chosen_angle_rad=candidates.index(best)*math.pi/12, **best))
        print('fixed sections:', name, flush=True)
    # Refine area quadrature independently; this is not polygonal boundary sampling.
    coarse = Section('minkowski4', n8)
    fine = Section('minkowski4', n8, quadrature=8192)
    quadrature_error = float(np.max(abs(coarse.mixed-fine.mixed)))
    if quadrature_error > 1e-9:
        raise RuntimeError('mixed-area quadrature did not converge')
    return dict(derivative_errors=derivative_errors, mixed_area_refinement_error=quadrature_error,
                scope='fixed 2D halfspaces; Python/NumPy solver; not full corridor generation time', rows=rows)


def frame_experiments():
    rows = []
    u = np.linspace(0, 1, 1001)
    w = 10*u**3 - 15*u**4 + 6*u**5
    for degrees in [15, 30, 60, 90, 120]:
        theta = math.radians(degrees)
        # Both frames have first column (1,0,0), with the second column in yz.
        blended = np.sqrt((1-w+w*math.cos(theta))**2 + (w*math.sin(theta))**2)
        orthogonal = np.sqrt(np.cos(w*theta)**2 + np.sin(w*theta)**2)
        if max(abs(orthogonal-1)) > 1e-14:
            raise RuntimeError('orthogonal frame area changed')
        rows.append(dict(angle_degrees=degrees, blended_min_area_ratio=float(min(blended)),
                         orthogonal_min_area_ratio=float(min(orthogonal)),
                         predicted_min_ratio=math.cos(theta/2)))
    # The chosen rotation path has zero first/second jets at both ends, so
    # successive such frame spans join C2. This does not certify a whole tube.
    derivative = pol.polyder(np.array([0.,0.,0.,10.,-15.,6.]))
    assert max(abs(pol.polyval([0., 1.], derivative))) == 0
    assert max(abs(pol.polyval([0., 1.], pol.polyder(derivative)))) == 0
    return dict(scope='equal circular sections; prescribed C2 frame; no obstacle certificate', rows=rows)


def support_max(power):
    """Dense bracketing plus golden-section extrema: a numerical reference."""
    def value(t):
        v = pol.polyval(t, power)
        return v[0] + np.hypot(v[1], v[2])
    grid = np.linspace(0, 1, 513)
    values = value(grid)
    candidates = np.where((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))[0]+1
    result = max(values[0], values[-1])
    golden = (5**.5-1)/2
    for i in candidates:
        lo, hi = grid[i-1], grid[i+1]
        a, b = hi-golden*(hi-lo), lo+golden*(hi-lo)
        va, vb = value(a), value(b)
        for _ in range(48):
            if va < vb:
                lo, a, va = a, b, vb
                b = lo+golden*(hi-lo); vb = value(b)
            else:
                hi, b, vb = b, a, va
                a = hi-golden*(hi-lo); va = value(a)
        result = max(result, va, vb)
    return float(result)


def basis_experiments(basis):
    rng = np.random.default_rng(20261102)
    rows = []
    for degree in [3, 5]:
        bz = np.array(basis[str(degree)]['bezier'])
        mv = np.array(basis[str(degree)]['minvo'])
        inverse = np.linalg.inv(mv)
        minima = [min(pol.polyval(extrema(inverse[:, j]), inverse[:, j])) for j in range(degree+1)]
        sum_target = np.r_[1., np.zeros(degree)]
        sum_error = max(abs(np.sum(inverse, axis=1)-sum_target))
        conversion = mv @ np.linalg.inv(bz)
        gaps, changes = [], []
        elapsed = 0.
        for _ in range(200):
            controls = rng.normal(0., .25, (degree+1, 3))
            controls[:, 1] += 1.  # projected shape norm stays well conditioned
            power = np.linalg.solve(bz, controls)
            actual = support_max(power)
            hb = max(controls[:, 0]+np.linalg.norm(controls[:, 1:], axis=1))
            start = time.perf_counter()
            transformed = conversion @ controls
            hm = max(transformed[:, 0]+np.linalg.norm(transformed[:, 1:], axis=1))
            elapsed += time.perf_counter()-start
            if min(hb-actual, hm-actual) < -2e-7:
                raise RuntimeError('numerical control upper bound failed')
            gaps.append([hb-actual, hm-actual]); changes.append(hm-hb)
        weights = np.r_[np.zeros((degree+1)//2), np.ones((degree+1)//2)]
        mixed_weights = conversion @ weights
        rows.append(dict(degree=degree, tested_directions=200,
                         basis_minimum=float(min(minima)), partition_coefficient_error=float(sum_error),
                         mean_bezier_support_gap=float(np.mean(gaps, axis=0)[0]),
                         mean_minvo_support_gap=float(np.mean(gaps, axis=0)[1]),
                         minvo_tighter=int(sum(x < -1e-9 for x in changes)),
                         minvo_looser=int(sum(x > 1e-9 for x in changes)),
                         conversion_and_support_us=elapsed*1e6/200,
                         endpoint_blend_weights=mixed_weights.tolist(),
                         coupled_controls=int(sum((abs(mixed_weights)>1e-12) & (abs(mixed_weights-1)>1e-12)))))
    return dict(scope='same random polynomial curves, projected SOC support; floating basis checks', rows=rows)


def solve_linear_sections(a, b):
    """Maximize log(r0*r1) over positive radii and fixed linear support rows."""
    x = np.full(2, .2*min(b)/max(1., np.max(np.sum(abs(a), axis=1))))
    iterations = 0
    for mu in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
        for _ in range(50):
            iterations += 1
            gap = b-a@x
            g = -1/x + mu*a.T@(1/gap)
            h = np.diag(1/x**2) + mu*a.T@(a/gap[:, None]**2)
            step = np.linalg.solve(h, -g)
            slope = g@step
            if -slope < 1e-12:
                break
            f = -sum(np.log(x)) - mu*sum(np.log(gap))
            alpha = 1.
            for _ in range(40):
                candidate = x+alpha*step
                slack = b-a@candidate
                if min(candidate) > 0 and min(slack) > 0:
                    after = -sum(np.log(candidate))-mu*sum(np.log(slack))
                    if after <= f+1e-3*alpha*slope:
                        x = candidate; break
                alpha *= .5
            else:
                raise RuntimeError('linear support line search failed')
        else:
            raise RuntimeError('linear support iteration cap reached')
    return x, iterations


def subdivide_controls(c):
    left, right = [c[0]], [c[-1]]
    work = c.copy()
    while len(work) > 1:
        work = .5*(work[:-1]+work[1:])
        left.append(work[0]); right.append(work[-1])
    return np.array(left), np.array(right[::-1])


def exchange_experiments():
    rng = np.random.default_rng(1241)
    rows = []
    for degree in [3, 5]:
        bz = p2b(degree)
        w = np.r_[np.zeros((degree+1)//2), np.ones((degree+1)//2)]
        wp = np.linalg.solve(bz, w)
        for case in range(12):
            c = rng.uniform(-.2, .65, degree+1)
            c[0] = c[-1] = 0.
            cp = np.linalg.solve(bz, c)
            modes = {}
            for depth in [0, 1, 2]:
                pieces = [np.column_stack([c, w])]
                for _ in range(depth):
                    pieces = [child for piece in pieces for child in subdivide_controls(piece)]
                controls = np.vstack(pieces)
                start = time.perf_counter()
                x, iterations = solve_linear_sections(np.column_stack([1-controls[:, 1], controls[:, 1]]),
                                                       1-controls[:, 0])
                modes['bezier_depth'+str(depth)] = dict(product=float(np.prod(x)),
                    solve_ms=(time.perf_counter()-start)*1000, constraints=len(controls), iterations=iterations)
            points = [0., .5, 1.]
            start = time.perf_counter()
            iterations = 0
            for exchange in range(16):
                weight = pol.polyval(points, wp)
                x, count = solve_linear_sections(np.column_stack([1-weight, weight]),
                                                1-pol.polyval(points, cp))
                iterations += count
                radius = wp*(x[1]-x[0]); radius[0] += x[0]
                slack = -cp-radius; slack[0] += 1.
                critical = extrema(slack)
                values = pol.polyval(critical, slack)
                worst = int(np.argmin(values))
                if values[worst] >= -1e-9:
                    break
                points.append(float(critical[worst]))
            else:
                raise RuntimeError('exchange cap reached')
            exchange_ms = (time.perf_counter()-start)*1000
            aa = -cp; aa[0] += 1.
            q = pol.polysub(pol.polymul(aa, aa), pol.polymul(radius, radius))
            q = np.pad(q, (0, 2*degree+1-len(q)))
            certificate = bernstein_nonnegative(p2b(2*degree) @ q, 1e-8)
            if not certificate[0]:
                raise RuntimeError('final squared-support numerical subdivision failed')
            # Independent dense samples complement, but do not replace, the oracle.
            dense = pol.polyval(np.linspace(0, 1, 4097), slack)
            if min(dense) < -2e-9:
                raise RuntimeError('exchange dense cross-check failed')
            modes['exchange'] = dict(product=float(np.prod(x)), solve_ms=exchange_ms,
                                     constraints=len(points), rounds=exchange+1, iterations=iterations,
                                     min_slack=float(min(values)), certificate_leaves=certificate[1],
                                     unresolved=certificate[2])
            rows.append(dict(degree=degree, case=case, methods=modes))
    return dict(scope='two-radius fixed-plane convex slice; SOC norm reduces to positive scalar radius; '
                      'product is a surrogate, not corridor volume; floating oracle', rows=rows)


def hermite_power(p0, p1, v0, v1, a0, a1, h, degree):
    delta = p1-p0
    if degree == 3:
        c = np.array([p0, h*v0, 3*delta-2*h*v0-h*v1, -2*delta+h*v0+h*v1])
    else:
        c = np.array([p0, h*v0, .5*h*h*a0,
                      10*delta-6*h*v0-4*h*v1-1.5*h*h*a0+.5*h*h*a1,
                      -15*delta+8*h*v0+7*h*v1+1.5*h*h*a0-h*h*a1,
                      6*delta-3*h*v0-3*h*v1-.5*h*h*a0+.5*h*h*a1])
    return c/h**np.arange(degree+1)[:, None]


def piece_energy(coeff, h, order):
    # Gaussian quadrature is independent of the reference solver's energy code.
    nodes, weights = np.polynomial.legendre.leggauss(6)
    values = pol.polyval((nodes+1)*h/2, pol.polyder(coeff, order))
    return h/2 * np.sum(weights*np.sum(values*values, axis=0))


def spline_experiments(directory):
    rows = list(csv.DictReader((directory/'spline.csv').open()))
    details = []
    for path in sorted(directory.glob('*_degree*.json')):
        data = json.loads(path.read_text())
        degree = data['degree']
        coeff = np.array(data['coefficients']).reshape(-1, degree+1, 3)
        values = np.array(data['values']); knots = np.array(data['knots'])
        widths = []
        for c, h in zip(coeff, np.diff(knots)):
            for j in [1, 2]:
                normalized = c[:, j]*h**np.arange(degree+1)
                widths.append(min(pol.polyval(extrema(normalized), normalized)))
        velocity = np.vstack([c[1] for c in coeff] +
                             [pol.polyval(knots[-1]-knots[-2], pol.polyder(coeff[-1]))])
        acc = np.vstack([2*c[2] for c in coeff] +
                        [pol.polyval(knots[-1]-knots[-2], pol.polyder(coeff[-1], 2))])
        derivative = 2 if degree == 3 else 3
        total = sum(piece_energy(c, h, derivative) for c, h in zip(coeff, np.diff(knots)))
        stationarity = 0.
        # Independently perturb shared jets and reconstruct the two affected
        # Hermite segments, retaining all values and endpoint boundary data.
        for knot in range(1, len(knots)-1):
            for order in range(1, derivative):
                target = velocity if order == 1 else acc
                for coordinate in range(3):
                    base = target[knot, coordinate]
                    energies = []
                    for sign in [-1, 1]:
                        target[knot, coordinate] = base+sign*1e-5
                        local = 0.
                        for i in [knot-1, knot]:
                            h = knots[i+1]-knots[i]
                            c = hermite_power(values[i], values[i+1], velocity[i], velocity[i+1],
                                              acc[i], acc[i+1], h, degree)
                            local += piece_energy(c, h, derivative)
                        energies.append(local)
                    target[knot, coordinate] = base
                    stationarity = max(stationarity, abs(energies[1]-energies[0])/2e-5)
        if stationarity > 1e-5*max(1., total):
            raise RuntimeError('minimum energy stationarity cross-check failed')
        zero_jet_energy = sum(720*np.sum((values[i+1]-values[i])**2)/h**5
                              for i, h in enumerate(np.diff(knots)))
        details.append(dict(case=path.stem, degree=degree, critical_point_min_halfwidth=float(min(widths)),
                            stationarity_error=float(stationarity), min_derivative_energy=float(total),
                            zero_jet_quintic_jerk_energy=float(zero_jet_energy)))
    return dict(scope='fixed frame, scalar offset and two halfwidth fields; cubic clamps endpoint velocity, '
                      'quintic also clamps acceleration; energies have different feasible sets',
                timing_rows=rows, independent_checks=details)


def merge_voxels(voxels):
    """Disjoint exact x runs, then matching y rectangles, then matching z boxes."""
    def intervals(values):
        values = sorted(values)
        start = end = values[0]
        for value in values[1:]:
            if value == end+1:
                end = value
            else:
                yield start, end+1
                start = end = value
        yield start, end+1
    groups = {}
    for x, y, z in voxels:
        groups.setdefault((int(y), int(z)), []).append(int(x))
    rectangles = {}
    for (y, z), xs in groups.items():
        for lo, hi in intervals(xs):
            rectangles.setdefault((lo, hi, z), []).append(y)
    groups = {}
    for (x0, x1, z), ys in rectangles.items():
        for y0, y1 in intervals(ys):
            groups.setdefault((x0, x1, y0, y1), []).append(z)
    boxes = []
    for (x0, x1, y0, y1), zs in groups.items():
        for z0, z1 in intervals(zs):
            boxes.append((x0, x1, y0, y1, z0, z1))
    return boxes


def voxel_experiments(cloud=None, resolution=.25):
    cube = np.array([(x,y,z) for x in range(16) for y in range(16) for z in range(16)], dtype=int)
    l_shape = np.array([(x,y,z) for x in range(20) for y in range(20) for z in range(6)
                        if x < 4 or y < 4], dtype=int)
    # Deterministic holes must not be filled by the merger.
    perforated = np.array([v for v in cube if not (v[0] in (7,8) and v[1] in (7,8))])
    rows = []
    for name, voxels in [('solid_cube', cube), ('l_shape', l_shape), ('perforated', perforated)]:
        start = time.perf_counter(); boxes = merge_voxels(voxels)
        elapsed = (time.perf_counter()-start)*1000
        expanded = [(x,y,z) for x0,x1,y0,y1,z0,z1 in boxes for x in range(x0,x1)
                    for y in range(y0,y1) for z in range(z0,z1)]
        if len(expanded) != len(set(expanded)) or set(expanded) != set(map(tuple, voxels)):
            raise RuntimeError('voxel union changed')
        rows.append(dict(case=name, voxels=len(voxels), boxes=len(boxes), merge_ms=elapsed,
                         exact_union_crosscheck=True))
    if cloud:
        points = np.loadtxt(cloud, ndmin=2)
        indices = np.rint((points-points[0])/resolution).astype(np.int64)
        error = float(np.max(abs(points-(points[0]+resolution*indices))))
        if error < 1e-5:
            indices = np.unique(indices, axis=0)
            start = time.perf_counter(); boxes = merge_voxels(indices)
            elapsed = (time.perf_counter()-start)*1000
            represented = sum((x1-x0)*(y1-y0)*(z1-z0) for x0,x1,y0,y1,z0,z1 in boxes)
            if represented != len(indices):
                raise RuntimeError('merged voxel count mismatch')
            rows.append(dict(case='saved_map', source=str(cloud), input_points=len(points),
                             voxels=len(indices), boxes=len(boxes), merge_ms=elapsed,
                             represented_voxels=represented, grid_residual_m=error))
        else:
            rows.append(dict(case='saved_map', source=str(cloud), grid_residual_m=error,
                             skipped='input did not satisfy the voxel grid contract'))
    a = resolution/2
    directions = np.array([[1.,0,0], [1.,1,0], [1.,1,1]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return dict(scope='exact occupied-cell union only; no corridor/FIRI model change',
                resolution_m=resolution, sphere_minus_box_support_m=(math.sqrt(3)*a-a*np.sum(abs(directions),axis=1)).tolist(),
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cloud', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [('spline', lambda: spline_experiments(args.spline)),
            ('frames', frame_experiments),
            ('sections', section_experiments),
            ('basis', lambda: basis_experiments(json.loads((args.spline/'basis.json').read_text()))),
            ('exchange', exchange_experiments),
            ('voxels', lambda: voxel_experiments(args.cloud))]
    results = {'seed': 20261102, 'numpy_version': np.__version__,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'limitations': 'Independent mathematical prototypes; no new production safety claim.'}
    for name, job in jobs:
        print('starting', name, flush=True)
        start = time.perf_counter()
        results[name] = job()
        results[name]['wall_ms'] = (time.perf_counter()-start)*1000
        (args.output/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
        print('finished', name, flush=True)


if __name__ == '__main__':
    main()
