#!/usr/bin/env python3
"""Serial ablations against the frozen abc0631 generator, without installing them.

Each variant is built in an isolated directory and loaded only by its test and
benchmark subprocess. Build address space: 2 GiB; runtime: 512 MiB. The existing
runner supplies continuous geometry audits and identical FIRI settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = 'abc0631'


def replace(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('Frozen source does not match the intended experiment')
    return text.replace(before, after)


def variant_sources(name, source=None):
    if name == 'group_schur':
        return variant_sources('schur', variant_sources('blocked_cache'))
    if name == 'combined':
        return variant_sources('feasible_late', variant_sources('schur', variant_sources('blocked_cache')))
    paths = ['direct_corridor/src/direct_corridor.cpp',
             'direct_corridor/src/hermite_optimizer.hpp']
    cpp, header = source if source is not None else [
        subprocess.check_output(['git', 'show', f'{BASE}:{p}'], cwd=ROOT, text=True) for p in paths]
    if name == 'lazy_blocks':
        cpp, header = variant_sources('blocked_cache')
        begin = header.index('    struct Candidate { int id; double distance; };')
        end = header.index('    ++metrics.certificate_leaves;\n    if (!failed.empty())', begin)
        direction_begin = header.index('      const Vec3 delta = sphere.center - cm;', begin)
        direction_end = header.index('      ++metrics.obstacle_pairs;', direction_begin)
        helper = (Path(__file__).parent/'lazy_blocks.inc').read_text()
        helper = helper.replace('      // @separator_body@', header[direction_begin:direction_end].rstrip())
        header = replace(header, header[begin:end], helper)
    elif name == 'schur':
        header = replace(header, 'std::vector<H5> diagonal, upper;',
                         'std::vector<H5> diagonal, upper, transfer;')
        header = replace(header, 'factors.resize(count);',
                         'factors.resize(count);\n    transfer.resize(count - 1);')
        header = replace(header, '''              h -= upper[i - 1].transpose() *
                   factors[i - 1].solve(upper[i - 1]);
              rhs[i] -= upper[i - 1].transpose() *
                        factors[i - 1].solve(rhs[i - 1]);''', '''              h -= upper[i - 1].transpose() * transfer[i - 1];
              rhs[i] -= upper[i - 1].transpose() * rhs[i - 1];''')
        header = replace(header, '''          }
          if (factored) break;''', '''            rhs[i] = factors[i].solve(rhs[i]).eval();
            if (i + 1 < count) transfer[i] = factors[i].solve(upper[i]);
          }
          if (factored) break;''')
        header = replace(header, '''          step[i] = factors[i].solve(
              rhs[i] - (i + 1 < count ? V5(upper[i] * step[i + 1])
                                     : V5::Zero()));''', '''          step[i] = rhs[i] - (i + 1 < count
              ? V5(transfer[i] * step[i + 1]) : V5::Zero());''')
    elif name in ('lazy_obstacles', 'blocked_obstacles', 'blocked_cache'):
        begin = header.index('    const double first_radius = map.obstacles()[ids.front()].radius;')
        end = header.index('    ++metrics.certificate_leaves;\n    if (!failed.empty())', begin)
        direction_begin = header.index('      const Vec3 delta = sphere.center - cm;', begin)
        direction_end = header.index('      ++metrics.obstacle_pairs;', direction_begin)
        helper = (Path(__file__).parent/(('blocked_obstacles' if name == 'blocked_cache' else name) + '.inc')).read_text()
        helper = helper.replace('      // @separator_body@', header[direction_begin:direction_end].rstrip())
        header = replace(header, header[begin:end], helper)
        if name == 'blocked_cache':
            groups = (Path(__file__).parent/'obstacle_groups.inc').read_text()
            header = replace(header, 'struct QueryCache {', groups + '\nstruct QueryCache {')
            header = replace(header, '  std::vector<std::vector<int>> ids;',
                             '  std::vector<std::vector<int>> ids;\n  std::vector<ObstacleGroups> groups;')
            header = replace(header, '''      const std::size_t size = scratch.capacity() * sizeof(int);
      if (size <= byte_limit - bytes) {
        bytes += size;
        entry = int(ids.size());
        ids.push_back(std::move(scratch));
        boxes.emplace(key, entry);
      }''', '''      const std::size_t index_bytes = scratch.capacity() * sizeof(int);
      const std::size_t group_bytes = ((scratch.size() + 31) / 32) * sizeof(ObstacleGroups::Block);
      if (index_bytes + group_bytes <= byte_limit - bytes) {
        auto grouped = obstacleGroups(scratch, map);
        const std::size_t size = index_bytes + grouped.blocks.capacity() * sizeof(ObstacleGroups::Block);
        if (size <= byte_limit - bytes) {
          bytes += size;
          entry = int(ids.size());
          ids.push_back(std::move(scratch));
          groups.push_back(std::move(grouped));
          boxes.emplace(key, entry);
        }
      }''')
            start = header.index('    struct Candidate { int id; double distance; };')
            end = header.index('    if (!uniform_radius) {', start)
            header = replace(header, header[start:end], '''    struct Candidate { int id; double distance; };
    using Block = ObstacleGroups::Block;
    constexpr int block_size = 32;
    ObstacleGroups temporary;
    const ObstacleGroups* grouped;
    if (depth == 0 && query_cache.spans[p.span] >= 0)
      grouped = &query_cache.groups[query_cache.spans[p.span]];
    else {
      temporary = obstacleGroups(ids, map);
      grouped = &temporary;
    }
    auto blocks = grouped->blocks;
    const bool uniform_radius = grouped->uniform_radius;
    std::vector<Candidate> remaining;
    remaining.reserve(ids.size());
    for (int id : ids)
      remaining.push_back({id, (metric * (map.obstacles()[id].center - cm)).squaredNorm()});
''')
    elif name in ('feasible_step', 'feasible_late'):
        helper = (Path(__file__).parent/'feasible_step.inc').read_text()
        header = replace(header, '  void solve(std::vector<Node>& nodes, const std::vector<Cut>& cuts,',
                         helper + '\n  void solve(std::vector<Node>& nodes, const std::vector<Cut>& cuts,')
        header = replace(header, '        double alpha = 1;',
                         '        double alpha = feasibleStep(nodes, cuts, active);')
        if name == 'feasible_late':
            header = replace(header, 'struct Optimizer {', 'struct Optimizer {\n  bool use_feasible_step = false;')
            header = replace(header, 'double alpha = feasibleStep(nodes, cuts, active);',
                             'double alpha = use_feasible_step ? feasibleStep(nodes, cuts, active) : 1.;')
            cpp = replace(cpp, '              Optimizer optimizer(o, stop);',
                          '              Optimizer optimizer(o, stop);\n              optimizer.use_feasible_step = iteration > 0;')
    elif name == 'decrement':
        # Match inner accuracy to the current barrier parameter. Final-stage
        # tolerance remains the baseline value; this is an ablation, not a
        # claim of identical iterates or of optimality at an iteration cap.
        header = replace(header, 'if (-slope < 1e-7 * count) break;',
                         'if (-slope < std::max(1e-7, .01 * mu) * count) break;')
    elif name == 'workset':
        header = replace(header, '''    const auto active = irredundant(nodes, cuts);
    ++metrics.optimization_passes;
    solve(nodes, cuts, active, metrics);
    metrics.active_constraints = active.size();''', '''    const auto full = irredundant(nodes, cuts);
    std::vector<int> active;
    std::vector<bool> used(cuts.size(), false);
    for (int id : full) {
      const auto& c = cuts[id];
      if (c.linear || c.h - value(c, nodes) < .3 * std::max(.1, std::abs(c.h))) {
        active.push_back(id);
        used[id] = true;
      }
    }
    for (int pass = 0; pass < 4; ++pass) {
      auto candidate = nodes;  // globally interior restart after rejected solve
      ++metrics.optimization_passes;
      solve(candidate, cuts, active, metrics);
      bool added = false;
      for (int id : full) {
        if (!used[id] && value(cuts[id], candidate) >= cuts[id].h - 1e-8) {
          used[id] = true;
          active.push_back(id);
          added = true;
        }
      }
      if (!added) {
        nodes.swap(candidate);
        metrics.active_constraints = active.size();
        return;
      }
    }
    ++metrics.optimization_passes;
    solve(nodes, cuts, full, metrics);
    metrics.active_constraints = full.size();''')
    elif name in ('certificate_split', 'certificate_adaptive'):
        header = replace(header, '''  void add(const Patch& p, const Vec3& normal, double bound,
           const std::vector<Node>& nodes) {''', '''  void add(const Patch& p, const Vec3& normal, double bound,
           const std::vector<Node>& nodes) {
    if (use_shape_metric) {
      const auto halves = split(p);
      addControls(halves.first, normal, bound, nodes);
      addControls(halves.second, normal, bound, nodes);
    } else {
      addControls(p, normal, bound, nodes);
    }
  }
  void addControls(const Patch& p, const Vec3& normal, double bound,
                   const std::vector<Node>& nodes) {''')
        if name == 'certificate_adaptive':
            header = replace(header, '''    if (use_shape_metric) {
      const auto halves = split(p);
      addControls(halves.first, normal, bound, nodes);
      addControls(halves.second, normal, bound, nodes);
    } else {
      addControls(p, normal, bound, nodes);
    }''', '''    if (use_shape_metric) {
      auto support = [&](const Patch& part) {
        std::array<Vec3, 6> c;
        std::array<Mat32, 6> m;
        controls(part, nodes, c, m);
        double upper = -INFINITY;
        for (int j = 0; j < 6; ++j)
          upper = std::max(upper, normal.dot(c[j]) + (m[j].transpose() * normal).norm());
        return upper;
      };
      const double upper = support(p);
      if (bound - upper < .05) {
        const auto halves = split(p);
        const double tighter = std::max(support(halves.first), support(halves.second));
        if (upper - tighter > .005) {
          addControls(halves.first, normal, bound, nodes);
          addControls(halves.second, normal, bound, nodes);
          return;
        }
      }
    }
    addControls(p, normal, bound, nodes);''')
    elif name == 'adaptive_nodes':
        cpp = replace(cpp, '''        for (int k = 1; k < int(count); ++k)
          samples.push_back({k * total / count, false});''', '''        // Keep all seed vertices. Coarsen away from turning vertices;
        // the original continuous certificates still accept
        // or reject the resulting reference and may trigger refinement.
        std::vector<double> turns;
        for (std::size_t j = 1; j + 1 < clean.size(); ++j) {
          samples.push_back({arc[j], true});
          if ((clean[j] - clean[j - 1]).normalized().dot(
                  (clean[j + 1] - clean[j]).normalized()) < std::cos(.2))
            turns.push_back(arc[j]);
        }
        for (int k = 1; k < int(count); ++k) {
          const double s = k * total / count;
          bool near_turn = false;
          for (double turn : turns) near_turn |= std::abs(turn - s) < 2 * spacing;
          if (k % 2 || near_turn) samples.push_back({s, false});
        }''')
    elif name != 'baseline':
        raise ValueError(name)
    return cpp, header


def test_variant(dest, env):
    # Tests include the private optimizer. Recompile against this variant too:
    # loading a changed private layout into old inline test code violates ODR.
    source = subprocess.check_output(
        ['git', 'show', f'{BASE}:direct_corridor/tests/test_corridor.cpp'],
        cwd=ROOT, text=True).replace('../src/hermite_optimizer.hpp', 'hermite_optimizer.hpp')
    (dest/'test_corridor.cpp').write_text(source)
    cmd = ['prlimit', '--as=2147483648', '--cpu=120', 'c++', '-std=c++17', '-O3',
           '-DNDEBUG', '-I' + str(ROOT/'direct_corridor/include'), '-I/usr/include/eigen3',
           str(dest/'test_corridor.cpp'), '-L' + str(dest/'lib'), '-lgcopter_direct_corridor',
           '-Wl,-rpath,' + str(dest/'lib'), '-o', str(dest/'tests')]
    with (dest/'test_build.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=150)
    with (dest/'tests.log').open('w') as log:
        result = subprocess.run(['timeout', '-k', '2', '45', 'prlimit', '--as=536870912',
                                 '--cpu=30', str(dest/'tests')],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variants', nargs='+', choices=['baseline', 'schur', 'decrement',
                                                'workset', 'adaptive_nodes', 'certificate_split',
                                                'certificate_adaptive', 'feasible_step', 'lazy_obstacles',
                                                'blocked_obstacles', 'blocked_cache', 'feasible_late', 'lazy_blocks',
                                                'combined', 'group_schur'])
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output', type=Path,
                   default=Path('.scratch/parameterization_round4/pipeline'))
    p.add_argument('--cases', type=int, default=3)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--samples', type=int, default=65536)
    args = p.parse_args()
    os.chdir(ROOT)
    inputs = json.loads(args.inputs.read_text())[:args.cases]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for name in args.variants:
        dest = args.output.resolve() / name
        dest.mkdir(parents=True, exist_ok=False)
        lib = dest / 'lib'
        lib.mkdir()
        cpp, header = variant_sources(name)
        (dest / 'direct_corridor.cpp').write_text(cpp)
        (dest / 'hermite_optimizer.hpp').write_text(header)
        metadata = {'base': subprocess.check_output(['git', 'rev-parse', BASE], text=True).strip(),
                    'variant': name, 'sha256': {k: hashlib.sha256(v.encode()).hexdigest()
                    for k, v in [('cpp', cpp), ('header', header)]}}
        (dest / 'sources.json').write_text(json.dumps(metadata, indent=2) + '\n')
        cmd = ['prlimit', '--as=2147483648', '--cpu=120', 'c++', '-std=c++17',
               '-O3', '-DNDEBUG', '-fPIC', '-shared', '-Wall', '-Wextra',
               '-I' + str(ROOT/'direct_corridor/include'), '-I/usr/include/eigen3',
               str(dest/'direct_corridor.cpp'), '-o', str(lib/'libgcopter_direct_corridor.so')]
        print(f'{name}: serial build', flush=True)
        with (dest/'build.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=150)
        run_env = dict(env, LD_LIBRARY_PATH=str(lib) + ':' + env.get('LD_LIBRARY_PATH', ''))
        code = test_variant(dest, run_env)
        metadata['test_exit_code'] = code
        # Failed experimental tests are evidence, not a reason to silently hide
        # a trial. Benchmarks perform their own audits and record any failure.
        cmd = ['python3', 'gcopter/tools/run_firi_comparison.py', '--output', str(dest/'results'),
               '--policy', 'matched-box', '--range', '3', '--model', 'solid',
               '--repeats', str(args.repeats), '--samples', str(args.samples)]
        for request in inputs:
            cmd += ['--input', request]
        print(f'{name}: tests exit {code}; serial matched-input comparison', flush=True)
        with (dest/'benchmark.log').open('w') as log:
            run = subprocess.run(cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        metadata['benchmark_exit_code'] = run.returncode
        (dest/'sources.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print((dest/'benchmark.log').read_text(), flush=True)


if __name__ == '__main__':
    main()
