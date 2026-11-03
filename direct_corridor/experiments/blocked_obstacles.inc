    // Keep spatial groups intact while planes remove individual spheres.
    // Bounds can discard/retain a whole group; only straddling groups need
    // per-sphere predicates. Per-group minima avoid a heap and array compaction.
    struct Candidate { int id; double distance; };
    struct Block {
      Vec3 center, half;
      double min_radius, max_radius;
      unsigned long long alive;
      int begin, end, nearest;
    };
    constexpr int block_size = 32;
    std::vector<Candidate> remaining;
    remaining.reserve(ids.size());
    std::vector<Block> blocks;
    blocks.reserve((ids.size() + block_size - 1) / block_size);
    bool uniform_radius = true;
    const double first_radius = map.obstacles()[ids.front()].radius;
    for (int begin = 0; begin < int(ids.size()); begin += block_size) {
      const int end = std::min(begin + block_size, int(ids.size()));
      Vec3 lo = map.obstacles()[ids[begin]].center, hi = lo;
      double rmin = INFINITY, rmax = 0;
      for (int j = begin; j < end; ++j) {
        const auto& sphere = map.obstacles()[ids[j]];
        lo = lo.cwiseMin(sphere.center);
        hi = hi.cwiseMax(sphere.center);
        rmin = std::min(rmin, sphere.radius);
        rmax = std::max(rmax, sphere.radius);
        uniform_radius &= sphere.radius == first_radius;
        remaining.push_back({ids[j], (metric * (sphere.center - cm)).squaredNorm()});
      }
      blocks.push_back({.5 * (lo + hi), .5 * (hi - lo), rmin, rmax,
                        (1ull << (end - begin)) - 1, begin, end, -1});
    }
    if (!uniform_radius) {
      const double inverse_scale = 1 / min_scale;
      for (auto& candidate : remaining)
        candidate.distance = std::sqrt(candidate.distance) -
            (map.obstacles()[candidate.id].radius + pad) * inverse_scale;
    }
    auto better = [&](int a, int b) {
      return b < 0 || remaining[a].distance < remaining[b].distance ||
          (remaining[a].distance == remaining[b].distance && remaining[a].id < remaining[b].id);
    };
    auto nearest = [&](Block& b) {
      b.nearest = -1;
      for (auto bits = b.alive; bits; bits &= bits - 1) {
        const int j = b.begin + __builtin_ctzll(bits);
        if (better(j, b.nearest)) b.nearest = j;
      }
    };
    int next = -1;
    for (auto& b : blocks) {
      nearest(b);
      if (better(b.nearest, next)) next = b.nearest;
    }
    std::vector<int> failed;
    unsigned processed = 0;
    while (next >= 0) {
      if ((++processed & 31u) == 0) stop();
      auto& selected = blocks[next / block_size];
      selected.alive &= ~(1ull << (next - selected.begin));
      nearest(selected);
      const int id = remaining[next].id;
      const auto& sphere = map.obstacles()[id];
      // @separator_body@
      ++metrics.obstacle_pairs;
      next = -1;
      if (separated) {
        add(p, n, h, nodes);
        const Vec3 absolute = n.cwiseAbs();
        for (auto& b : blocks) {
          if (!b.alive) continue;
          const double projection = n.dot(b.center), extent = absolute.dot(b.half);
          if (projection - extent - b.max_radius - pad >= h + 1e-10) {
            b.alive = 0;
            b.nearest = -1;
            continue;
          }
          if (projection + extent - b.min_radius - pad >= h - 1e-10) {
            b.nearest = -1;
            for (auto bits = b.alive; bits; bits &= bits - 1) {
              const int j = b.begin + __builtin_ctzll(bits);
              const auto& other = map.obstacles()[remaining[j].id];
              ++metrics.obstacle_pairs;
              if (n.dot(other.center) - other.radius - pad >= h)
                b.alive &= ~(1ull << (j - b.begin));
              else if (better(j, b.nearest)) b.nearest = j;
            }
          }
          if (b.nearest >= 0 && better(b.nearest, next)) next = b.nearest;
        }
      } else {
        failed.push_back(id);
        for (const auto& b : blocks)
          if (b.nearest >= 0 && better(b.nearest, next)) next = b.nearest;
      }
    }
