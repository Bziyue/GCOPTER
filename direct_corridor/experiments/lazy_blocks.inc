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
    const double inverse_scale = 1 / min_scale;
    const Eigen::Matrix3d absolute_metric = metric.cwiseAbs();
    std::vector<double> distances(ids.size()), priority(blocks.size());
    std::vector<unsigned char> evaluated(blocks.size(), 0);
    std::vector<int> heap(blocks.size());
    std::iota(heap.begin(), heap.end(), 0);
    for (int i = 0; i < int(blocks.size()); ++i) {
      const auto& b = blocks[i];
      const Vec3 lower = ((metric * (b.center - cm)).cwiseAbs() -
                         absolute_metric * b.half).cwiseMax(0.);
      priority[i] = lower.squaredNorm();
      if (!uniform_radius)
        priority[i] = std::sqrt(priority[i]) - (b.max_radius + pad) * inverse_scale;
    }
    auto better = [&](int a, int b) {
      return b < 0 || distances[a] < distances[b] ||
             (distances[a] == distances[b] && ids[a] < ids[b]);
    };
    auto nearest = [&](int i) {
      auto& b = blocks[i];
      b.nearest = -1;
      for (auto bits = b.alive; bits; bits &= bits - 1) {
        const int j = b.begin + __builtin_ctzll(bits);
        if (!evaluated[i]) {
          const auto& sphere = map.obstacles()[ids[j]];
          distances[j] = (metric * (sphere.center - cm)).squaredNorm();
          if (!uniform_radius)
            distances[j] = std::sqrt(distances[j]) - (sphere.radius + pad) * inverse_scale;
        }
        if (better(j, b.nearest)) b.nearest = j;
      }
      evaluated[i] = 1;
      priority[i] = b.nearest >= 0 ? distances[b.nearest] : INFINITY;
    };
    auto compare = [&](int a, int b) {
      return priority[a] > priority[b] || (priority[a] == priority[b] && a > b);
    };
    std::make_heap(heap.begin(), heap.end(), compare);
    std::vector<int> failed;
    unsigned processed = 0;
    while (!heap.empty()) {
      if ((++processed & 31u) == 0) stop();
      std::pop_heap(heap.begin(), heap.end(), compare);
      const int selected_id = heap.back();
      heap.pop_back();
      auto& selected = blocks[selected_id];
      if (!evaluated[selected_id]) {
        nearest(selected_id);
        heap.push_back(selected_id);
        std::push_heap(heap.begin(), heap.end(), compare);
        continue;
      }
      const int next = selected.nearest;
      selected.alive &= ~(1ull << (next - selected.begin));
      nearest(selected_id);
      const int id = ids[next];
      const auto& sphere = map.obstacles()[id];
      // @separator_body@
      ++metrics.obstacle_pairs;
      if (separated) {
        add(p, n, h, nodes);
        const Vec3 absolute = n.cwiseAbs();
        for (int i = 0; i < int(blocks.size()); ++i) {
          auto& b = blocks[i];
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
              const auto& other = map.obstacles()[ids[j]];
              ++metrics.obstacle_pairs;
              if (n.dot(other.center) - other.radius - pad >= h)
                b.alive &= ~(1ull << (j - b.begin));
              else if (evaluated[i] && better(j, b.nearest)) b.nearest = j;
            }
            if (evaluated[i]) priority[i] = b.nearest >= 0 ? distances[b.nearest] : INFINITY;
          }
        }
      } else {
        failed.push_back(id);
      }
      heap.clear();
      for (int i = 0; i < int(blocks.size()); ++i)
        if (blocks[i].alive) heap.push_back(i);
      std::make_heap(heap.begin(), heap.end(), compare);
    }
