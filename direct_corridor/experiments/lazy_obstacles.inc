    // Spatially consecutive BVH query results form conservative sphere boxes.
    // A heap visits blocks lazily. An accepted plane can exclude a complete
    // box without evaluating each sphere's metric or repeatedly filtering it.
    struct Block {
      Vec3 center, half;
      double radius;
      int begin, end;
    };
    struct Item {
      double distance;
      int id;
      bool block;
    };
    struct Plane {
      Vec3 n, absolute;
      double h;
    };
    std::vector<Block> blocks;
    bool uniform_radius = true;
    const double first_radius = map.obstacles()[ids.front()].radius;
    constexpr int block_size = 32;
    blocks.reserve((ids.size() + block_size - 1) / block_size);
    for (int begin = 0; begin < int(ids.size()); begin += block_size) {
      const int end = std::min(begin + block_size, int(ids.size()));
      Vec3 lo = map.obstacles()[ids[begin]].center, hi = lo;
      double radius = 0;
      for (int j = begin; j < end; ++j) {
        const auto& sphere = map.obstacles()[ids[j]];
        lo = lo.cwiseMin(sphere.center);
        hi = hi.cwiseMax(sphere.center);
        radius = std::max(radius, sphere.radius);
        uniform_radius &= sphere.radius == first_radius;
      }
      blocks.push_back({.5 * (lo + hi), .5 * (hi - lo), radius, begin, end});
    }
    const double inverse_scale = 1 / min_scale;
    const Eigen::Matrix3d absolute_metric = metric.cwiseAbs();
    std::vector<Item> heap;
    heap.reserve(blocks.size() + block_size * 8);
    for (int i = 0; i < int(blocks.size()); ++i) {
      const auto& b = blocks[i];
      const Vec3 lower = ((metric * (b.center - cm)).cwiseAbs() -
                         absolute_metric * b.half).cwiseMax(0.);
      double distance = lower.squaredNorm();
      if (!uniform_radius)
        distance = std::sqrt(distance) - (b.radius + pad) * inverse_scale;
      heap.push_back({distance, i, true});
    }
    auto compare = [](const Item& a, const Item& b) {
      if (a.distance != b.distance) return a.distance > b.distance;
      if (a.block != b.block) return !a.block;
      return a.id > b.id;
    };
    std::make_heap(heap.begin(), heap.end(), compare);
    std::vector<Plane> planes;
    auto excluded = [&](const Sphere& sphere) {
      for (const auto& plane : planes) {
        ++metrics.obstacle_pairs;
        if (plane.n.dot(sphere.center) - sphere.radius - pad >= plane.h) return true;
      }
      return false;
    };
    std::vector<int> failed;
    unsigned processed = 0;
    while (!heap.empty()) {
      if ((++processed & 31u) == 0) stop();
      std::pop_heap(heap.begin(), heap.end(), compare);
      const Item item = heap.back();
      heap.pop_back();
      if (item.block) {
        const auto& b = blocks[item.id];
        bool hidden = false;
        for (const auto& plane : planes) {
          const double lower = plane.n.dot(b.center) - plane.absolute.dot(b.half) - b.radius - pad;
          if (lower >= plane.h + 1e-10) { hidden = true; break; }
        }
        if (hidden) continue;
        for (int j = b.begin; j < b.end; ++j) {
          const int id = ids[j];
          const auto& sphere = map.obstacles()[id];
          if (excluded(sphere)) continue;
          double distance = (metric * (sphere.center - cm)).squaredNorm();
          if (!uniform_radius)
            distance = std::sqrt(distance) - (sphere.radius + pad) * inverse_scale;
          heap.push_back({distance, id, false});
          std::push_heap(heap.begin(), heap.end(), compare);
        }
        continue;
      }
      const int id = item.id;
      const auto& sphere = map.obstacles()[id];
      if (excluded(sphere)) continue;
      // @separator_body@
      ++metrics.obstacle_pairs;
      if (separated) {
        add(p, n, h, nodes);
        planes.push_back({n, n.cwiseAbs(), h});
      } else {
        failed.push_back(id);
      }
    }
