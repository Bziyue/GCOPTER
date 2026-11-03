struct ObstacleGroups {
  struct Block {
    Vec3 center, half;
    double min_radius, max_radius;
    unsigned long long alive;
    int begin, end, nearest;
  };
  std::vector<Block> blocks;
  bool uniform_radius = true;
};
inline ObstacleGroups obstacleGroups(const std::vector<int>& ids, const ObstacleIndex& map) {
  ObstacleGroups result;
  result.blocks.reserve((ids.size() + 31) / 32);
  for (int begin = 0; begin < int(ids.size()); begin += 32) {
    const int end = std::min(begin + 32, int(ids.size()));
    Vec3 lo = map.obstacles()[ids[begin]].center, hi = lo;
    double rmin = INFINITY, rmax = 0;
    for (int j = begin; j < end; ++j) {
      const auto& sphere = map.obstacles()[ids[j]];
      lo = lo.cwiseMin(sphere.center);
      hi = hi.cwiseMax(sphere.center);
      rmin = std::min(rmin, sphere.radius);
      rmax = std::max(rmax, sphere.radius);
      result.uniform_radius &= sphere.radius == map.obstacles()[ids.front()].radius;
    }
    result.blocks.push_back({.5*(lo+hi), .5*(hi-lo), rmin, rmax,
                            (1ull << (end-begin))-1, begin, end, -1});
  }
  return result;
}
