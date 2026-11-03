// SPDX-License-Identifier: MIT
#include "refit.hpp"
#include "corridor_benchmark_common.hpp"
using gcopter::direct::experiment::ShapeFitResult;
struct OtherPatch : corridor_volume::Patch {
  bool valid = true, active = true;
  std::size_t* unresolved;
  bool contains(const Vec3& p)const {
    return active&&corridor_volume::Patch::contains(p, *unresolved);
  }
};
static void exportCorridor(const Corridor& c, const std::filesystem::path& path) {
  std::ofstream f(path);
  f << std::setprecision(17) << "u,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
  for (int k = 0; k <= 1200; ++k){
    auto s = c.section(k / 1200.);
    f << k / 1200. << ',' << s.center.x() << ',' << s.center.y() << ',' << s.center.z();
    for (int j = 0; j < 6; ++j){
      f << ',' << s.shape.data()[j];
    }
    f << '\n';
  }
}
static double audit(const Corridor& c, const ObstacleIndex& map, const Bounds& bounds, double pad) {
  double violation = 0;
  std::vector<int> ids;
  for (size_t i = 0; i < c.pieces(); ++i){
    for (int k = 0; k <= 8; ++k){
      auto s = c.section(c.knots()[i] + (c.knots()[i + 1] - c.knots()[i]) * k / 8.);
      for (int j = 0; j < 24; ++j){
        double angle = j * 2 * std::acos(-1.) / 24;
        Vec3 p = s.point({std::cos(angle), std::sin(angle)});
        violation = std::max({violation, (bounds.lower.array() + pad - p.array()).maxCoeff(),
                              (p.array() + pad - bounds.upper.array()).maxCoeff()});
        map.query(p - Vec3::Constant(pad), p + Vec3::Constant(pad), ids);
        for (int id:ids){
          auto& o = map.obstacles()[id];
          violation = std::max(violation, o.radius + pad - (p - o.center).norm());
        }
      }
    }
  }
  return violation;
}
int main(int argc, char**argv) {
  try{
    if (argc != 13) {
      throw std::runtime_error(
              "seed map output xmin ymin zmin xmax ymax zmax resolution repeats samples_per_shift");
    }
    auto seed = read(argv[1]), points = read(argv[2]);
    Bounds bounds{Vec3(std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6])),
                  Vec3(std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9]))};
    double radius = std::sqrt(3) * std::stod(argv[10]) / 2;
    int repeats = std::stoi(argv[11]), samples = std::stoi(argv[12]);
    if (repeats < 1){
      throw std::runtime_error("invalid repeats");
    }
    std::filesystem::path dir = argv[3];
    std::filesystem::create_directories(dir);
    std::vector<Sphere> spheres;
    for (const auto& p:points){
      spheres.push_back({p, radius});
    }
    ObstacleIndex index(std::move(spheres));
    Options o;
    o.control_spacing = .4;
    o.min_radius = .03;
    o.budget_ms = 10000;
    o.robot_radius = .2;
    o.safety_margin = .05;
    o.local_range = 3;
    o.local_progress = 7;
    double diagonal = 0;
    for (auto box:expansionBoxes(seed, bounds, 3, .252)){
      diagonal = std::max(diagonal, (box.bounds.upper - box.bounds.lower).norm());
    }
    o.max_radius = .5 * diagonal + .001;
    o.max_offset = diagonal + .001;
    std::ofstream output(dir / "generation.jsonl");
    output << std::setprecision(12);
    Result base;
    std::array<ShapeFitResult, 2> variants;
    for (int repeat = -1; repeat < repeats; ++repeat) {
      base = generate(seed, index, bounds, o);
      if (!base){
        throw std::runtime_error(base.message);
      }
      output << "{\"mode\":\"baseline\",\"repeat\":" << repeat << ",\"generation_ms\":" <<
        base.metrics.total_ms << ",\"nodes\":" << base.metrics.nodes << "}\n";
      for (int j = 0; j < 2; ++j) {
        int slot = (j + (repeat + 2) % 2) % 2;
        int degree = slot == 0?5:3;
        auto result = gcopter::direct::experiment::refitShape(base.corridor, seed, index, bounds, o,
                                                              degree, repeat == -1);
        output << "{\"mode\":\"" << (slot ==
                                     0?"hermite-free":"cubic-c2") << "\",\"repeat\":" << repeat <<
          ",\"generation_ms\":" << base.metrics.total_ms + result.metrics.total_ms <<
          ",\"base_generation_ms\":" << base.metrics.total_ms;
        gcopter::direct::experiment::writeShapeMetrics(output, result.metrics);
        output << "}\n";
        variants[slot] = std::move(result);
      }
      output.flush();
    }
    exportCorridor(base.corridor, dir / "baseline_sections.csv");
    std::ofstream vol(dir / "volume.jsonl");
    vol << std::setprecision(12);
    const auto base_patches = corridor_volume::patches(base.corridor);
    double baseline_audit = audit(base.corridor, index, bounds, .25);
    for (int slot = 0; slot < 2; ++slot) {
      const char* name = slot == 0?"hermite-free":"cubic-c2";
      const auto& c = variants[slot].corridor;
      exportCorridor(c, dir / (std::string(name) + "_sections.csv"));
      std::size_t unresolved = 0;
      std::vector<OtherPatch> other;
      for (const auto& p:corridor_volume::patches(c)){
        OtherPatch q;
        static_cast<corridor_volume::Patch&>(q) = p;
        q.unresolved = &unresolved;
        other.push_back(q);
      }
      // Both comparisons use the union of all three bounding grids and identical QMC points.
      for (const auto& variant:variants) {
        for (const auto& patch:corridor_volume::patches(variant.corridor)){
          OtherPatch q;
          static_cast<corridor_volume::Patch&>(q) = patch;
          q.active = false;
          q.unresolved = &unresolved;
          other.push_back(q);
        }
      }
      auto measure = corridor_volume::measure(base_patches, other, seed, samples);
      double penetration = audit(c, index, bounds, .25);
      vol << "{\"mode\":\"" << name << "\",\"baseline_volume_m3\":" << measure.volume[0].mean <<
        ",\"baseline_ci95_m3\":" << measure.volume[0].ci95 << ",\"volume_m3\":" <<
        measure.volume[1].mean << ",\"ci95_m3\":" << measure.volume[1].ci95 <<
        ",\"paired_delta_m3\":" << measure.paired_difference.mean << ",\"paired_ci95_m3\":" <<
        measure.paired_difference.ci95 << ",\"samples\":" << measure.samples <<
        ",\"unresolved_roots\":" << measure.unresolved + unresolved <<
        ",\"baseline_max_penetration_m\":" << baseline_audit << ",\"max_penetration_m\":" <<
      penetration << "}\n";
      vol.flush();
      if (measure.unresolved + unresolved||penetration > 1e-6||baseline_audit > 1e-6){
        throw std::runtime_error("independent geometry audit failed");
      }
      std::cout << name << " volume=" << measure.volume[1].mean << " vs " <<
        measure.volume[0].mean << '\n';
    }
  }
  catch (const std::exception& e){
    std::cerr << e.what() << '\n';
    return 1;
  }
}
