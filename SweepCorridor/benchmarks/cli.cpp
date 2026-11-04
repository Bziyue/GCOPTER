// SPDX-License-Identifier: MIT
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "SweepCorridor.hpp"
using namespace SweepCorridor;
int main(int argc, char** argv) {
  try {
    if (argc < 12 || argc > 16) {
      std::cerr << "Usage: sweep_cli seed.xyz obstacles.xyz "
                   "obstacle_radius robot_radius xmin ymin zmin xmax ymax zmax "
                   "output_dir [safety_margin [control_spacing [budget_ms "
                   "[min_radius]]]]\n";
      return 2;
    }
    auto read = [](const char* path) {
      std::ifstream f(path);
      if (!f) throw std::runtime_error("cannot open input");
      std::vector<Vec3> p;
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        Vec3 v;
        std::string extra;
        if (!(row >> v.x() >> v.y() >> v.z()) || (row >> extra) ||
            !v.allFinite())
          throw std::runtime_error("expected finite xyz rows");
        p.push_back(v);
      }
      return p;
    };
    auto seed = read(argv[1]), cloud = read(argv[2]);
    std::vector<Sphere> spheres;
    for (auto p : cloud) spheres.push_back({p, std::stod(argv[3])});
    ObstacleIndex map(std::move(spheres));
    Options options;
    options.robot_radius = std::stod(argv[4]);
    if (argc > 12) options.safety_margin = std::stod(argv[12]);
    if (argc > 13) options.control_spacing = std::stod(argv[13]);
    if (argc > 14) options.budget_ms = std::stod(argv[14]);
    if (argc > 15) options.min_radius = std::stod(argv[15]);
    Bounds bounds{
        Vec3(std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7])),
        Vec3(std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]))};
    const auto result = SweepCorridor::Generator(options).generate(seed, map, bounds);
    std::cout << statusName(result.status) << ": " << result.message
              << "; ms=" << result.metrics.total_ms
              << " controls=" << result.metrics.controls
              << " constraints=" << result.metrics.constraints << '\n';
    if (!result) return 1;
    std::filesystem::create_directories(argv[11]);
    const auto dir = std::filesystem::path(argv[11]);
    std::ofstream coefficients(dir / "controls.csv"),
        rings(dir / "surface.csv"), metrics(dir / "metrics.json");
    coefficients << std::setprecision(17)
                 << "u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
    for (std::size_t k = 0; k < result.corridor.controls().size(); ++k) {
      const Vec3 c = result.corridor.controls()[k].head<3>();
      const Mat32 m = Eigen::Map<const Mat32>(result.corridor.controls()[k].data()+3);
      coefficients << result.corridor.knots()[k / 4] << ','
                   << result.corridor.knots()[k / 4 + 1] << ',' << c.x() << ','
                   << c.y() << ',' << c.z();
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 3; ++i) coefficients << ',' << m(i, j);
      coefficients << '\n';
    }
    rings << std::setprecision(17) << "u,angle,x,y,z\n";
    for (int k = 0; k <= 400; ++k) {
      double u = k / 400.;
      auto s = result.corridor.section(u);
      for (int j = 0; j < 64; ++j) {
        double a = j * 2 * std::acos(-1.) / 64;
        Vec3 p = s.point({std::cos(a), std::sin(a)});
        rings << u << ',' << a << ',' << p.x() << ',' << p.y() << ',' << p.z()
              << '\n';
      }
    }
    metrics << "{\"status\":\"SUCCESS\",\"total_ms\":"
            << result.metrics.total_ms
            << ",\"representation\":\"native_cubic_v3\""
            << ",\"nodes\":" << result.metrics.controls
            << ",\"newton_steps\":" << result.metrics.iterations
            << ",\"controls\":" << result.metrics.controls
            << ",\"constraints\":" << result.metrics.constraints
            << ",\"refinements\":" << result.metrics.refinements << "}\n";
    if (!coefficients || !rings || !metrics)
      throw std::runtime_error("output write failed");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
