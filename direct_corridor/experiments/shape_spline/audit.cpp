// SPDX-License-Identifier: MIT
#include "refit.hpp"
#include "corridor_benchmark_common.hpp"
#include "tube_query.hpp"
int main(int argc, char**argv) {
  try{
    if (argc != 13) {
      throw std::runtime_error(
              "seed map trajectory_root case_name xmin ymin zmin xmax ymax zmax resolution output");
    }
    auto seed = read(argv[1]), points = read(argv[2]);
    Bounds bounds{Vec3(std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7])),
                  Vec3(std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]))};
    double radius = std::sqrt(3) * std::stod(argv[11]) / 2;
    std::vector<Sphere>spheres;
    for (auto p:points){
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
    double diagonal = 0, length = 0;
    for (auto box:expansionBoxes(seed, bounds, 3, .252)){
      diagonal = std::max(diagonal, (box.bounds.upper - box.bounds.lower).norm());
    }
    o.max_radius = .5 * diagonal + .001;
    o.max_offset = diagonal + .001;
    for (size_t i = 1; i < seed.size(); ++i){
      length += (seed[i] - seed[i - 1]).norm();
    }
    auto base = generate(seed, index, bounds, o);
    if (!base){
      throw std::runtime_error(base.message);
    }
    const int pieces = std::max(2, int(std::ceil(length)));
    std::ofstream out(argv[12]);
    out << std::setprecision(12);
    for (const std::string mode:{"baseline", "hermite-free", "cubic-c2"}) {
      Corridor c = base.corridor;
      if (mode != "baseline"){
        c = gcopter::direct::experiment::refitShape(base.corridor, seed, index, bounds, o,
                                                    mode == "hermite-free"?5:3).corridor;
      }
      auto patches = corridor_volume::patches(c);
      tube_query::Query query(c, true);
      std::filesystem::path path = std::filesystem::path(argv[3]) / mode / argv[4] /
                                   "hermite_trajectory.csv";
      std::ifstream in(path);
      if (!in){
        throw std::runtime_error("missing trajectory CSV");
      }
      std::string line;
      std::getline(in, line);
      std::vector<double> times;
      std::vector<Vec3> positions;
      while (std::getline(in, line)){
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double t;
        Vec3 p;
        row >> t >> p.x() >> p.y() >> p.z();
        times.push_back(t);
        positions.push_back(p);
      }
      // Consecutive equal CSV times mark piece boundaries (both sides are exported).
      std::vector<std::pair<size_t, size_t> > segments;
      size_t begin = 0;
      for (size_t i = 1; i < times.size(); ++i){
        if (std::abs(times[i] - times[i - 1]) < 1e-10){
          segments.push_back({begin, i - 1});
          begin = i;
        }
      }
      segments.push_back({begin, times.size() - 1});
      if (int(segments.size()) != pieces){
        throw std::runtime_error("trajectory segment count");
      }
      size_t total = 0, selected_outside = 0, inside_other_root = 0, globally_outside = 0,
             unresolved = 0;
      double maximum = 0;
      bool maximum_inside = false;
      int segment = 0;
      for (auto span:segments) {
        for (size_t i = span.first; i <= span.second; ++i){
          double alpha = double(i - span.first) / (span.second - span.first);
          auto value = query(positions[i], (segment + alpha) / pieces);
          ++total;
          bool outside = value.radial > .005||std::abs(value.plane) > .005||!value.valid;
          if (!outside){
            continue;
          }
          ++selected_outside;
          bool inside = false;
          for (const auto& patch:patches){
            if (patch.contains(positions[i], unresolved)){
              inside = true;
              break;
            }
          }
          inside_other_root += inside;
          globally_outside += !inside;
          double violation = std::max(value.radial, std::abs(value.plane));
          if (violation > maximum){
            maximum = violation;
            maximum_inside = inside;
          }
        }
        ++segment;
      }
      out << "{\"mode\":\"" << mode << "\",\"points\":" << total << ",\"selected_root_outside\":" <<
        selected_outside << ",\"inside_via_another_root\":" << inside_other_root <<
        ",\"outside_entire_union\":" << globally_outside << ",\"unresolved_roots\":" <<
      unresolved <<
        ",\"max_selected_residual_m\":" << maximum << ",\"max_residual_point_inside_union\":" <<
      (maximum_inside?"true":"false") << "}\n";
    }
  }
  catch (const std::exception& e){
    std::cerr << e.what() << '\n';
    return 1;
  }
}
