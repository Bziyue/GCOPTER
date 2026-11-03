// SPDX-License-Identifier: MIT
#include "corridor_benchmark_common.hpp"
int main(int argc, char** argv) {
  try {
    if (argc < 14 || argc > 16)
      throw std::runtime_error(
          "usage: benchmark seed.xyz map.xyz output_dir points|solid xmin ymin "
          "zmin xmax ymax zmax resolution repeats samples_per_shift "
          "[legacy|matched-box [range]]");
    const auto seed = read(argv[1]), points = read(argv[2]);
    const std::filesystem::path dir = argv[3];
    const std::string mode = argv[4];
    if (mode != "points" && mode != "solid")
      throw std::runtime_error("invalid model");
    const bool solid = mode == "solid";
    const std::string policy = argc > 14 ? argv[14] : "legacy";
    const double range = argc > 15 ? std::stod(argv[15]) : 3.;
    if ((policy != "legacy" && policy != "matched-box") ||
        !std::isfinite(range) || range <= 0)
      throw std::runtime_error("invalid expansion policy/range");
    Bounds bounds{
        Vec3(std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7])),
        Vec3(std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]))};
    const double resolution = std::stod(argv[11]),
                 obstacle_radius = solid ? std::sqrt(3) * resolution / 2 : 0;
    const int repeats = std::stoi(argv[12]), samples = std::stoi(argv[13]);
    if (seed.size() < 2 || points.empty() || repeats < 1 || samples < 1)
      throw std::runtime_error("invalid benchmark input");
    Options options;
    options.control_spacing = .4;
    options.min_radius = .03;
    options.budget_ms = 10000;
    options.robot_radius = solid ? .2 : 0;
    options.safety_margin = solid ? .05 : 0;
    options.certificate_margin = solid ? .002 : 1e-8;
    const auto expansion =
        expansionBoxes(seed, bounds, range,
                       options.robot_radius + options.safety_margin +
                           options.certificate_margin);
    if (policy == "matched-box") {
      double diagonal = 0;
      for (const auto& box : expansion)
        diagonal =
            std::max(diagonal, (box.bounds.upper - box.bounds.lower).norm());
      // These algebraic bounds cannot bind before the shared local boxes:
      // an ellipse's diameter is no larger than its containing box diagonal;
      // a center's displacement from a seed point in that box is <= diagonal.
      options.max_radius = .5 * diagonal + .001;
      options.max_offset = diagonal + .001;
      options.local_range = range;
      options.local_progress = 7;
    }
    std::vector<Sphere> spheres;
    for (const auto& p : points) spheres.push_back({p, obstacle_radius});
    auto timer = Clock::now();
    ObstacleIndex map(std::move(spheres));
    const double index_ms = ms(timer);
    Result hermite;
    FiriResult firi;
    std::vector<double> ht, ft, ct, et, st;
    for (int rep = -1; rep < repeats; ++rep) {
      auto h = [&] { hermite = generate(seed, map, bounds, options); };
      auto f = [&] {
        firi = generateFiri(seed, points, bounds, obstacle_radius,
                            options.robot_radius + options.safety_margin, solid,
                            range);
      };
      if (rep % 2 == 0) {
        h();
        f();
      } else {
        f();
        h();
      }
      if (!hermite)
        throw std::runtime_error(std::string("Hermite: ") +
                                 statusName(hermite.status) + " " +
                                 hermite.message);
      if (rep >= 0) {
        ht.push_back(hermite.metrics.total_ms);
        ft.push_back(firi.total_ms);
        ct.push_back(firi.cover_ms);
        et.push_back(firi.erosion_ms);
        st.push_back(firi.shortcut_ms);
      }
    }
    double max_axis = 0, box_violation = 0;
    for (int k = 0; k <= 1000; ++k) {
      const auto section = hermite.corridor.section(k / 1000.);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(
          section.shape.transpose() * section.shape);
      max_axis = std::max(max_axis, std::sqrt(eig.eigenvalues()[1]));
      if (policy == "matched-box") {
        const double arc = k / 1000. * expansion.back().end;
        auto at = std::upper_bound(
            expansion.begin(), expansion.end(), arc,
            [](double s, const ExpansionBox& box) { return s < box.end; });
        if (at == expansion.end()) --at;
        const Vec3 extent = section.shape.rowwise().norm();
        box_violation =
            std::max({box_violation,
                      (section.center + extent - at->bounds.upper).maxCoeff(),
                      (at->bounds.lower - section.center + extent).maxCoeff()});
      }
    }
    if (box_violation > 1e-7)
      throw std::runtime_error("independent local expansion audit failed");
    bool hermite_sample_safe = true;
    std::size_t hermite_audit_samples = 0;
    if (solid) {
      const double pad = options.robot_radius + options.safety_margin;
      std::vector<int> ids;
      for (int k = 0; k <= 256; ++k) {
        const auto section = hermite.corridor.section(k / 256.);
        for (double radius : {0., .5, 1.})
          for (int j = 0; j < 32; ++j) {
            const double theta = j * 2 * std::acos(-1.) / 32;
            const Vec3 p = section.point(
                radius * Eigen::Vector2d(std::cos(theta), std::sin(theta)));
            ++hermite_audit_samples;
            map.query(p - Vec3::Constant(pad), p + Vec3::Constant(pad), ids);
            for (int id : ids)
              if ((p - map.obstacles()[id].center).norm() <
                  map.obstacles()[id].radius + pad - 1e-7)
                hermite_sample_safe = false;
            if ((p.array() < bounds.lower.array() + pad - 1e-7).any() ||
                (p.array() > bounds.upper.array() - pad + 1e-7).any())
              hermite_sample_safe = false;
          }
      }
    }
    std::vector<corridor_volume::Poly> polys;
    int empty = 0;
    bool connected = true;
    for (const auto& h : firi.polys) {
      polys.emplace_back(h);
      empty += !polys.back().valid;
    }
    if (empty) connected = false;
    for (std::size_t i = 1; i < firi.polys.size(); ++i) {
      Eigen::MatrixX4d joined(firi.polys[i - 1].rows() + firi.polys[i].rows(),
                              4);
      joined << firi.polys[i - 1], firi.polys[i];
      connected &= corridor_volume::Poly(joined).valid;
    }
    const bool endpoints = polys.front().contains(seed.front()) &&
                           polys.back().contains(seed.back());
    connected &= endpoints;
    // Continuous half-space separation audit against the SAME solid spheres.
    bool safe = true;
    if (solid)
      for (const auto& poly : polys)
        if (poly.valid) {
          std::vector<int> ids;
          const double pad = options.robot_radius + options.safety_margin;
          map.query(poly.box.lo - Vec3::Constant(pad),
                    poly.box.hi + Vec3::Constant(pad), ids);
          for (int id : ids)
            if ((poly.h.leftCols<3>() * map.obstacles()[id].center +
                 poly.h.col(3))
                    .maxCoeff() < obstacle_radius + pad - 1e-7)
              safe = false;
          if ((poly.box.lo.array() < bounds.lower.array() + pad - 2e-7).any() ||
              (poly.box.hi.array() > bounds.upper.array() - pad + 2e-7).any())
            safe = false;
        }
    timer = Clock::now();
    const auto volume =
        corridor_volume::measure(corridor_volume::patches(hermite.corridor),
                                 polys, seed, samples, 8, 1., range);
    const double volume_ms = ms(timer);
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "metrics.json");
    out << std::setprecision(12);
    out << "{\"model\":\"" << mode << "\",\"seed_points\":" << seed.size()
        << ",\"expansion_policy\":\"" << policy
        << "\",\"local_range_m\":" << range
        << ",\"local_boxes\":" << expansion.size()
        << ",\"hermite_radius_bound_m\":" << options.max_radius
        << ",\"hermite_offset_bound_m\":" << options.max_offset
        << ",\"hermite_max_axis_m\":" << max_axis
        << ",\"hermite_max_center_shift_m\":"
        << hermite.metrics.max_center_shift
        << ",\"hermite_box_violation_m\":" << box_violation
        << ",\"hermite_optimization_ms\":" << hermite.metrics.optimization_ms
        << ",\"hermite_certificate_ms\":" << hermite.metrics.certificate_ms
        << ",\"hermite_constraints\":" << hermite.metrics.constraints
        << ",\"hermite_active_constraints\":" << hermite.metrics.active_constraints
        << ",\"hermite_newton_steps\":" << hermite.metrics.newton_steps
        << ",\"hermite_obstacle_pairs\":" << hermite.metrics.obstacle_pairs
        << ",\"hermite_sample_safe\":" << (hermite_sample_safe ? "true" : "false")
        << ",\"hermite_audit_samples\":" << hermite_audit_samples
        << ",\"hermite_refinements\":" << hermite.metrics.refinements
        << ",\"obstacles\":" << points.size() << ",\"index_ms\":" << index_ms
        << ",\"hermite_nodes\":" << hermite.metrics.nodes
        << ",\"firi_before_shortcut\":" << firi.before_shortcut
        << ",\"firi_polys\":" << polys.size() << ",\"firi_empty\":" << empty
        << ",\"firi_connected\":" << (connected ? "true" : "false")
        << ",\"firi_endpoints\":" << (endpoints ? "true" : "false")
        << ",\"firi_sphere_safe\":" << (safe ? "true" : "false")
        << ",\"hermite_median_ms\":" << quantile(ht, .5)
        << ",\"hermite_p95_ms\":" << quantile(ht, .95)
        << ",\"firi_median_ms\":" << quantile(ft, .5)
        << ",\"firi_p95_ms\":" << quantile(ft, .95)
        << ",\"firi_cover_ms\":" << quantile(ct, .5)
        << ",\"firi_erosion_ms\":" << quantile(et, .5)
        << ",\"firi_shortcut_ms\":" << quantile(st, .5)
        << ",\"hermite_times_ms\":";
    list(out, ht);
    out << ",\"firi_times_ms\":";
    list(out, ft);
    out << ",\"volume_samples\":" << volume.samples
        << ",\"volume_cells\":" << volume.cells
        << ",\"volume_ms\":" << volume_ms
        << ",\"unresolved_roots\":" << volume.unresolved;
    const char* names[] = {"hermite_union", "firi_union", "hermite_roi",
                           "firi_roi", "intersection"};
    for (int i = 0; i < 5; ++i)
      out << ",\"" << names[i] << "_m3\":" << volume.volume[i].mean << ",\""
          << names[i] << "_ci95\":" << volume.volume[i].ci95;
    out << "}\n";
    std::ofstream hp(dir / "firi_planes.csv");
    hp << "poly,nx,ny,nz,d\n" << std::setprecision(17);
    for (std::size_t i = 0; i < polys.size(); ++i)
      for (int j = 0; j < polys[i].h.rows(); ++j)
        hp << i << ',' << polys[i].h(j, 0) << ',' << polys[i].h(j, 1) << ','
           << polys[i].h(j, 2) << ',' << polys[i].h(j, 3) << '\n';
    std::cout << mode << " H=" << quantile(ht, .5)
              << "ms F=" << quantile(ft, .5) << "ms V=" << volume.volume[0].mean
              << '/' << volume.volume[1].mean << " connected=" << connected
              << " safe=" << safe << " volume_ms=" << volume_ms << '\n';
    return safe && connected && hermite_sample_safe && volume.unresolved == 0
               ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
