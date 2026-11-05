// SPDX-License-Identifier: MIT
#include "demo_generation.hpp"
#include "../../SweepCorridor/experiments/h3/generator.hpp"

namespace demo_generation {
Result generate(bool h3, const std::vector<sc::Vec3>& seed,
                const sc::ObstacleIndex& map, const sc::Bounds& bounds,
                const sc::Options& options) {
  Result out;
  if (h3) {
    sc::experiments::h3::OptimizerSettings settings;
    settings.profile = options.profile;
    auto r = sc::experiments::h3::generate(seed, map, bounds, options, settings);
    out.status = r.status;
    out.message = std::move(r.message);
    out.metrics.total_ms = r.total_ms;
    out.metrics.controls = r.optimizer.controls;
    out.metrics.constraints = r.optimizer.constraints;
    out.metrics.iterations = r.optimizer.steps;
    out.metrics.refinements = r.refinements;
    out.metrics.solver = std::move(r.optimizer);
    out.jet_updates = r.jets.accepted;
    out.joint_steps = r.jets.joint_steps;
    out.end_steps = r.jets.end_steps;
    out.refreshed_spans = r.jets.refresh_spans;
    out.budget_skips = r.jets.joint_budget_skips;
    if (out) out.corridor = Corridor(std::move(r.corridor));
  } else {
    auto r = sc::Generator(options).generate(seed, map, bounds);
    out.status = r.status;
    out.message = std::move(r.message);
    out.metrics = std::move(r.metrics);
    if (out) out.corridor = Corridor(std::move(r.corridor));
  }
  return out;
}
} // namespace demo_generation
