// SPDX-License-Identifier: MIT
#pragma once
#include <SweepCorridor.hpp>
#include <variant>

namespace demo_generation {
namespace sc = SweepCorridor;
// Display either certified generator output directly, without refitting H3
// into a C2 model or changing the production library's import/query API.
class Corridor {
  std::variant<sc::Corridor, sc::internal::cubic::Geometry> geometry_;
public:
  Corridor() = default;
  explicit Corridor(sc::Corridor value) : geometry_(std::move(value)) {}
  explicit Corridor(sc::internal::cubic::Geometry value) : geometry_(std::move(value)) {}
  static constexpr int degree = 3;
  std::size_t pieces() const { return knots().empty() ? 0 : knots().size()-1; }
  const std::vector<double>& knots() const {
    if (const auto* c = std::get_if<sc::Corridor>(&geometry_)) return c->knots();
    return std::get<sc::internal::cubic::Geometry>(geometry_).breaks;
  }
  const std::vector<sc::Control>& controls() const {
    if (const auto* c = std::get_if<sc::Corridor>(&geometry_)) return c->controls();
    return std::get<sc::internal::cubic::Geometry>(geometry_).controls;
  }
  sc::Section section(double u) const {
    return std::visit([u](const auto& c) { return c.section(u); }, geometry_);
  }
  const sc::Corridor& c3() const { return std::get<sc::Corridor>(geometry_); }
};
struct Result {
  sc::Status status = sc::Status::InvalidInput;
  std::string message;
  Corridor corridor;
  sc::Metrics metrics;
  int jet_updates = 0;
  int joint_steps = 0;
  explicit operator bool() const { return status == sc::Status::Success; }
};
Result generate(bool h3, const std::vector<sc::Vec3>& seed,
                const sc::ObstacleIndex& map, const sc::Bounds& bounds,
                const sc::Options& options);
} // namespace demo_generation
