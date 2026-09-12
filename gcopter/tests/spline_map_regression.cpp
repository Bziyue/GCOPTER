#include <TrajectoryOptComponents/PolytopeSpatialMap.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

/** @brief Check the complete position-plus-domain gradient, including preservation inside the unit ball. */
int main()
{
    traj_opt_components::PolyhedronV poly(3, 4);
    poly << 0.1,1,0,0, 0.2,0,1,0, -0.1,0,0,1;
    traj_opt_components::PolytopeSpatialMap map;
    map.reset(&poly, 1);
    for (double scale : {0.5, 2.0})
    {
        Eigen::VectorXd x(4); x << 0.2,0.4,0.3,0.5; x *= scale;
        auto objective = [&](const Eigen::VectorXd &point) {
            double value = map.toPhysical(point, 1).squaredNorm();
            Eigen::VectorXd ignored = Eigen::VectorXd::Zero(4);
            map.addNormPenalty(point, 0, 4, ignored, value);
            return value;
        };
        Eigen::VectorXd gradient(x.size());
        map.backwardInto(x, 2.0 * map.toPhysical(x, 1), 1, gradient);
        const auto before = gradient;
        double value = map.toPhysical(x, 1).squaredNorm();
        map.addNormPenalty(x, 0, 4, gradient, value);
        if (x.squaredNorm() <= 1.0)
            require(gradient.isApprox(before, 1e-14), "domain penalty overwrote physical gradient");
        for (int i = 0; i < x.size(); ++i)
        {
            auto plus = x; auto minus = x; plus(i) += 1e-6; minus(i) -= 1e-6;
            const double numerical = (objective(plus) - objective(minus)) / 2e-6;
            require(std::abs(numerical - gradient(i)) < 2e-7 * std::max(1.0, std::abs(numerical)),
                    "complete mapped gradient differs from finite differences");
        }
    }
    std::cout << "mapped objective gradient passed\n";
}
