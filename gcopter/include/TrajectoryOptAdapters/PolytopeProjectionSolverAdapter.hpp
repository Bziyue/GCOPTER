#pragma once

#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>

#include <cfloat>

namespace traj_opt_adapters
{
class PolytopeProjectionSolverAdapter
{
public:
    using CostFunction = double (*)(void *instance,
                                    const Eigen::VectorXd &x,
                                    Eigen::VectorXd &g);

    static inline void optimize(Eigen::VectorXd &x,
                                double &min_cost,
                                CostFunction cost_function,
                                void *instance)
    {
        lbfgs::lbfgs_parameter_t params;
        params.past = 0;
        params.delta = 1.0e-5;
        params.g_epsilon = FLT_EPSILON;
        params.max_iterations = 128;
        lbfgs::lbfgs_optimize(x,
                              min_cost,
                              cost_function,
                              nullptr,
                              nullptr,
                              instance,
                              params);
    }
};
} // namespace traj_opt_adapters
