#ifndef SPLINE_TRAJECTORY_LINEAR_TIME_COST_HPP
#define SPLINE_TRAJECTORY_LINEAR_TIME_COST_HPP

#include <vector>
#include <Eigen/Eigen>

namespace gcopter
{
struct TimeCost
{
    double weight = 0.0;

    double operator()(const std::vector<double> &Ts, Eigen::VectorXd &grad) const
    {
        double cost = 0.0;
        for (size_t i = 0; i < Ts.size(); ++i)
        {
            cost += weight * Ts[i];
            grad(i) += weight;
        }
        return cost;
    }
};
}

#endif
