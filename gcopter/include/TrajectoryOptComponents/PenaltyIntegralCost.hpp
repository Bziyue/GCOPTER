#ifndef SPLINE_TRAJECTORY_PENALTY_INTEGRAL_COST_HPP
#define SPLINE_TRAJECTORY_PENALTY_INTEGRAL_COST_HPP

#include "SplineTrajectory/SplineTrajectory.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "gcopter/flatness.hpp"

#include <cmath>
#include <vector>

namespace gcopter
{
struct DynamicObstacleTrajectory
{
    int drone_id = -1;
    double start_time = 0.0;
    SplineTrajectory::QuinticSpline3D traj;
};

struct PenaltyIntegralCost
{
    using VectorType = Eigen::Vector3d;

    const PolyhedraH *h_polys = nullptr;
    const Eigen::VectorXi *h_poly_idx = nullptr;
    double smooth_eps = 0.0;
    Eigen::VectorXd magnitude_bounds;
    Eigen::VectorXd penalty_weights;
    flatness::FlatnessMap *flatmap = nullptr;
    const std::vector<DynamicObstacleTrajectory> *dynamic_obstacles = nullptr;
    double dynamic_obstacle_weight = 0.0;
    Eigen::Vector3d dynamic_obstacle_ellipsoid = Eigen::Vector3d::Ones();
    int self_drone_id = -1;

    void reset(const PolyhedraH *polys,
               const Eigen::VectorXi *indices,
               double smoothing,
               const Eigen::VectorXd &magnitudeBounds,
               const Eigen::VectorXd &penaltyWeights,
               flatness::FlatnessMap *fm,
               const std::vector<DynamicObstacleTrajectory> *dynamicObstacles = nullptr,
               double dynamicObstacleWeight = 0.0,
               const Eigen::Vector3d &dynamicObstacleEllipsoid = Eigen::Vector3d::Ones(),
               int selfDroneId = -1)
    {
        h_polys = polys;
        h_poly_idx = indices;
        smooth_eps = smoothing;
        magnitude_bounds = magnitudeBounds;
        penalty_weights = penaltyWeights;
        flatmap = fm;
        dynamic_obstacles = dynamicObstacles;
        dynamic_obstacle_weight = dynamicObstacleWeight;
        dynamic_obstacle_ellipsoid = dynamicObstacleEllipsoid;
        self_drone_id = selfDroneId;
    }

    double operator()(double /*t*/, double t_global, int seg_idx,
                      const VectorType &p, const VectorType &v,
                      const VectorType &a, const VectorType &j,
                      const VectorType &/*s*/, VectorType &gp,
                      VectorType &gv, VectorType &ga, VectorType &gj,
                      VectorType &/*gs*/, double &gt) const
    {
        if (!h_polys || !h_poly_idx || !flatmap)
            return 0.0;

        const double velSqrMax = magnitude_bounds(0) * magnitude_bounds(0);
        const double omgSqrMax = magnitude_bounds(1) * magnitude_bounds(1);
        const double thetaMax = magnitude_bounds(2);
        const double thrustMean = 0.5 * (magnitude_bounds(3) + magnitude_bounds(4));
        const double thrustRadi = 0.5 * std::fabs(magnitude_bounds(4) - magnitude_bounds(3));
        const double thrustSqrRadi = thrustRadi * thrustRadi;

        const double weightPos = penalty_weights(0);
        const double weightVel = penalty_weights(1);
        const double weightOmg = penalty_weights(2);
        const double weightTheta = penalty_weights(3);
        const double weightThrust = penalty_weights(4);

        double thr = 0.0;
        Eigen::Vector4d quat(1.0, 0.0, 0.0, 0.0);
        Eigen::Vector3d omg(0.0, 0.0, 0.0);
        flatmap->forward(v, a, j, 0.0, 0.0, thr, quat, omg);

        VectorType gradPos = VectorType::Zero();
        VectorType gradVel = VectorType::Zero();
        VectorType gradOmg = VectorType::Zero();
        double gradThr = 0.0;
        Eigen::Vector4d gradQuat = Eigen::Vector4d::Zero();

        const int poly_id = (*h_poly_idx)(seg_idx);
        const PolyhedronH &poly = (*h_polys)[poly_id];
        const int K = poly.rows();
        double pena = 0.0;

        for (int k = 0; k < K; ++k)
        {
            const Eigen::Vector3d outerNormal = poly.block<1, 3>(k, 0);
            const double violaPos = outerNormal.dot(p) + poly(k, 3);
            double violaPosPena = 0.0;
            double violaPosPenaD = 0.0;
            if (smoothedL1(violaPos, smooth_eps, violaPosPena, violaPosPenaD))
            {
                gradPos += weightPos * violaPosPenaD * outerNormal;
                pena += weightPos * violaPosPena;
            }
        }

        double violaVel = v.squaredNorm() - velSqrMax;
        double violaVelPena = 0.0;
        double violaVelPenaD = 0.0;
        if (smoothedL1(violaVel, smooth_eps, violaVelPena, violaVelPenaD))
        {
            gradVel += weightVel * violaVelPenaD * 2.0 * v;
            pena += weightVel * violaVelPena;
        }

        double violaOmg = omg.squaredNorm() - omgSqrMax;
        double violaOmgPena = 0.0;
        double violaOmgPenaD = 0.0;
        if (smoothedL1(violaOmg, smooth_eps, violaOmgPena, violaOmgPenaD))
        {
            gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
            pena += weightOmg * violaOmgPena;
        }

        double cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
        if (cos_theta > 1.0)
            cos_theta = 1.0;
        else if (cos_theta < -1.0)
            cos_theta = -1.0;
        double violaTheta = std::acos(cos_theta) - thetaMax;
        double violaThetaPena = 0.0;
        double violaThetaPenaD = 0.0;
        if (smoothedL1(violaTheta, smooth_eps, violaThetaPena, violaThetaPenaD))
        {
            const double denom = std::sqrt(std::max(1.0 - cos_theta * cos_theta, 1e-12));
            gradQuat += weightTheta * violaThetaPenaD /
                        denom * 4.0 *
                        Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
            pena += weightTheta * violaThetaPena;
        }

        double violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;
        double violaThrustPena = 0.0;
        double violaThrustPenaD = 0.0;
        if (smoothedL1(violaThrust, smooth_eps, violaThrustPena, violaThrustPenaD))
        {
            gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
            pena += weightThrust * violaThrustPena;
        }

        VectorType totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
        double totalGradPsi = 0.0, totalGradPsiD = 0.0;
        flatmap->backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                          totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                          totalGradPsi, totalGradPsiD);

        gp += totalGradPos;
        gv += totalGradVel;
        ga += totalGradAcc;
        gj += totalGradJer;

        pena += dynamicObstaclePenalty(t_global, p, gp, gt);

        return pena;
    }

private:
    double dynamicObstaclePenalty(const double t_global,
                                  const VectorType &p,
                                  VectorType &gp,
                                  double &gt) const
    {
        if (!dynamic_obstacles || dynamic_obstacle_weight <= 0.0)
        {
            return 0.0;
        }

        const Eigen::Array3d safe_axes = dynamic_obstacle_ellipsoid.array().max(1.0e-3);
        const Eigen::Array3d inv_axes2 = safe_axes.square().inverse();

        double penalty = 0.0;
        for (const auto &obstacle : *dynamic_obstacles)
        {
            if (obstacle.drone_id == self_drone_id ||
                !obstacle.traj.isInitialized() ||
                obstacle.traj.getNumSegments() <= 0)
            {
                continue;
            }

            const double traj_start = obstacle.traj.getStartTime();
            const double rel_time = std::max(0.0, t_global - obstacle.start_time);
            const double end_time = obstacle.traj.getEndTime();

            VectorType obstacle_pos;
            VectorType obstacle_vel;
            if (traj_start + rel_time <= end_time)
            {
                obstacle_pos = obstacle.traj.getTrajectory().evaluate(traj_start + rel_time, SplineTrajectory::Deriv::Pos);
                obstacle_vel = obstacle.traj.getTrajectory().evaluate(traj_start + rel_time, SplineTrajectory::Deriv::Vel);
            }
            else
            {
                obstacle_pos = obstacle.traj.getTrajectory().evaluate(end_time, SplineTrajectory::Deriv::Pos);
                obstacle_vel = obstacle.traj.getTrajectory().evaluate(end_time, SplineTrajectory::Deriv::Vel);
                obstacle_pos += (traj_start + rel_time - end_time) * obstacle_vel;
            }

            const VectorType delta = p - obstacle_pos;
            const double ellipsoid_distance_sq =
                delta.array().square().matrix().cwiseProduct(inv_axes2.matrix()).sum();
            const double violation = 1.0 - ellipsoid_distance_sq;
            if (violation <= 0.0)
            {
                continue;
            }

            const double violation_sq = violation * violation;
            penalty += dynamic_obstacle_weight * violation_sq * violation;

            const VectorType grad_ellipsoid =
                2.0 * delta.array().matrix().cwiseProduct(inv_axes2.matrix());
            const VectorType grad_pos =
                -3.0 * dynamic_obstacle_weight * violation_sq * grad_ellipsoid;
            gp += grad_pos;
            gt += grad_pos.dot(-obstacle_vel);
        }

        return penalty;
    }

    static inline bool smoothedL1(const double &x,
                                  const double &mu,
                                  double &f,
                                  double &df)
    {
        if (x < 0.0)
        {
            return false;
        }
        else if (x > mu)
        {
            f = x - 0.5 * mu;
            df = 1.0;
            return true;
        }
        else
        {
            const double xdmu = x / mu;
            const double sqrxdmu = xdmu * xdmu;
            const double mumxd2 = mu - 0.5 * x;
            f = mumxd2 * sqrxdmu * xdmu;
            df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return true;
        }
    }
};
}

#endif
