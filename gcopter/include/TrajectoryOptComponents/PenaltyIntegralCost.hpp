#ifndef SPLINE_TRAJECTORY_PENALTY_INTEGRAL_COST_HPP
#define SPLINE_TRAJECTORY_PENALTY_INTEGRAL_COST_HPP

#include "TrajectoryOptComponents/PenaltyUtils.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "TrajectoryOptComponents/SpatialCosts/AngularRateBoundPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/FlatnessState.hpp"
#include "TrajectoryOptComponents/SpatialCosts/PolytopePositionPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/ThrustBandPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/TiltAnglePenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/VelocityBoundPenalty.hpp"
#include "gcopter/flatness.hpp"

namespace gcopter
{
using traj_opt_components::PolyhedronH;
using traj_opt_components::PolyhedraH;

struct PenaltyIntegralCost
{
    using VectorType = Eigen::Vector3d;

    const PolyhedraH *h_polys = nullptr;
    const Eigen::VectorXi *h_poly_idx = nullptr;
    double smooth_eps = 0.0;
    Eigen::VectorXd magnitude_bounds;
    Eigen::VectorXd penalty_weights;
    flatness::FlatnessMap *flatmap = nullptr;

    void reset(const PolyhedraH *polys,
               const Eigen::VectorXi *indices,
               double smoothing,
               const Eigen::VectorXd &magnitudeBounds,
               const Eigen::VectorXd &penaltyWeights,
               flatness::FlatnessMap *fm)
    {
        h_polys = polys;
        h_poly_idx = indices;
        smooth_eps = smoothing;
        magnitude_bounds = magnitudeBounds;
        penalty_weights = penaltyWeights;
        flatmap = fm;
    }

    double operator()(double /*t*/, double /*t_global*/, int seg_idx,
                      const VectorType &p, const VectorType &v,
                      const VectorType &a, const VectorType &j,
                      const VectorType &/*s*/, VectorType &gp,
                      VectorType &gv, VectorType &ga, VectorType &gj,
                      VectorType &/*gs*/, double &/*gt*/) const
    {
        if (!h_polys || !h_poly_idx || !flatmap)
            return 0.0;

        const double weightPos = penalty_weights(0);
        const double weightVel = penalty_weights(1);
        const double weightOmg = penalty_weights(2);
        const double weightTheta = penalty_weights(3);
        const double weightThrust = penalty_weights(4);

        const auto flatness_state =
            traj_opt_components::evaluateFlatnessPenaltyState(flatmap, v, a, j);

        VectorType gradPos = VectorType::Zero();
        VectorType gradVel = VectorType::Zero();
        VectorType gradOmg = VectorType::Zero();
        double gradThr = 0.0;
        Eigen::Vector4d gradQuat = Eigen::Vector4d::Zero();

        const int poly_id = (*h_poly_idx)(seg_idx);
        const PolyhedronH &poly = (*h_polys)[poly_id];
        double pena = 0.0;

        pena += traj_opt_components::accumulatePolytopePositionPenalty(poly,
                                                                       p,
                                                                       smooth_eps,
                                                                       weightPos,
                                                                       gradPos);
        pena += traj_opt_components::accumulateVelocityBoundPenalty(v,
                                                                    magnitude_bounds(0),
                                                                    smooth_eps,
                                                                    weightVel,
                                                                    gradVel);
        pena += traj_opt_components::accumulateAngularRateBoundPenalty(flatness_state.angular_rate,
                                                                       magnitude_bounds(1),
                                                                       smooth_eps,
                                                                       weightOmg,
                                                                       gradOmg);
        pena += traj_opt_components::accumulateTiltAnglePenalty(flatness_state.quaternion,
                                                                magnitude_bounds(2),
                                                                smooth_eps,
                                                                weightTheta,
                                                                gradQuat);
        pena += traj_opt_components::accumulateThrustBandPenalty(flatness_state.thrust,
                                                                 magnitude_bounds(3),
                                                                 magnitude_bounds(4),
                                                                 smooth_eps,
                                                                 weightThrust,
                                                                 gradThr);

        VectorType totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
        double totalGradPsi = 0.0, totalGradPsiD = 0.0;
        flatmap->backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                          totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                          totalGradPsi, totalGradPsiD);

        gp += totalGradPos;
        gv += totalGradVel;
        ga += totalGradAcc;
        gj += totalGradJer;

        return pena;
    }

};
}

#endif
