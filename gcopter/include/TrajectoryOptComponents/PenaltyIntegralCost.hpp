#ifndef SPLINE_TRAJECTORY_PENALTY_INTEGRAL_COST_HPP
#define SPLINE_TRAJECTORY_PENALTY_INTEGRAL_COST_HPP

#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "gcopter/flatness.hpp"

#include <cmath>

namespace gcopter
{
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

        return pena;
    }

private:
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
