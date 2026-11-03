#ifndef SPLINE_SFC_OPTIMIZER_HPP
#define SPLINE_SFC_OPTIMIZER_HPP

#include "SplineTrajectory/SplineOptimizer.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"
#include "TrajectoryOptAdapters/GcopterPenaltyIntegralCostAdapter.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "TrajectoryOptComponents/LinearTimeCost.hpp"
#include "TrajectoryOptComponents/PolytopeSpatialMap.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <vector>

namespace gcopter
{
    using traj_opt_components::LinearTimeCost;
    using traj_opt_components::PolyhedraH;
    using traj_opt_components::PolyhedraV;
    using traj_opt_components::PolyhedronH;
    using traj_opt_components::PolyhedronV;
    using PenaltyIntegralCost = traj_opt_adapters::GcopterPenaltyIntegralCostAdapter;

    class SplineSFCOptimizer
    {
    public:
        using SplineType = SplineTrajectory::QuinticSplineND<3>;
        using Polynomial = typename SplineType::Polynomial;
        struct Parameterization
        {
            SplineTrajectory::QuadInvTimeMap time;
            traj_opt_components::PolytopeSpatialMap space;
            SplineTrajectory::NoAuxiliaryMap auxiliary;
        };
        using OptimizerType = SplineTrajectory::SplineOptimizer<SplineType, Parameterization>;

    private:
        OptimizerType optimizer_;
        traj_opt_components::PolytopeSpatialMap spatial_map_;
        LinearTimeCost time_cost_;
        PenaltyIntegralCost integral_cost_;
        flatness::FlatnessMap flatmap_;

        Eigen::Matrix3d headPVA_;
        Eigen::Matrix3d tailPVA_;

        PolyhedraV vPolytopes_;
        PolyhedraH hPolytopes_;
        Eigen::Matrix3Xd shortPath_;

        Eigen::VectorXi pieceIdx_;
        Eigen::VectorXi vPolyIdx_;
        Eigen::VectorXi hPolyIdx_;

        int polyN_ = 0;
        int pieceN_ = 0;

        double smoothEps_ = 0.0;
        int integralRes_ = 0;
        Eigen::VectorXd magnitudeBd_;
        Eigen::VectorXd penaltyWt_;
        Eigen::VectorXd physicalPm_;
        double allocSpeed_ = 0.0;

        lbfgs::lbfgs_parameter_t lbfgs_params_{};

    private:
        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = std::sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / std::sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(std::sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &SplineSFCOptimizer::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

        void applyNormPenalty(const Eigen::VectorXd &x,
                              Eigen::Ref<Eigen::VectorXd> grad,
                              double &cost) const
        {
            if (pieceN_ <= 1)
                return;

            int offset = pieceN_;
            const int inner_count = pieceN_ - 1;
            for (int i = 0; i < inner_count; ++i)
            {
                const int k = vPolytopes_[vPolyIdx_(i)].cols();
                Eigen::Map<const Eigen::VectorXd> q(x.data() + offset, k);
                Eigen::Map<Eigen::VectorXd> gradQ(grad.data() + offset, k);

                const double sqrNormQ = q.squaredNorm();
                const double sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    double c = sqrNormViolation * sqrNormViolation;
                    const double dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }

                offset += k;
            }
        }

        struct DomainCost
        {
            const SplineSFCOptimizer *owner;
            double operator()(const SplineTrajectory::DecisionView &state,
                              Eigen::Ref<Eigen::VectorXd> gradient) const
            {
                double cost = 0;
                owner->applyNormPenalty(state.variables, gradient, cost);
                return cost;
            }
        };
        struct Objective
        {
            LinearTimeCost &duration;
            PenaltyIntegralCost &integral;
            DomainCost decision;
        };

        static double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &g)
        {
            auto &self = *static_cast<SplineSFCOptimizer *>(ptr);
            Objective objective{self.time_cost_, self.integral_cost_, {&self}};
            return self.optimizer_.costAndGradient(x, g, objective);
        }

    public:
        bool setup(const double &timeWeight,
                   const Eigen::Matrix3d &initialPVA,
                   const Eigen::Matrix3d &terminalPVA,
                   const PolyhedraH &safeCorridor,
                   const double &lengthPerPiece,
                   const double &smoothingFactor,
                   const int &integralResolution,
                   const Eigen::VectorXd &magnitudeBounds,
                   const Eigen::VectorXd &penaltyWeights,
                   const Eigen::VectorXd &physicalParams)
        {
            headPVA_ = initialPVA;
            tailPVA_ = terminalPVA;

            hPolytopes_ = safeCorridor;
            for (size_t i = 0; i < hPolytopes_.size(); i++)
            {
                const Eigen::ArrayXd norms =
                    hPolytopes_[i].leftCols<3>().rowwise().norm();
                hPolytopes_[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes_, vPolytopes_))
            {
                return false;
            }

            polyN_ = hPolytopes_.size();
            smoothEps_ = smoothingFactor;
            integralRes_ = integralResolution;
            magnitudeBd_ = magnitudeBounds;
            penaltyWt_ = penaltyWeights;
            physicalPm_ = physicalParams;
            allocSpeed_ = magnitudeBd_(0) * 3.0;

            getShortestPath(headPVA_.col(0), tailPVA_.col(0),
                            vPolytopes_, smoothEps_, shortPath_);
            const Eigen::Matrix3Xd deltas = shortPath_.rightCols(polyN_) - shortPath_.leftCols(polyN_);
            pieceIdx_ = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx_.array() += 1;
            pieceN_ = pieceIdx_.sum();

            vPolyIdx_.resize(pieceN_ - 1);
            hPolyIdx_.resize(pieceN_);
            for (int i = 0, j = 0, k; i < polyN_; i++)
            {
                k = pieceIdx_(i);
                for (int l = 0; l < k; l++, j++)
                {
                    if (l < k - 1)
                    {
                        vPolyIdx_(j) = 2 * i;
                    }
                    else if (i < polyN_ - 1)
                    {
                        vPolyIdx_(j) = 2 * i + 1;
                    }
                    hPolyIdx_(j) = i;
                }
            }

            spatial_map_.reset(&vPolytopes_, &vPolyIdx_, pieceN_);
            optimizer_ = OptimizerType(Parameterization{{}, spatial_map_, {}});
            SplineTrajectory::OptimizerOptions config;
            config.energy_weight = 1.0;
            config.integration_steps = integralRes_;
            typename SplineType::CoefficientMatrix waypoints(pieceN_ + 1, 3);
            waypoints.row(0) = headPVA_.col(0).transpose();

            Eigen::Matrix3Xd innerPoints;
            Eigen::VectorXd timeAlloc;
            setInitial(shortPath_, allocSpeed_, pieceIdx_, innerPoints, timeAlloc);
            for (int i = 0; i < innerPoints.cols(); ++i)
            {
                waypoints.row(i + 1) = innerPoints.col(i).transpose();
            }
            waypoints.row(pieceN_) = tailPVA_.col(0).transpose();

            SplineTrajectory::BoundaryConditions<3> bc;
            bc.start_velocity = headPVA_.col(1);
            bc.start_acceleration = headPVA_.col(2);
            bc.end_velocity = tailPVA_.col(1);
            bc.end_acceleration = tailPVA_.col(2);

            typename OptimizerType::Problem problem;
            problem.durations.assign(timeAlloc.data(), timeAlloc.data() + timeAlloc.size());
            problem.waypoints = waypoints;
            problem.start_time = 0.0;
            problem.boundary = bc;
            if (!optimizer_.prepare(problem, config))
            {
                return false;
            }

            time_cost_.weight = timeWeight;
            flatmap_.reset(physicalPm_(0), physicalPm_(1), physicalPm_(2),
                           physicalPm_(3), physicalPm_(4), physicalPm_(5));
            integral_cost_.reset(&hPolytopes_, &hPolyIdx_, smoothEps_, magnitudeBd_, penaltyWt_, &flatmap_);

            return true;
        }

        /** @brief Optimize a prepared problem; failure clears the output polynomial. */
        double optimize(Polynomial &polynomial, const double &relCostTol)
        {
            Eigen::VectorXd x = optimizer_.initialGuess();
            lbfgs_params_.mem_size = 256;
            lbfgs_params_.past = 3;
            lbfgs_params_.min_step = 1e-32;
            lbfgs_params_.g_epsilon = 0;
            lbfgs_params_.delta = relCostTol;
            Objective objective{time_cost_, integral_cost_, {this}};
            const char *check = std::getenv("GCOPTER_GRAD_CHECK");
            if (check && std::string(check) == "1")
            {
                const auto result = SplineTrajectory::checkGradients(optimizer_, x, objective);
                std::cerr << "[GradCheck] relative error: " << result.relative_error
                          << ", absolute error: " << result.error_norm << std::endl;
            }
            double cost = 0;
            const int status = lbfgs::lbfgs_optimize(x, cost, costFunctional, nullptr, nullptr, this, lbfgs_params_);
            if (status < 0)
            {
                polynomial = Polynomial{};
                std::cerr << "Optimization Failed: " << lbfgs::lbfgs_strerror(status) << std::endl;
                return INFINITY;
            }
            // Synchronize the accepted vector before publishing an owning polynomial.
            if (!optimizer_.updateAccepted(x)) { polynomial = Polynomial{}; return INFINITY; }
            polynomial = optimizer_.copyPolynomial();
            return cost;
        }
    };
}
#endif
