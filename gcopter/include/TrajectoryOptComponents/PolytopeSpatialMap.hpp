#pragma once

#include "TrajectoryOptAdapters/PolytopeProjectionSolverAdapter.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"

#include <cmath>

#include <Eigen/Eigen>

namespace traj_opt_components
{
class PolytopeSpatialMap
{
public:
    using VectorType = Eigen::Vector3d;

    const PolyhedraV *v_polys = nullptr;
    const Eigen::VectorXi *v_poly_idx = nullptr;
    int num_segments = 0;
    bool identity_mode = false;
    PolyhedraV owned_polys;
    Eigen::VectorXi owned_poly_idx;

    void reset(const PolyhedraV *polys,
               const Eigen::VectorXi *indices,
               int segments,
               bool identity = false)
    {
        v_polys = polys;
        v_poly_idx = indices;
        num_segments = segments;
        identity_mode = identity;
    }

    void reset(const PolyhedronV *poly,
               int segments,
               bool identity = false)
    {
        owned_polys.clear();
        owned_polys.push_back(*poly);
        owned_poly_idx = Eigen::VectorXi::Zero(segments);
        reset(&owned_polys, &owned_poly_idx, segments, identity);
    }

    int getUnconstrainedDim(int index) const
    {
        if (identity_mode || !v_polys || !v_poly_idx || index <= 0 || index > num_segments)
        {
            return 3;
        }
        return (*v_polys)[(*v_poly_idx)(index - 1)].cols();
    }

    /** @brief Map a borrowed variable block to position without temporary vectors.
     * @param xi Variables of getUnconstrainedDim(index); borrowed for this call.
     * @param index Waypoint index on the prepared trajectory.
     * @return Position in the corridor frame; near-zero normalized coordinates select its first vertex. */
    VectorType toPhysical(const Eigen::Ref<const Eigen::VectorXd> &xi, int index) const
    {
        if (identity_mode || !v_polys || !v_poly_idx || index <= 0 || index > num_segments)
            return xi.head<3>();
        const auto &poly = (*v_polys)[(*v_poly_idx)(index - 1)];
        const double norm = xi.norm();
        if (norm < 1e-12) return poly.col(0);
        VectorType position = poly.col(0);
        for (int i = 0; i + 1 < xi.size(); ++i)
        {
            const double q = xi(i) / norm;
            position.noalias() += (q * q) * poly.col(i + 1);
        }
        return position;
    }

    Eigen::VectorXd toUnconstrained(const Eigen::VectorXd &p, int index) const
    {
        if (identity_mode || !v_polys || !v_poly_idx || index <= 0 || index > num_segments)
        {
            return p;
        }

        Eigen::Matrix3Xd point(3, 1);
        point.col(0) = p.head<3>();
        Eigen::VectorXd xi;
        backwardP(point,
                  Eigen::VectorXi::Constant(1, (*v_poly_idx)(index - 1)),
                  *v_polys,
                  xi);
        return xi;
    }

    /** @brief Return an owning gradient for callers outside the optimizer's reusable path.
     * @param xi Current waypoint variables, borrowed for this call.
     * @param grad_p Position gradient in the corridor frame.
     * @param index Waypoint index.
     * @return Independent variable gradient; evaluation uses backwardGradInto to avoid this allocation. */
    Eigen::VectorXd backwardGrad(const Eigen::VectorXd &xi, const Eigen::VectorXd &grad_p, int index) const
    {
        Eigen::VectorXd gradient(xi.size());
        backwardGradInto(xi, grad_p.head<3>(), index, gradient);
        return gradient;
    }

    /** @brief Write the normalized-square-map pullback into caller-owned storage.
     * @param xi Current waypoint variables, borrowed for this call.
     * @param grad_p Position gradient in the corridor frame.
     * @param index Waypoint index.
     * @param[out] gradient Same-sized output, exclusive and non-overlapping with xi.
     * @note No allocation; near-zero coordinates have the constant first-vertex fallback's zero gradient. */
    void backwardGradInto(const Eigen::Ref<const Eigen::VectorXd> &xi, const VectorType &grad_p,
                          int index, Eigen::Ref<Eigen::VectorXd> gradient) const
    {
        if (identity_mode || !v_polys || !v_poly_idx || index <= 0 || index > num_segments)
        {
            gradient = grad_p;
            return;
        }
        const auto &poly = (*v_polys)[(*v_poly_idx)(index - 1)];
        const double norm = xi.norm();
        gradient.setZero();
        if (norm < 1e-12) return;
        double radial = 0.0;
        for (int i = 0; i + 1 < xi.size(); ++i)
        {
            const double q = xi(i) / norm;
            gradient(i) = 2.0 * q * poly.col(i + 1).dot(grad_p);
            radial += q * gradient(i);
        }
        for (int i = 0; i < xi.size(); ++i)
            gradient(i) = (gradient(i) - radial * (xi(i) / norm)) / norm;
    }

    /** @brief Add the unit-sphere penalty without overwriting physical objective gradients.
     * @param x Complete decision vector, borrowed for this call.
     * @param spatial_offset Beginning of spatial variables.
     * @param spatial_dim Number of spatial variables.
     * @param[in,out] grad Complete gradient, preserving all prior contributions.
     * @param[in,out] cost Objective accumulator. */
    void addNormPenalty(const Eigen::VectorXd &x,
                        int spatial_offset,
                        int spatial_dim,
                        Eigen::VectorXd &grad,
                        double &cost) const
    {
        if (identity_mode || !v_polys || !v_poly_idx || spatial_dim <= 0)
        {
            return;
        }

        const auto xi = x.segment(spatial_offset, spatial_dim);
        auto grad_xi = grad.segment(spatial_offset, spatial_dim);
        normRestrictionLayer(xi, *v_poly_idx, *v_polys, cost, grad_xi);
    }

private:
    static inline void normRestrictionLayer(const Eigen::Ref<const Eigen::VectorXd> &xi,
                                            const Eigen::VectorXi &v_idx,
                                            const PolyhedraV &v_polys,
                                            double &cost,
                                            Eigen::Ref<Eigen::VectorXd> grad_xi)
    {
        const long size_p = v_idx.size();

        double sqr_norm_q, sqr_norm_violation, c, dc;
        for (long i = 0, j = 0, k; i < size_p; ++i, j += k)
        {
            k = v_polys[v_idx(i)].cols();
            const auto q = xi.segment(j, k);
            sqr_norm_q = q.squaredNorm();
            sqr_norm_violation = sqr_norm_q - 1.0;
            if (sqr_norm_violation > 0.0)
            {
                c = sqr_norm_violation * sqr_norm_violation;
                dc = 3.0 * c;
                c *= sqr_norm_violation;
                cost += c;
                grad_xi.segment(j, k) += dc * 2.0 * q;
            }
        }
    }

    static inline double costTinyNLS(void *ptr,
                                     const Eigen::VectorXd &xi,
                                     Eigen::VectorXd &gradXi)
    {
        const long n = xi.size();
        const Eigen::Matrix3Xd &ov_poly = *(Eigen::Matrix3Xd *)ptr;

        const double sqr_norm_xi = xi.squaredNorm();
        const double inv_norm_xi = 1.0 / std::sqrt(sqr_norm_xi);
        const Eigen::VectorXd unit_xi = xi * inv_norm_xi;
        const Eigen::VectorXd r = unit_xi.head(n - 1);
        const Eigen::Vector3d delta = ov_poly.rightCols(n - 1) * r.cwiseProduct(r) +
                                      ov_poly.col(1) - ov_poly.col(0);
        double cost = delta.squaredNorm();
        gradXi.head(n - 1) = (ov_poly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                             r.array() * 2.0;
        gradXi(n - 1) = 0.0;
        gradXi = (gradXi - unit_xi.dot(gradXi) * unit_xi).eval() * inv_norm_xi;

        const double sqr_norm_violation = sqr_norm_xi - 1.0;
        if (sqr_norm_violation > 0.0)
        {
            double c = sqr_norm_violation * sqr_norm_violation;
            const double dc = 3.0 * c;
            c *= sqr_norm_violation;
            cost += c;
            gradXi += dc * 2.0 * xi;
        }

        return cost;
    }

    static inline void backwardP(const Eigen::Matrix3Xd &P,
                                 const Eigen::VectorXi &v_idx,
                                 const PolyhedraV &v_polys,
                                 Eigen::VectorXd &xi)
    {
        const long size_p = P.cols();
        long xi_dim = 0;
        for (long i = 0; i < size_p; ++i)
        {
            xi_dim += v_polys[v_idx(i)].cols();
        }
        xi.resize(xi_dim);

        double min_sqr_d;
        Eigen::Matrix3Xd ov_poly;
        for (long i = 0, j = 0, k, l; i < size_p; ++i, j += k)
        {
            l = v_idx(i);
            k = v_polys[l].cols();

            ov_poly.resize(3, k + 1);
            ov_poly.col(0) = P.col(i);
            ov_poly.rightCols(k) = v_polys[l];
            Eigen::VectorXd x(k);
            x.setConstant(std::sqrt(1.0 / static_cast<double>(k)));
            traj_opt_adapters::PolytopeProjectionSolverAdapter::optimize(x,
                                                                         min_sqr_d,
                                                                         &PolytopeSpatialMap::costTinyNLS,
                                                                         &ov_poly);

            xi.segment(j, k) = x;
        }
    }
};
}
