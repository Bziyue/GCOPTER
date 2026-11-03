#pragma once

#include "SplineTrajectory/SplineConvexHull.hpp"
#include "SplineTrajectory/SplineOptimizer.hpp"
#include "TrajectoryOptComponents/SpatialCosts/VelocityBoundPenalty.hpp"
#include "TrajectoryOptComponents/SpatialCosts/PolytopePositionPenalty.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"

namespace traj_opt_adapters
{
/** @brief 同一源段的所有细分控制点使用同一归一化凸多面体。借用对象须覆盖求解生命周期。 */
struct SegmentPolytopePenalty
{
    const traj_opt_components::PolyhedraH *polys = nullptr;
    const Eigen::VectorXi *indices = nullptr;
    double smoothing = 0.01, weight = 0;
    double operator()(int, int segment, double, const Eigen::Vector3d &point,
                      Eigen::Vector3d &gradient, double &violation) const
    {
        return traj_opt_components::accumulatePolytopePositionPenalty(
            (*polys)[(*indices)(segment)], point, smoothing, weight, gradient, &violation);
    }
};

/**
 * @brief 细分 Bézier 的位置与速度软惩罚，复用固定拓扑工作区。
 * @note 参考 diff-planner-swarm BezierTrajectoryCost；不依赖其规划器业务类型。
 *       position(piece, segment, alpha, p, grad, violation) 返回控制点代价密度。
 *       凸位置集合和速度球有凸包充分性；有限权重软罚本身不是硬安全证书。
 *       非凸位置回调只有控制点惩罚语义，不能推广凸包充分性。
 */
template<class Spline, class PositionPenalty>
class BezierPenaltyCost
{
    using Hull = SplineTrajectory::ConvexHullWorkspace<3>;
    Hull position_hull_, velocity_hull_;
    typename Hull::CoefficientMatrix position_gradient_, velocity_gradient_;
    bool velocity_enabled_ = false;
    double speed_ = 0, speed_weight_ = 0, smoothing_ = 0.01;

public:
    PositionPenalty position;
    double max_position_violation = 0, max_speed_violation = 0;

    /** @brief 每次问题初始化调用一次；迭代内不重新分配或归一化几何。 */
    void configure(int segments, int depth, double speed, double speed_weight, double smoothing)
    {
        position_hull_.configure(segments, Spline::kCoefficientCount,
            {SplineTrajectory::ConvexHullBasis::Bezier, 0, depth});
        position_gradient_.resize(position_hull_.controls().rows(), 3);
        velocity_enabled_ = speed_weight > 0;
        speed_ = speed; speed_weight_ = speed_weight; smoothing_ = smoothing;
        if (velocity_enabled_)
        {
            // 从幂基导数构造速度，避免大世界坐标的位置控制点相减。
            velocity_hull_.configure(segments, Spline::kCoefficientCount,
                {SplineTrajectory::ConvexHullBasis::Bezier, 1, depth});
            velocity_gradient_.resize(velocity_hull_.controls().rows(), 3);
        }
    }

    double operator()(const SplineTrajectory::ParameterView<Spline> &state,
                      SplineTrajectory::CoefficientGradient<Spline> &gradient)
    {
        max_position_violation = max_speed_violation = 0;
        double cost = 0;
        position_hull_.update(state.polynomial);
        const double count = double(position_hull_.controls().rows()) / state.durations.size();
        bool active = false;
        for (int piece = 0; piece < position_hull_.numPieces(); ++piece)
        {
            const auto &info = position_hull_.pieceInfo(piece);
            const int segment = info.source_segment;
            const double scale = state.durations[segment] / count;
            for (int control = 0; control < Spline::kCoefficientCount; ++control)
            {
                const int row = piece * Spline::kCoefficientCount + control;
                const double alpha = info.source_fraction_begin +
                    (info.source_fraction_end - info.source_fraction_begin) * control / (Spline::kCoefficientCount - 1.0);
                Eigen::Vector3d partial = Eigen::Vector3d::Zero();
                double violation = 0;
                const double density = position(piece, segment, alpha,
                    position_hull_.controls().row(row).transpose(), partial, violation);
                max_position_violation = std::max(max_position_violation, violation);
                cost += scale * density;
                gradient.durations(segment) += density / count;
                position_gradient_.row(row) = (scale * partial).transpose();
                active = active || partial.squaredNorm() > 0;
            }
        }
        if (active) position_hull_.backwardAdd(position_gradient_, gradient.coefficients, gradient.durations);
        if (!velocity_enabled_) return cost;

        velocity_hull_.update(state.polynomial);
        const int cp = velocity_hull_.controlsPerPiece();
        const double velocity_count = double(velocity_hull_.controls().rows()) / state.durations.size();
        active = false;
        for (int piece = 0; piece < velocity_hull_.numPieces(); ++piece)
        {
            const int segment = velocity_hull_.pieceInfo(piece).source_segment;
            const double scale = state.durations[segment] / velocity_count;
            for (int control = 0; control < cp; ++control)
            {
                const int row = piece * cp + control;
                const Eigen::Vector3d v = velocity_hull_.controls().row(row).transpose();
                Eigen::Vector3d partial = Eigen::Vector3d::Zero();
                max_speed_violation = std::max(max_speed_violation, v.norm() - speed_);
                const double density = traj_opt_components::accumulateVelocityBoundPenalty(
                    v, speed_, smoothing_, speed_weight_, partial);
                cost += scale * density;
                gradient.durations(segment) += density / velocity_count;
                velocity_gradient_.row(row) = (scale * partial).transpose();
                active = active || partial.squaredNorm() > 0;
            }
        }
        if (active) velocity_hull_.backwardAdd(velocity_gradient_, gradient.coefficients, gradient.durations);
        return cost;
    }
};
} // namespace traj_opt_adapters
