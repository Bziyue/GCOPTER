#include "misc/visualizer.hpp"
#include "gcopter/spline_sfc_optimizer.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

struct Config
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;
    int droneCount;
    int maxSamplingAttempts;
    int maxPlanningAttemptsPerDrone;
    double minStartGoalDistanceRatio;
    double dynamicObstacleWeight;
    std::string dynamicsExportTriggerTopic;
    std::string dynamicsExportFile;
    double dynamicsSampleDt;

    template <typename T>
    static T declareAndGet(const rclcpp::Node::SharedPtr &node,
                           const std::string &name,
                           const T &defaultValue)
    {
        node->declare_parameter<T>(name, defaultValue);
        return node->get_parameter(name).get_value<T>();
    }

    explicit Config(const rclcpp::Node::SharedPtr &node)
    {
        mapTopic = declareAndGet<std::string>(node, "MapTopic", "/voxel_map");
        targetTopic = declareAndGet<std::string>(node, "TargetTopic", "/move_base_simple/goal");
        dilateRadius = declareAndGet<double>(node, "DilateRadius", 0.5);
        voxelWidth = declareAndGet<double>(node, "VoxelWidth", 0.25);
        mapBound = declareAndGet<std::vector<double>>(node, "MapBound", {-25.0, 25.0, -25.0, 25.0, 0.0, 5.0});
        timeoutRRT = declareAndGet<double>(node, "TimeoutRRT", 0.02);
        maxVelMag = declareAndGet<double>(node, "MaxVelMag", 4.0);
        maxBdrMag = declareAndGet<double>(node, "MaxBdrMag", 2.1);
        maxTiltAngle = declareAndGet<double>(node, "MaxTiltAngle", 1.05);
        minThrust = declareAndGet<double>(node, "MinThrust", 2.0);
        maxThrust = declareAndGet<double>(node, "MaxThrust", 12.0);
        vehicleMass = declareAndGet<double>(node, "VehicleMass", 0.61);
        gravAcc = declareAndGet<double>(node, "GravAcc", 9.8);
        horizDrag = declareAndGet<double>(node, "HorizDrag", 0.70);
        vertDrag = declareAndGet<double>(node, "VertDrag", 0.80);
        parasDrag = declareAndGet<double>(node, "ParasDrag", 0.01);
        speedEps = declareAndGet<double>(node, "SpeedEps", 0.0001);
        weightT = declareAndGet<double>(node, "WeightT", 20.0);
        chiVec = declareAndGet<std::vector<double>>(node, "ChiVec", {1.0e+4, 1.0e+4, 1.0e+4, 1.0e+4, 1.0e+5});
        smoothingEps = declareAndGet<double>(node, "SmoothingEps", 1.0e-2);
        integralIntervs = declareAndGet<int>(node, "IntegralIntervs", 16);
        relCostTol = declareAndGet<double>(node, "RelCostTol", 1.0e-5);
        droneCount = declareAndGet<int>(node, "DroneCount", 0);
        maxSamplingAttempts = declareAndGet<int>(node, "MaxSamplingAttempts", 200);
        maxPlanningAttemptsPerDrone = declareAndGet<int>(node, "MaxPlanningAttemptsPerDrone", 30);
        minStartGoalDistanceRatio = declareAndGet<double>(node, "MinStartGoalDistanceRatio", 0.5);
        dynamicObstacleWeight = declareAndGet<double>(node, "DynamicObstacleWeight", 2.0e4);
        dynamicsExportTriggerTopic = declareAndGet<std::string>(node, "DynamicsExportTriggerTopic", "/gcopter/save_dynamics_trigger");
        dynamicsExportFile = declareAndGet<std::string>(node, "DynamicsExportFile",
                                                        "/home/zdp/CodeField/GCOPTER/src/GCOPTER/gcopter/scripts/latest_trajectory_coefficients.json");
        dynamicsSampleDt = declareAndGet<double>(node, "DynamicsSampleDt", 0.01);
    }
};

class GlobalPlanner
{
private:
    struct PlannedTrajectory
    {
        int droneId = -1;
        Eigen::Vector3d start = Eigen::Vector3d::Zero();
        Eigen::Vector3d goal = Eigen::Vector3d::Zero();
        SplineTrajectory::QuinticSpline3D traj;
        std::vector<Eigen::Vector3d> route;
        std::vector<Eigen::MatrixX4d> corridor;
        double trajStamp = 0.0;
    };

    Config config;

    rclcpp::Node::SharedPtr node;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr mapSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr targetSub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr dynamicsTriggerSub;

    bool mapInitialized;
    bool chainPlanningStarted;
    bool chainPlanningCompleted;
    bool exportRequested;

    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;
    std::vector<PlannedTrajectory> plannedTrajectories;
    double swarmAnimationStamp;
    double lastAnimationPublish;
    std::mt19937 rng;

    static std::string quoteForJson(const std::string &text)
    {
        std::ostringstream oss;
        oss << '"';
        for (const char ch : text)
        {
            switch (ch)
            {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            default:
                oss << ch;
                break;
            }
        }
        oss << '"';
        return oss.str();
    }

    static void writeStdVector(std::ostream &ofs,
                               const std::vector<double> &values)
    {
        ofs << '[';
        for (size_t i = 0; i < values.size(); ++i)
        {
            if (i > 0)
            {
                ofs << ", ";
            }
            ofs << std::setprecision(17) << values[i];
        }
        ofs << ']';
    }

    static void writeEigenVector(std::ostream &ofs,
                                 const Eigen::Vector3d &vec)
    {
        ofs << '['
            << std::setprecision(17) << vec.x() << ", "
            << std::setprecision(17) << vec.y() << ", "
            << std::setprecision(17) << vec.z() << ']';
    }

    static void writeEigenVectorList(std::ostream &ofs,
                                     const std::vector<Eigen::Vector3d> &points)
    {
        ofs << '[';
        for (size_t i = 0; i < points.size(); ++i)
        {
            if (i > 0)
            {
                ofs << ", ";
            }
            writeEigenVector(ofs, points[i]);
        }
        ofs << ']';
    }

    static void writeEigenMatrix(std::ostream &ofs,
                                 const Eigen::MatrixXd &mat)
    {
        ofs << '[';
        for (Eigen::Index row = 0; row < mat.rows(); ++row)
        {
            if (row > 0)
            {
                ofs << ",\n        ";
            }
            ofs << '[';
            for (Eigen::Index col = 0; col < mat.cols(); ++col)
            {
                if (col > 0)
                {
                    ofs << ", ";
                }
                ofs << std::setprecision(17) << mat(row, col);
            }
            ofs << ']';
        }
        ofs << ']';
    }

    std::array<float, 3> colorForId(const int id) const
    {
        static const std::array<std::array<float, 3>, 8> kPalette = {{
            {0.10f, 0.60f, 0.95f},
            {0.95f, 0.40f, 0.20f},
            {0.20f, 0.75f, 0.35f},
            {0.95f, 0.75f, 0.15f},
            {0.70f, 0.35f, 0.95f},
            {0.15f, 0.80f, 0.80f},
            {0.95f, 0.25f, 0.60f},
            {0.55f, 0.55f, 0.55f},
        }};
        return kPalette[static_cast<size_t>(id) % kPalette.size()];
    }

    std::vector<std::array<float, 3>> buildSwarmColors(const size_t count) const
    {
        std::vector<std::array<float, 3>> colors;
        colors.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            colors.push_back(colorForId(static_cast<int>(i)));
        }
        return colors;
    }

    std::vector<Eigen::Vector3d> sampleExecutedTrail(const PlannedTrajectory &planned,
                                                     const double delta) const
    {
        std::vector<Eigen::Vector3d> trail;
        if (!planned.traj.isInitialized() || planned.traj.getNumSegments() <= 0)
        {
            return trail;
        }

        const double query_time = std::clamp(delta, 0.0, planned.traj.getDuration());
        const double sample_dt = std::max(0.02, config.dynamicsSampleDt);
        trail.push_back(planned.traj.getTrajectory().evaluate(0.0, SplineTrajectory::Deriv::Pos));
        for (double t = sample_dt; t < query_time; t += sample_dt)
        {
            trail.push_back(planned.traj.getTrajectory().evaluate(t, SplineTrajectory::Deriv::Pos));
        }
        if (query_time > 0.0)
        {
            trail.push_back(planned.traj.getTrajectory().evaluate(query_time, SplineTrajectory::Deriv::Pos));
        }

        return trail;
    }

    Eigen::Vector3d dynamicObstacleEllipsoid() const
    {
        const double relative_radius = 2.0 * config.dilateRadius;
        return Eigen::Vector3d(relative_radius, relative_radius, relative_radius);
    }

    double minStartGoalDistance() const
    {
        const double x_len = config.mapBound[1] - config.mapBound[0];
        const double y_len = config.mapBound[3] - config.mapBound[2];
        return config.minStartGoalDistanceRatio * std::min(x_len, y_len);
    }

    bool violatesDynamicClearance(const Eigen::Vector3d &lhs,
                                  const Eigen::Vector3d &rhs) const
    {
        const Eigen::Array3d normalized =
            (lhs - rhs).array() / dynamicObstacleEllipsoid().array().max(1.0e-3);
        return normalized.matrix().squaredNorm() < 1.0;
    }

    bool isEndpointAvailable(const Eigen::Vector3d &point) const
    {
        if (voxelMap.query(point) != voxel_map::Unoccupied)
        {
            return false;
        }

        for (const auto &planned : plannedTrajectories)
        {
            if (violatesDynamicClearance(point, planned.start) ||
                violatesDynamicClearance(point, planned.goal))
            {
                return false;
            }
        }

        return true;
    }

    Eigen::Vector3d sampleFreePoint()
    {
        const double x_low = config.mapBound[0] + config.dilateRadius;
        const double x_high = config.mapBound[1] - config.dilateRadius;
        const double y_low = config.mapBound[2] + config.dilateRadius;
        const double y_high = config.mapBound[3] - config.dilateRadius;
        const double z_low = std::min(config.mapBound[4] + config.dilateRadius,
                                      config.mapBound[5] - config.dilateRadius);
        const double z_high = std::max(config.mapBound[4] + config.dilateRadius,
                                       config.mapBound[5] - config.dilateRadius);

        std::uniform_real_distribution<double> x_dist(x_low, x_high);
        std::uniform_real_distribution<double> y_dist(y_low, y_high);
        std::uniform_real_distribution<double> z_dist(z_low, z_high);

        return Eigen::Vector3d(x_dist(rng), y_dist(rng), z_dist(rng));
    }

    bool sampleStartGoalPair(Eigen::Vector3d &start,
                             Eigen::Vector3d &goal)
    {
        for (int attempt = 1; attempt <= config.maxSamplingAttempts; ++attempt)
        {
            start = sampleFreePoint();
            goal = sampleFreePoint();

            if (!isEndpointAvailable(start) || !isEndpointAvailable(goal))
            {
                continue;
            }

            if (violatesDynamicClearance(start, goal))
            {
                continue;
            }

            if ((goal - start).norm() <= minStartGoalDistance())
            {
                continue;
            }

            return true;
        }

        return false;
    }

    std::vector<gcopter::DynamicObstacleTrajectory> buildDynamicObstacles() const
    {
        std::vector<gcopter::DynamicObstacleTrajectory> obstacles;
        obstacles.reserve(plannedTrajectories.size());
        for (const auto &planned : plannedTrajectories)
        {
            if (!planned.traj.isInitialized() || planned.traj.getNumSegments() <= 0)
            {
                continue;
            }
            gcopter::DynamicObstacleTrajectory obstacle;
            obstacle.drone_id = planned.droneId;
            obstacle.start_time = 0.0;
            obstacle.traj = planned.traj;
            obstacles.push_back(obstacle);
        }
        return obstacles;
    }

    bool solveTrajectory(const int drone_id,
                         const Eigen::Vector3d &start,
                         const Eigen::Vector3d &goal,
                         const std::vector<gcopter::DynamicObstacleTrajectory> &dynamicObstacles,
                         PlannedTrajectory &planned)
    {
        std::vector<Eigen::Vector3d> route;
        sfc_gen::planPath<voxel_map::VoxelMap>(start,
                                               goal,
                                               voxelMap.getOrigin(),
                                               voxelMap.getCorner(),
                                               &voxelMap,
                                               config.timeoutRRT,
                                               route);
        if (route.size() <= 1)
        {
            return false;
        }

        std::vector<Eigen::MatrixX4d> hPolys;
        std::vector<Eigen::Vector3d> pc;
        voxelMap.getSurf(pc);

        sfc_gen::convexCover(route,
                             pc,
                             voxelMap.getOrigin(),
                             voxelMap.getCorner(),
                             7.0,
                             3.0,
                             hPolys);
        sfc_gen::shortCut(hPolys);
        if (hPolys.empty())
        {
            return false;
        }

        Eigen::Matrix3d iniState;
        Eigen::Matrix3d finState;
        iniState << start, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        finState << goal, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

        gcopter::SplineSFCOptimizer gcopter;

        Eigen::VectorXd magnitudeBounds(5);
        Eigen::VectorXd penaltyWeights(5);
        Eigen::VectorXd physicalParams(6);
        magnitudeBounds(0) = config.maxVelMag;
        magnitudeBounds(1) = config.maxBdrMag;
        magnitudeBounds(2) = config.maxTiltAngle;
        magnitudeBounds(3) = config.minThrust;
        magnitudeBounds(4) = config.maxThrust;
        penaltyWeights(0) = config.chiVec[0];
        penaltyWeights(1) = config.chiVec[1];
        penaltyWeights(2) = config.chiVec[2];
        penaltyWeights(3) = config.chiVec[3];
        penaltyWeights(4) = config.chiVec[4];
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        planned.traj = SplineTrajectory::QuinticSpline3D();
        if (!gcopter.setup(config.weightT,
                           iniState,
                           finState,
                           hPolys,
                           INFINITY,
                           config.smoothingEps,
                           config.integralIntervs,
                           magnitudeBounds,
                           penaltyWeights,
                           physicalParams,
                           dynamicObstacles.empty() ? nullptr : &dynamicObstacles,
                           config.dynamicObstacleWeight,
                           dynamicObstacleEllipsoid(),
                           drone_id))
        {
            return false;
        }

        const double optimum = gcopter.optimize(planned.traj, config.relCostTol);
        if (!std::isfinite(optimum) ||
            !planned.traj.isInitialized() ||
            planned.traj.getNumSegments() <= 0)
        {
            return false;
        }

        planned.droneId = drone_id;
        planned.start = start;
        planned.goal = goal;
        planned.route = route;
        planned.corridor = hPolys;
        planned.trajStamp = node->now().seconds();
        return true;
    }

    void visualizePlannedSwarm()
    {
        visualizer.clearTrajectoryVisuals();
        visualizer.clearStartGoalVisuals();
        visualizer.clearAnimationFrame(static_cast<int>(plannedTrajectories.size()) + 32);

        std::vector<Eigen::MatrixX4d> allCorridors;
        for (const auto &planned : plannedTrajectories)
        {
            allCorridors.insert(allCorridors.end(), planned.corridor.begin(), planned.corridor.end());
        }
        if (!allCorridors.empty())
        {
            visualizer.visualizePolytope(allCorridors);
        }

        for (size_t i = 0; i < plannedTrajectories.size(); ++i)
        {
            const auto color = colorForId(static_cast<int>(i));
            const auto &planned = plannedTrajectories[i];
            visualizer.visualize(planned.traj, planned.route, planned.droneId, color, true, true);
            visualizer.visualizeStartGoal(planned.start, config.dilateRadius * 0.9, 2 * planned.droneId, false, color);
            visualizer.visualizeStartGoal(planned.goal, config.dilateRadius * 0.9, 2 * planned.droneId + 1, false, color);
        }

        swarmAnimationStamp = node->now().seconds();
        lastAnimationPublish = 0.0;
    }

    bool writeTrajectoryExport(const std::filesystem::path &coeffFile) const
    {
        if (plannedTrajectories.empty())
        {
            return false;
        }

        std::ofstream ofs(coeffFile);
        if (!ofs.is_open())
        {
            RCLCPP_ERROR(node->get_logger(), "Failed to open coefficient export file: %s", coeffFile.c_str());
            return false;
        }

        const Eigen::Vector3d ellipsoid = dynamicObstacleEllipsoid();

        ofs << "{\n";
        ofs << "  \"format_version\": 2,\n";
        ofs << "  \"export_type\": \"gcopter_chain_swarm\",\n";
        ofs << "  \"generated_at_ros_time\": " << std::setprecision(17) << node->now().seconds() << ",\n";
        ofs << "  \"trajectory_count\": " << plannedTrajectories.size() << ",\n";
        ofs << "  \"dynamics_export_trigger_topic\": " << quoteForJson(config.dynamicsExportTriggerTopic) << ",\n";
        ofs << "  \"dynamic_obstacle\": {\n";
        ofs << "    \"weight\": " << std::setprecision(17) << config.dynamicObstacleWeight << ",\n";
        ofs << "    \"relative_ellipsoid\": ";
        writeEigenVector(ofs, ellipsoid);
        ofs << "\n  },\n";
        ofs << "  \"constraints\": {\n";
        ofs << "    \"MaxVelMag\": " << std::setprecision(17) << config.maxVelMag << ",\n";
        ofs << "    \"MaxBdrMag\": " << std::setprecision(17) << config.maxBdrMag << ",\n";
        ofs << "    \"MaxTiltAngle\": " << std::setprecision(17) << config.maxTiltAngle << ",\n";
        ofs << "    \"MinThrust\": " << std::setprecision(17) << config.minThrust << ",\n";
        ofs << "    \"MaxThrust\": " << std::setprecision(17) << config.maxThrust << "\n";
        ofs << "  },\n";
        ofs << "  \"physical_params\": {\n";
        ofs << "    \"VehicleMass\": " << std::setprecision(17) << config.vehicleMass << ",\n";
        ofs << "    \"GravAcc\": " << std::setprecision(17) << config.gravAcc << ",\n";
        ofs << "    \"HorizDrag\": " << std::setprecision(17) << config.horizDrag << ",\n";
        ofs << "    \"VertDrag\": " << std::setprecision(17) << config.vertDrag << ",\n";
        ofs << "    \"ParasDrag\": " << std::setprecision(17) << config.parasDrag << ",\n";
        ofs << "    \"SpeedEps\": " << std::setprecision(17) << config.speedEps << "\n";
        ofs << "  },\n";
        ofs << "  \"sampling\": {\n";
        ofs << "    \"dt\": " << std::setprecision(17) << config.dynamicsSampleDt << "\n";
        ofs << "  },\n";
        ofs << "  \"trajectories\": [\n";

        for (size_t idx = 0; idx < plannedTrajectories.size(); ++idx)
        {
            const auto &planned = plannedTrajectories[idx];
            const auto &ppoly = planned.traj.getTrajectory();

            ofs << "    {\n";
            ofs << "      \"drone_id\": " << planned.droneId << ",\n";
            ofs << "      \"trajectory_stamp\": " << std::setprecision(17) << planned.trajStamp << ",\n";
            ofs << "      \"duration\": " << std::setprecision(17) << planned.traj.getDuration() << ",\n";
            ofs << "      \"start\": ";
            writeEigenVector(ofs, planned.start);
            ofs << ",\n";
            ofs << "      \"goal\": ";
            writeEigenVector(ofs, planned.goal);
            ofs << ",\n";
            ofs << "      \"route\": ";
            writeEigenVectorList(ofs, planned.route);
            ofs << ",\n";
            ofs << "      \"num_segments\": " << ppoly.getNumSegments() << ",\n";
            ofs << "      \"num_coefficients\": " << ppoly.getNumCoeffs() << ",\n";
            ofs << "      \"breakpoints\": ";
            writeStdVector(ofs, ppoly.getBreakpoints());
            ofs << ",\n";
            ofs << "      \"coefficients\": ";
            writeEigenMatrix(ofs, ppoly.getCoefficients());
            ofs << "\n";
            ofs << "    }";
            if (idx + 1 < plannedTrajectories.size())
            {
                ofs << ",";
            }
            ofs << "\n";
        }

        ofs << "  ]\n";
        ofs << "}\n";
        return true;
    }

    bool exportTrajectoryArtifacts()
    {
        if (plannedTrajectories.empty())
        {
            return false;
        }

        std::error_code ec;
        const std::filesystem::path coeffFile(config.dynamicsExportFile);
        const std::filesystem::path exportDir = coeffFile.parent_path();
        std::filesystem::create_directories(exportDir, ec);
        if (ec)
        {
            RCLCPP_ERROR(node->get_logger(), "Failed to create export directory %s: %s",
                         exportDir.c_str(), ec.message().c_str());
            return false;
        }

        if (!writeTrajectoryExport(coeffFile))
        {
            return false;
        }

        RCLCPP_INFO(node->get_logger(),
                    "Exported %zu planned trajectories to: %s",
                    plannedTrajectories.size(),
                    coeffFile.c_str());
        return true;
    }

    bool runChainPlanning()
    {
        plannedTrajectories.clear();
        const auto planning_begin = std::chrono::steady_clock::now();

        RCLCPP_INFO(node->get_logger(),
                    "Starting chain planning for %d drones. Dynamic-obstacle ellipsoid = [%.2f, %.2f, %.2f], min start-goal distance = %.2f m.",
                    config.droneCount,
                    dynamicObstacleEllipsoid().x(),
                    dynamicObstacleEllipsoid().y(),
                    dynamicObstacleEllipsoid().z(),
                    minStartGoalDistance());

        for (int drone_id = 0; drone_id < config.droneCount; ++drone_id)
        {
            const auto dynamicObstacles = buildDynamicObstacles();
            bool solved = false;
            for (int attempt = 1; attempt <= config.maxPlanningAttemptsPerDrone; ++attempt)
            {
                Eigen::Vector3d start;
                Eigen::Vector3d goal;
                if (!sampleStartGoalPair(start, goal))
                {
                    RCLCPP_ERROR(node->get_logger(),
                                 "Drone %d: failed to sample a valid start-goal pair after %d tries.",
                                 drone_id,
                                 config.maxSamplingAttempts);
                    return false;
                }

                RCLCPP_INFO(node->get_logger(),
                            "[%d/%d] Planning drone %d, attempt %d. start=(%.2f, %.2f, %.2f) goal=(%.2f, %.2f, %.2f)",
                            drone_id + 1,
                            config.droneCount,
                            drone_id,
                            attempt,
                            start.x(), start.y(), start.z(),
                            goal.x(), goal.y(), goal.z());

                PlannedTrajectory planned;
                if (solveTrajectory(drone_id, start, goal, dynamicObstacles, planned))
                {
                    plannedTrajectories.push_back(planned);
                    RCLCPP_INFO(node->get_logger(),
                                "Drone %d planned successfully. Progress: %zu/%d, duration=%.3f s, segments=%d.",
                                drone_id,
                                plannedTrajectories.size(),
                                config.droneCount,
                                planned.traj.getDuration(),
                                planned.traj.getNumSegments());
                    solved = true;
                    break;
                }

                RCLCPP_WARN(node->get_logger(),
                            "Drone %d planning failed on attempt %d. Resampling start/goal.",
                            drone_id,
                            attempt);
            }

            if (!solved)
            {
                RCLCPP_ERROR(node->get_logger(),
                             "Aborting chain planning: drone %d failed after %d planning attempts.",
                             drone_id,
                             config.maxPlanningAttemptsPerDrone);
                return false;
            }
        }

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_begin).count();
        RCLCPP_INFO(node->get_logger(),
                    "Chain planning finished. Planned %zu/%d drones in %.2f ms.",
                    plannedTrajectories.size(),
                    config.droneCount,
                    elapsed_ms);
        return true;
    }

    void publishPrimaryTrajectoryStats(const double delta)
    {
        if (plannedTrajectories.empty())
        {
            return;
        }

        const auto &planned = plannedTrajectories.front();
        if (!planned.traj.isInitialized() || planned.traj.getNumSegments() <= 0)
        {
            return;
        }

        const double clamped_delta = std::clamp(delta, 0.0, planned.traj.getDuration());
        if (clamped_delta <= 0.0)
        {
            return;
        }

        Eigen::VectorXd physicalParams(6);
        physicalParams(0) = config.vehicleMass;
        physicalParams(1) = config.gravAcc;
        physicalParams(2) = config.horizDrag;
        physicalParams(3) = config.vertDrag;
        physicalParams(4) = config.parasDrag;
        physicalParams(5) = config.speedEps;

        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        double thr;
        Eigen::Vector4d quat;
        Eigen::Vector3d omg;
        const auto &ppoly = planned.traj.getTrajectory();
        const Eigen::Vector3d vel = ppoly.evaluate(clamped_delta, SplineTrajectory::Deriv::Vel);
        const Eigen::Vector3d acc = ppoly.evaluate(clamped_delta, SplineTrajectory::Deriv::Acc);
        const Eigen::Vector3d jer = ppoly.evaluate(clamped_delta, SplineTrajectory::Deriv::Jerk);

        flatmap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);
        std_msgs::msg::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
        speedMsg.data = vel.norm();
        thrMsg.data = thr;
        tiltMsg.data = std::acos(std::clamp(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)), -1.0, 1.0));
        bdrMsg.data = omg.norm();
        visualizer.speedPub->publish(speedMsg);
        visualizer.thrPub->publish(thrMsg);
        visualizer.tiltPub->publish(tiltMsg);
        visualizer.bdrPub->publish(bdrMsg);
    }

    void animateTrajectories()
    {
        if (plannedTrajectories.empty())
        {
            return;
        }

        const double now_sec = node->now().seconds();

        // Throttle animation to ~30 fps to prevent flooding RViz
        // and avoid cascading performance degradation from trail resampling.
        constexpr double kAnimationDt = 1.0 / 30.0;
        if (now_sec - lastAnimationPublish < kAnimationDt)
        {
            return;
        }
        lastAnimationPublish = now_sec;

        const double delta = now_sec - swarmAnimationStamp;
        if (delta < 0.0)
        {
            return;
        }

        std::vector<Eigen::Vector3d> positions;
        std::vector<std::vector<Eigen::Vector3d>> trails;
        const auto colors = buildSwarmColors(plannedTrajectories.size());
        positions.reserve(plannedTrajectories.size());
        trails.reserve(plannedTrajectories.size());
        for (const auto &planned : plannedTrajectories)
        {
            if (!planned.traj.isInitialized() || planned.traj.getNumSegments() <= 0)
            {
                continue;
            }

            const double query_time = std::clamp(delta, 0.0, planned.traj.getDuration());
            positions.push_back(planned.traj.getTrajectory().evaluate(query_time, SplineTrajectory::Deriv::Pos));
            trails.push_back(sampleExecutedTrail(planned, delta));
        }

        if (!positions.empty())
        {
            // Publish all bodies and trails as a single MarkerArray for
            // atomic update — guarantees all drones are rendered together.
            visualizer.visualizeAnimationFrame(positions, trails,
                                               config.dilateRadius, 0.16, colors);
            publishPrimaryTrajectoryStats(delta);
        }
    }

    void executeManualPlanning()
    {
        if (startGoal.size() != 2)
        {
            return;
        }

        PlannedTrajectory planned;
        const std::vector<gcopter::DynamicObstacleTrajectory> dynamicObstacles;
        if (!solveTrajectory(0, startGoal[0], startGoal[1], dynamicObstacles, planned))
        {
            RCLCPP_WARN(node->get_logger(), "Manual planning failed for the selected start/goal.");
            return;
        }

        plannedTrajectories.clear();
        plannedTrajectories.push_back(planned);
        chainPlanningCompleted = true;
        visualizePlannedSwarm();
        if (exportRequested && exportTrajectoryArtifacts())
        {
            exportRequested = false;
        }
    }

    inline void dynamicsTriggerCallBack(const std_msgs::msg::Bool::SharedPtr msg)
    {
        if (!msg->data)
        {
            return;
        }

        exportRequested = true;
        if (exportTrajectoryArtifacts())
        {
            exportRequested = false;
            return;
        }

        RCLCPP_INFO(node->get_logger(),
                    "Trajectory export requested. Waiting for valid planned trajectories before writing artifacts.");
    }

public:
    GlobalPlanner(const Config &conf,
                  const rclcpp::Node::SharedPtr &node_)
        : config(conf),
          node(node_),
          mapInitialized(false),
          chainPlanningStarted(false),
          chainPlanningCompleted(false),
          exportRequested(false),
          visualizer(node),
          swarmAnimationStamp(0.0),
          lastAnimationPublish(0.0),
          rng(std::random_device{}())
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);
        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
            config.mapTopic,
            rclcpp::QoS(1),
            std::bind(&GlobalPlanner::mapCallBack, this, std::placeholders::_1));

        targetSub = node->create_subscription<geometry_msgs::msg::PoseStamped>(
            config.targetTopic,
            rclcpp::QoS(1),
            std::bind(&GlobalPlanner::targetCallBack, this, std::placeholders::_1));

        dynamicsTriggerSub = node->create_subscription<std_msgs::msg::Bool>(
            config.dynamicsExportTriggerTopic,
            rclcpp::QoS(1),
            std::bind(&GlobalPlanner::dynamicsTriggerCallBack, this, std::placeholders::_1));
    }

    inline void mapCallBack(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (mapInitialized)
        {
            return;
        }

        size_t cur = 0;
        const size_t total = msg->data.size() / msg->point_step;
        const float *fdata = reinterpret_cast<const float *>(msg->data.data());
        for (size_t i = 0; i < total; i++)
        {
            cur = msg->point_step / sizeof(float) * i;

            if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
            {
                continue;
            }
            voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                 fdata[cur + 1],
                                                 fdata[cur + 2]));
        }

        voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));
        mapInitialized = true;
        RCLCPP_INFO(node->get_logger(), "Voxel map initialized. Planning mode: %s",
                    config.droneCount > 0 ? "chain swarm" : "manual target");
    }

    inline void targetCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (!mapInitialized)
        {
            return;
        }

        if (config.droneCount > 0)
        {
            RCLCPP_INFO(node->get_logger(),
                        "Ignoring manual target because DroneCount=%d enables chain planning.",
                        config.droneCount);
            return;
        }

        if (startGoal.size() >= 2)
        {
            startGoal.clear();
            visualizer.clearStartGoalVisuals();
        }

        const double zGoal = config.mapBound[4] + config.dilateRadius +
                             std::fabs(msg->pose.orientation.z) *
                                 (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
        const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
        if (voxelMap.query(goal) == voxel_map::Unoccupied)
        {
            startGoal.emplace_back(goal);
            visualizer.visualizeStartGoal(goal,
                                          config.dilateRadius * 0.9,
                                          static_cast<int>(startGoal.size() - 1),
                                          false);
            executeManualPlanning();
        }
        else
        {
            RCLCPP_WARN(node->get_logger(), "Infeasible position selected.");
        }
    }

    inline void process()
    {
        if (mapInitialized && config.droneCount > 0 && !chainPlanningStarted)
        {
            chainPlanningStarted = true;
            chainPlanningCompleted = runChainPlanning();
            if (chainPlanningCompleted)
            {
                visualizePlannedSwarm();
                if (exportRequested && exportTrajectoryArtifacts())
                {
                    exportRequested = false;
                }
            }
        }

        if (chainPlanningCompleted)
        {
            animateTrajectories();
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("global_planning_node");

    GlobalPlanner global_planner(Config(node), node);

    rclcpp::Rate lr(1000.0);
    while (rclcpp::ok())
    {
        global_planner.process();
        rclcpp::spin_some(node);
        lr.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
