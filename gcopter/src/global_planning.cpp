#include "misc/visualizer.hpp"
#include "gcopter/spline_sfc_optimizer.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

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

    template <typename T>
    static T getParamOrDefault(ros::NodeHandle &node,
                               const std::string &name,
                               const T &defaultValue)
    {
        T value;
        if (!node.getParam(name, value))
        {
            value = defaultValue;
            node.setParam(name, value);
        }
        return value;
    }

    explicit Config(ros::NodeHandle &node)
    {
        mapTopic = getParamOrDefault<std::string>(node, "MapTopic", "/voxel_map");
        targetTopic = getParamOrDefault<std::string>(node, "TargetTopic", "/move_base_simple/goal");
        dilateRadius = getParamOrDefault<double>(node, "DilateRadius", 0.5);
        voxelWidth = getParamOrDefault<double>(node, "VoxelWidth", 0.25);
        mapBound = getParamOrDefault<std::vector<double>>(node, "MapBound", {-25.0, 25.0, -25.0, 25.0, 0.0, 5.0});
        timeoutRRT = getParamOrDefault<double>(node, "TimeoutRRT", 0.02);
        maxVelMag = getParamOrDefault<double>(node, "MaxVelMag", 4.0);
        maxBdrMag = getParamOrDefault<double>(node, "MaxBdrMag", 2.1);
        maxTiltAngle = getParamOrDefault<double>(node, "MaxTiltAngle", 1.05);
        minThrust = getParamOrDefault<double>(node, "MinThrust", 2.0);
        maxThrust = getParamOrDefault<double>(node, "MaxThrust", 12.0);
        vehicleMass = getParamOrDefault<double>(node, "VehicleMass", 0.61);
        gravAcc = getParamOrDefault<double>(node, "GravAcc", 9.8);
        horizDrag = getParamOrDefault<double>(node, "HorizDrag", 0.70);
        vertDrag = getParamOrDefault<double>(node, "VertDrag", 0.80);
        parasDrag = getParamOrDefault<double>(node, "ParasDrag", 0.01);
        speedEps = getParamOrDefault<double>(node, "SpeedEps", 0.0001);
        weightT = getParamOrDefault<double>(node, "WeightT", 20.0);
        chiVec = getParamOrDefault<std::vector<double>>(node, "ChiVec", {1.0e+4, 1.0e+4, 1.0e+4, 1.0e+4, 1.0e+5});
        smoothingEps = getParamOrDefault<double>(node, "SmoothingEps", 1.0e-2);
        integralIntervs = getParamOrDefault<int>(node, "IntegralIntervs", 16);
        relCostTol = getParamOrDefault<double>(node, "RelCostTol", 1.0e-5);
    }
};

class GlobalPlanner
{
private:
    Config config;

    ros::NodeHandle node;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;

    SplineTrajectory::QuinticSpline3D traj;
    double trajStamp;

public:
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &node_)
        : config(conf),
          node(node_),
          mapInitialized(false),
          visualizer(node_),
          trajStamp(0.0)
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = node.subscribe(config.mapTopic, 1,
                                &GlobalPlanner::mapCallBack, this,
                                ros::TransportHints().tcpNoDelay());

        targetSub = node.subscribe(config.targetTopic, 1,
                                   &GlobalPlanner::targetCallBack, this,
                                   ros::TransportHints().tcpNoDelay());
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        if (!mapInitialized)
        {
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
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2)
        {
            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                                   startGoal[1],
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
            traj_opt_components::PolyhedraH hPolys;
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

            if (route.size() > 1)
            {
                visualizer.visualizePolytope(hPolys);

                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

                gcopter::SplineSFCOptimizer gcopter;

                Eigen::VectorXd magnitudeBounds(5);
                Eigen::VectorXd penaltyWeights(5);
                Eigen::VectorXd physicalParams(6);
                magnitudeBounds(0) = config.maxVelMag;
                magnitudeBounds(1) = config.maxBdrMag;
                magnitudeBounds(2) = config.maxTiltAngle;
                magnitudeBounds(3) = config.minThrust;
                magnitudeBounds(4) = config.maxThrust;
                penaltyWeights(0) = (config.chiVec)[0];
                penaltyWeights(1) = (config.chiVec)[1];
                penaltyWeights(2) = (config.chiVec)[2];
                penaltyWeights(3) = (config.chiVec)[3];
                penaltyWeights(4) = (config.chiVec)[4];
                physicalParams(0) = config.vehicleMass;
                physicalParams(1) = config.gravAcc;
                physicalParams(2) = config.horizDrag;
                physicalParams(3) = config.vertDrag;
                physicalParams(4) = config.parasDrag;
                physicalParams(5) = config.speedEps;
                const int quadratureRes = config.integralIntervs;

                traj = SplineTrajectory::QuinticSpline3D();

                if (!gcopter.setup(config.weightT,
                                   iniState, finState,
                                   hPolys, INFINITY,
                                   config.smoothingEps,
                                   quadratureRes,
                                   magnitudeBounds,
                                   penaltyWeights,
                                   physicalParams))
                {
                    return;
                }

                if (std::isinf(gcopter.optimize(traj, config.relCostTol)))
                {
                    return;
                }

                if (traj.isInitialized() && traj.getNumSegments() > 0)
                {
                    trajStamp = ros::Time::now().toSec();
                    visualizer.visualize(traj, route);
                }
            }
        }
    }

    inline void targetCallBack(const geometry_msgs::PoseStampedConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.5, static_cast<int>(startGoal.size()));
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!");
            }

            plan();
        }
    }

    inline void process()
    {
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

        if (traj.isInitialized() && traj.getNumSegments() > 0)
        {
            const double delta = ros::Time::now().toSec() - trajStamp;
            if (delta > 0.0 && delta < traj.getDuration())
            {
                double thr;
                Eigen::Vector4d quat;
                Eigen::Vector3d omg;

                const auto &ppoly = traj.getTrajectory();
                const Eigen::Vector3d vel = ppoly.evaluate(delta, SplineTrajectory::Deriv::Vel);
                const Eigen::Vector3d acc = ppoly.evaluate(delta, SplineTrajectory::Deriv::Acc);
                const Eigen::Vector3d jer = ppoly.evaluate(delta, SplineTrajectory::Deriv::Jerk);
                const Eigen::Vector3d pos = ppoly.evaluate(delta, SplineTrajectory::Deriv::Pos);

                flatmap.forward(vel,
                                acc,
                                jer,
                                0.0, 0.0,
                                thr, quat, omg);
                double speed = vel.norm();
                double bodyratemag = omg.norm();
                double tiltangle = acos(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)));
                std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
                speedMsg.data = speed;
                thrMsg.data = thr;
                tiltMsg.data = tiltangle;
                bdrMsg.data = bodyratemag;
                visualizer.speedPub.publish(speedMsg);
                visualizer.thrPub.publish(thrMsg);
                visualizer.tiltPub.publish(tiltMsg);
                visualizer.bdrPub.publish(bdrMsg);

                visualizer.visualizeSphere(pos,
                                           config.dilateRadius);
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle node;
    ros::NodeHandle privateNode("~");

    GlobalPlanner global_planner(Config(privateNode), node);

    ros::Rate lr(1000.0);
    while (ros::ok())
    {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}
