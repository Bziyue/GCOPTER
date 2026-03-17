#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "SplineTrajectory/SplineTrajectory.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/quickhull.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker.hpp>

class Visualizer
{
private:
    rclcpp::Node::SharedPtr node;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr routePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr wayPointsPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr trajectoryPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr meshPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edgePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr spherePub;

public:
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr speedPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr thrPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr tiltPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr bdrPub;

public:
    explicit Visualizer(const rclcpp::Node::SharedPtr &node_)
        : node(node_)
    {
        routePub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/route", 10);
        wayPointsPub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/waypoints", 10);
        trajectoryPub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/trajectory", 10);
        meshPub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/mesh", 1000);
        edgePub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/edge", 1000);
        spherePub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/spheres", 1000);
        speedPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/speed", 1000);
        thrPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/total_thrust", 1000);
        tiltPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/tilt_angle", 1000);
        bdrPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/body_rate", 1000);
    }

    inline void visualize(const SplineTrajectory::QuinticSpline3D &spline,
                          const std::vector<Eigen::Vector3d> &route)
    {
        visualization_msgs::msg::Marker routeMarker, wayPointsMarker, trajMarker;

        routeMarker.id = 0;
        routeMarker.type = visualization_msgs::msg::Marker::LINE_LIST;
        routeMarker.header.stamp = node->get_clock()->now();
        routeMarker.header.frame_id = "odom";
        routeMarker.pose.orientation.w = 1.00;
        routeMarker.action = visualization_msgs::msg::Marker::ADD;
        routeMarker.ns = "route";
        routeMarker.color.r = 1.00;
        routeMarker.color.g = 0.00;
        routeMarker.color.b = 0.00;
        routeMarker.color.a = 1.00;
        routeMarker.scale.x = 0.1;

        wayPointsMarker = routeMarker;
        wayPointsMarker.id = -wayPointsMarker.id - 1;
        wayPointsMarker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        wayPointsMarker.ns = "waypoints";
        wayPointsMarker.color.r = 1.00;
        wayPointsMarker.color.g = 0.00;
        wayPointsMarker.color.b = 0.00;
        wayPointsMarker.scale.x = 0.35;
        wayPointsMarker.scale.y = 0.35;
        wayPointsMarker.scale.z = 0.35;

        trajMarker = routeMarker;
        trajMarker.header.frame_id = "odom";
        trajMarker.id = 0;
        trajMarker.ns = "trajectory";
        trajMarker.color.r = 0.00;
        trajMarker.color.g = 0.50;
        trajMarker.color.b = 1.00;
        trajMarker.scale.x = 0.30;

        if (!route.empty())
        {
            bool first = true;
            Eigen::Vector3d last;
            for (const auto &it : route)
            {
                if (first)
                {
                    first = false;
                    last = it;
                    continue;
                }
                geometry_msgs::msg::Point point;

                point.x = last(0);
                point.y = last(1);
                point.z = last(2);
                routeMarker.points.push_back(point);
                point.x = it(0);
                point.y = it(1);
                point.z = it(2);
                routeMarker.points.push_back(point);
                last = it;
            }

            routePub->publish(routeMarker);
        }

        if (spline.isInitialized() && spline.getNumSegments() > 0)
        {
            const auto &wps = spline.getSpacePoints();
            for (int i = 0; i < wps.rows(); i++)
            {
                geometry_msgs::msg::Point point;
                point.x = wps(i, 0);
                point.y = wps(i, 1);
                point.z = wps(i, 2);
                wayPointsMarker.points.push_back(point);
            }

            wayPointsPub->publish(wayPointsMarker);
        }

        if (spline.isInitialized() && spline.getNumSegments() > 0)
        {
            const double T = 0.01;
            const double t_start = spline.getStartTime();
            const double t_end = spline.getEndTime();
            Eigen::Vector3d lastX = spline.getTrajectory().evaluate(t_start);
            for (double t = t_start + T; t < t_end; t += T)
            {
                geometry_msgs::msg::Point point;
                Eigen::Vector3d X = spline.getTrajectory().evaluate(t);
                point.x = lastX(0);
                point.y = lastX(1);
                point.z = lastX(2);
                trajMarker.points.push_back(point);
                point.x = X(0);
                point.y = X(1);
                point.z = X(2);
                trajMarker.points.push_back(point);
                lastX = X;
            }
            trajectoryPub->publish(trajMarker);
        }
    }

    inline void visualizePolytope(const traj_opt_components::PolyhedraH &hPolys)
    {
        Eigen::Matrix3Xd mesh(3, 0), curTris(3, 0), oldTris(3, 0);
        for (size_t id = 0; id < hPolys.size(); id++)
        {
            oldTris = mesh;
            Eigen::Matrix<double, 3, -1, Eigen::ColMajor> vPoly;
            geo_utils::enumerateVs(hPolys[id], vPoly);

            quickhull::QuickHull<double> tinyQH;
            const auto polyHull = tinyQH.getConvexHull(vPoly.data(), vPoly.cols(), false, true);
            const auto &idxBuffer = polyHull.getIndexBuffer();
            int hNum = idxBuffer.size() / 3;

            curTris.resize(3, hNum * 3);
            for (int i = 0; i < hNum * 3; i++)
            {
                curTris.col(i) = vPoly.col(idxBuffer[i]);
            }
            mesh.resize(3, oldTris.cols() + curTris.cols());
            mesh.leftCols(oldTris.cols()) = oldTris;
            mesh.rightCols(curTris.cols()) = curTris;
        }

        visualization_msgs::msg::Marker meshMarker, edgeMarker;

        meshMarker.id = 0;
        meshMarker.header.stamp = node->get_clock()->now();
        meshMarker.header.frame_id = "odom";
        meshMarker.pose.orientation.w = 1.00;
        meshMarker.action = visualization_msgs::msg::Marker::ADD;
        meshMarker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        meshMarker.ns = "mesh";
        meshMarker.color.r = 0.00;
        meshMarker.color.g = 0.00;
        meshMarker.color.b = 1.00;
        meshMarker.color.a = 0.15;
        meshMarker.scale.x = 1.0;
        meshMarker.scale.y = 1.0;
        meshMarker.scale.z = 1.0;

        edgeMarker = meshMarker;
        edgeMarker.type = visualization_msgs::msg::Marker::LINE_LIST;
        edgeMarker.ns = "edge";
        edgeMarker.color.r = 0.00;
        edgeMarker.color.g = 1.00;
        edgeMarker.color.b = 1.00;
        edgeMarker.color.a = 1.00;
        edgeMarker.scale.x = 0.02;

        geometry_msgs::msg::Point point;

        int ptnum = mesh.cols();

        for (int i = 0; i < ptnum; i++)
        {
            point.x = mesh(0, i);
            point.y = mesh(1, i);
            point.z = mesh(2, i);
            meshMarker.points.push_back(point);
        }

        for (int i = 0; i < ptnum / 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                point.x = mesh(0, 3 * i + j);
                point.y = mesh(1, 3 * i + j);
                point.z = mesh(2, 3 * i + j);
                edgeMarker.points.push_back(point);
                point.x = mesh(0, 3 * i + (j + 1) % 3);
                point.y = mesh(1, 3 * i + (j + 1) % 3);
                point.z = mesh(2, 3 * i + (j + 1) % 3);
                edgeMarker.points.push_back(point);
            }
        }

        meshPub->publish(meshMarker);
        edgePub->publish(edgeMarker);
    }

    inline void visualizeSphere(const Eigen::Vector3d &center,
                                const double &radius)
    {
        visualization_msgs::msg::Marker sphereMarkers, sphereDeleter;

        sphereMarkers.id = 0;
        sphereMarkers.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        sphereMarkers.header.stamp = node->get_clock()->now();
        sphereMarkers.header.frame_id = "odom";
        sphereMarkers.pose.orientation.w = 1.00;
        sphereMarkers.action = visualization_msgs::msg::Marker::ADD;
        sphereMarkers.ns = "spheres";
        sphereMarkers.color.r = 0.00;
        sphereMarkers.color.g = 0.00;
        sphereMarkers.color.b = 1.00;
        sphereMarkers.color.a = 1.00;
        sphereMarkers.scale.x = radius * 2.0;
        sphereMarkers.scale.y = radius * 2.0;
        sphereMarkers.scale.z = radius * 2.0;

        sphereDeleter = sphereMarkers;
        sphereDeleter.action = visualization_msgs::msg::Marker::DELETE;

        geometry_msgs::msg::Point point;
        point.x = center(0);
        point.y = center(1);
        point.z = center(2);
        sphereMarkers.points.push_back(point);

        spherePub->publish(sphereDeleter);
        spherePub->publish(sphereMarkers);
    }

    inline void visualizeStartGoal(const Eigen::Vector3d &center,
                                   const double &radius,
                                   int sg)
    {
        visualization_msgs::msg::Marker sphereMarkers, sphereDeleter;

        sphereMarkers.id = sg;
        sphereMarkers.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        sphereMarkers.header.stamp = node->get_clock()->now();
        sphereMarkers.header.frame_id = "odom";
        sphereMarkers.pose.orientation.w = 1.00;
        sphereMarkers.action = visualization_msgs::msg::Marker::ADD;
        sphereMarkers.ns = "StartGoal";
        sphereMarkers.color.r = 1.00;
        sphereMarkers.color.g = 0.00;
        sphereMarkers.color.b = 0.00;
        sphereMarkers.color.a = 1.00;
        sphereMarkers.scale.x = radius * 2.0;
        sphereMarkers.scale.y = radius * 2.0;
        sphereMarkers.scale.z = radius * 2.0;

        sphereDeleter = sphereMarkers;
        sphereDeleter.action = visualization_msgs::msg::Marker::DELETEALL;

        geometry_msgs::msg::Point point;
        point.x = center(0);
        point.y = center(1);
        point.z = center(2);
        sphereMarkers.points.push_back(point);

        if (sg == 0)
        {
            spherePub->publish(sphereDeleter);
            rclcpp::sleep_for(std::chrono::nanoseconds(1));
            sphereMarkers.header.stamp = node->get_clock()->now();
        }
        spherePub->publish(sphereMarkers);
    }
};

#endif
