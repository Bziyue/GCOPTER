#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "SplineTrajectory/SplineTrajectory.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/quickhull.hpp"

#include <chrono>
#include <cmath>
#include <array>
#include <iostream>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class Visualizer
{
private:
    rclcpp::Node::SharedPtr node;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr routePub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr wayPointsPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr meshPub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr edgePub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr animationPub;

public:
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr speedPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr thrPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr tiltPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr bdrPub;

public:
    explicit Visualizer(const rclcpp::Node::SharedPtr &node_)
        : node(node_)
    {
        const auto marker_qos = rclcpp::QoS(100).reliable().transient_local();
        routePub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/route", marker_qos);
        wayPointsPub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/waypoints", marker_qos);
        meshPub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/mesh", marker_qos);
        edgePub = node->create_publisher<visualization_msgs::msg::Marker>("/visualizer/edge", marker_qos);
        speedPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/speed", 1000);
        thrPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/total_thrust", 1000);
        tiltPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/tilt_angle", 1000);
        bdrPub = node->create_publisher<std_msgs::msg::Float64>("/visualizer/body_rate", 1000);
        animationPub = node->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/visualizer/animation", rclcpp::QoS(100).reliable().transient_local());
    }

    inline void clearTrajectoryVisuals()
    {
        visualization_msgs::msg::Marker marker;
        marker.action = visualization_msgs::msg::Marker::DELETEALL;
        routePub->publish(marker);
        wayPointsPub->publish(marker);
    }

    inline void clearStartGoalVisuals()
    {
        visualization_msgs::msg::MarkerArray array;
        const auto stamp = node->get_clock()->now();
        constexpr int kMaxStartGoalMarkers = 256;
        for (int id = 0; id < kMaxStartGoalMarkers; ++id)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.stamp = stamp;
            marker.header.frame_id = "odom";
            marker.ns = "StartGoal";
            marker.id = id;
            marker.action = visualization_msgs::msg::Marker::DELETE;
            array.markers.push_back(std::move(marker));
        }
        animationPub->publish(array);
    }

    inline void visualize(const SplineTrajectory::QuinticSpline3D &spline,
                          const std::vector<Eigen::Vector3d> &route)
    {
        visualize(spline, route, 0, {0.0f, 0.5f, 1.0f}, true, true);
    }

    inline void visualize(const SplineTrajectory::QuinticSpline3D &spline,
                          const std::vector<Eigen::Vector3d> &route,
                          const int marker_id,
                          const std::array<float, 3> &traj_color,
                          const bool show_route = true,
                          const bool show_waypoints = true)
    {
        visualization_msgs::msg::Marker routeMarker, wayPointsMarker;

        routeMarker.id = marker_id;
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
        routeMarker.scale.x = 0.06;

        wayPointsMarker = routeMarker;
        wayPointsMarker.id = marker_id;
        wayPointsMarker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        wayPointsMarker.ns = "waypoints";
        wayPointsMarker.color.r = traj_color[0];
        wayPointsMarker.color.g = traj_color[1];
        wayPointsMarker.color.b = traj_color[2];
        wayPointsMarker.scale.x = 0.35;
        wayPointsMarker.scale.y = 0.35;
        wayPointsMarker.scale.z = 0.35;

        if (show_route && !route.empty())
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

        if (show_waypoints &&
            spline.isInitialized() && spline.getNumSegments() > 0)
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

    }

    inline void visualizePolytope(const std::vector<Eigen::MatrixX4d> &hPolys)
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

    inline void visualizeAnimationFrame(
        const std::vector<Eigen::Vector3d> &positions,
        const std::vector<double> &yaws,
        const std::vector<std::vector<Eigen::Vector3d>> &trails,
        const std::string &mesh_resource,
        const double mesh_scale,
        const double mesh_rotate_yaw_deg,
        const double trail_width,
        const std::vector<std::array<float, 3>> &colors)
    {
        visualization_msgs::msg::MarkerArray array;
        const auto stamp = node->get_clock()->now();
        const double yaw_offset = mesh_rotate_yaw_deg * M_PI / 180.0;

        for (size_t i = 0; i < positions.size(); ++i)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.stamp = stamp;
            marker.header.frame_id = "odom";
            marker.ns = "agent_body";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = positions[i].x();
            marker.pose.position.y = positions[i].y();
            marker.pose.position.z = positions[i].z();
            const double yaw = (i < yaws.size() ? yaws[i] : 0.0) + yaw_offset;
            marker.pose.orientation.w = std::cos(0.5 * yaw);
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = std::sin(0.5 * yaw);
            marker.scale.x = mesh_scale;
            marker.scale.y = mesh_scale;
            marker.scale.z = mesh_scale;
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            marker.mesh_resource = mesh_resource;
            marker.mesh_use_embedded_materials = false;
            array.markers.push_back(std::move(marker));
        }

        for (size_t i = 0; i < trails.size(); ++i)
        {
            visualization_msgs::msg::Marker marker;
            marker.header.stamp = stamp;
            marker.header.frame_id = "odom";
            marker.ns = "executed_trail";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = trail_width;
            const auto &color = colors[i % colors.size()];
            marker.color.r = color[0];
            marker.color.g = color[1];
            marker.color.b = color[2];
            marker.color.a = 1.0;

            for (const auto &pt : trails[i])
            {
                geometry_msgs::msg::Point point;
                point.x = pt.x();
                point.y = pt.y();
                point.z = pt.z();
                marker.points.push_back(point);
            }
            array.markers.push_back(std::move(marker));
        }

        animationPub->publish(array);
    }

    inline void clearAnimationFrame(const int drone_count)
    {
        visualization_msgs::msg::MarkerArray array;
        const auto stamp = node->get_clock()->now();
        for (int id = 0; id < drone_count; ++id)
        {
            visualization_msgs::msg::Marker body_marker;
            body_marker.header.stamp = stamp;
            body_marker.header.frame_id = "odom";
            body_marker.ns = "agent_body";
            body_marker.id = id;
            body_marker.action = visualization_msgs::msg::Marker::DELETE;
            array.markers.push_back(std::move(body_marker));

            visualization_msgs::msg::Marker trail_marker;
            trail_marker.header.stamp = stamp;
            trail_marker.header.frame_id = "odom";
            trail_marker.ns = "executed_trail";
            trail_marker.id = id;
            trail_marker.action = visualization_msgs::msg::Marker::DELETE;
            array.markers.push_back(std::move(trail_marker));
        }
        animationPub->publish(array);
    }

    inline void visualizeStartGoal(const Eigen::Vector3d &center,
                                   const double &radius,
                                   int sg,
                                   bool clear_previous = false,
                                   const std::array<float, 3> &color = {1.0f, 0.0f, 0.0f})
    {
        if (clear_previous)
        {
            clearStartGoalVisuals();
        }

        visualization_msgs::msg::MarkerArray array;
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = node->get_clock()->now();
        marker.header.frame_id = "odom";
        marker.ns = "StartGoal";
        marker.id = sg;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = radius * 2.0;
        marker.scale.y = radius * 2.0;
        marker.scale.z = radius * 2.0;
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = 1.0;
        array.markers.push_back(std::move(marker));

        animationPub->publish(array);
    }
};

#endif
