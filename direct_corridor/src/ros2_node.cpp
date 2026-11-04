// SPDX-License-Identifier: MIT
#include <algorithm>
#include <iostream>
#include <memory>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sstream>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "gcopter/direct_corridor.hpp"
using namespace gcopter::direct;

// Single-threaded adapter: every incoming map/path first clears previous
// output. Core geometry remains ROS-independent; this adapter does not run
// trajectory control.
class DirectCorridorNode final : public rclcpp::Node {
 public:
  DirectCorridorNode() : Node("direct_corridor") {
    frame_ = declare_parameter<std::string>("frame", "map");
    resolution_ = declare_parameter<double>("voxel_resolution", .2);
    auto b = declare_parameter<std::vector<double>>("bounds",
                                                    {-25, -25, 0, 25, 25, 5});
    if (b.size() != 6 || !std::isfinite(resolution_) || resolution_ <= 0)
      throw std::invalid_argument(
          "expected six bounds and positive voxel resolution");
    bounds_ = {Vec3(b[0], b[1], b[2]), Vec3(b[3], b[4], b[5])};
    if (!bounds_.lower.allFinite() || !bounds_.upper.allFinite() ||
        (bounds_.upper.array() <= bounds_.lower.array()).any())
      throw std::invalid_argument("invalid bounds");
    options_.robot_radius = declare_parameter<double>("robot_radius", .2);
    options_.safety_margin = declare_parameter<double>("safety_margin", .02);
    options_.control_spacing = declare_parameter<double>("control_spacing", .5);
    options_.max_offset = declare_parameter<double>("max_offset", 1.5);
    options_.certificate_depth = declare_parameter<int>("certificate_depth", 9);
    options_.min_radius = declare_parameter<double>("min_radius", .02);
    options_.max_radius = declare_parameter<double>("max_radius", 1.5);
    options_.budget_ms = declare_parameter<double>("budget_ms", 1000.);
    const auto qos = rclcpp::QoS(1).reliable().transient_local();
    markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/markers", qos);
    coefficients_ =
        create_publisher<std_msgs::msg::Float64MultiArray>("~/controls", qos);
    reference_ = create_publisher<nav_msgs::msg::Path>("~/reference", qos);
    status_ = create_publisher<std_msgs::msg::String>("~/status", qos);
    map_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "~/map", rclcpp::QoS(1),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          mapCallback(*msg);
        });
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "~/seed", rclcpp::QoS(1),
        [this](nav_msgs::msg::Path::ConstSharedPtr msg) {
          pathCallback(*msg);
        });
    report("WAITING_FOR_MAP_AND_SEED");
  }

 private:
  void report(const std::string& text) {
    std_msgs::msg::String m;
    m.data = text;
    status_->publish(m);
    RCLCPP_INFO(get_logger(), "%s", text.c_str());
  }
  void clear() {
    visualization_msgs::msg::MarkerArray out;
    visualization_msgs::msg::Marker m;
    m.action = m.DELETEALL;
    out.markers.push_back(m);
    markers_->publish(out);
    coefficients_->publish(std_msgs::msg::Float64MultiArray());
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_;
    path.header.stamp = now();
    reference_->publish(path);
  }
  void mapCallback(const sensor_msgs::msg::PointCloud2& msg) {
    clear();
    map_.reset();
    try {
      if (msg.header.frame_id != frame_)
        throw std::invalid_argument("map frame mismatch; no implicit TF");
      if (msg.is_bigendian)
        throw std::invalid_argument("big-endian cloud unsupported");
      if (msg.point_step == 0 || msg.width == 0 || msg.height == 0 ||
          std::uint64_t(msg.row_step) * msg.height != msg.data.size() ||
          std::uint64_t(msg.width) * msg.point_step != msg.row_step ||
          std::uint64_t(msg.width) * msg.height > 1000000)
        throw std::invalid_argument(
            "empty, malformed, padded or over-capacity cloud");
      for (const std::string name : {"x", "y", "z"}) {
        auto f =
            std::find_if(msg.fields.begin(), msg.fields.end(),
                         [&](const auto& field) { return field.name == name; });
        if (f == msg.fields.end() ||
            f->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
            f->count != 1 ||
            std::uint64_t(f->offset) + sizeof(float) > msg.point_step)
          throw std::invalid_argument(
              "cloud requires scalar FLOAT32 xyz fields");
      }
      std::vector<Vec3> centers;
      sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"),
          z(msg, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        Vec3 p(*x, *y, *z);
        if (!p.allFinite()) continue;
        if ((p.array() < bounds_.lower.array()).any() ||
            (p.array() >= bounds_.upper.array()).any())
          continue;
        Vec3 cell =
            ((p - bounds_.lower) / resolution_).array().floor().matrix();
        centers.push_back(bounds_.lower +
                          (cell + Vec3::Constant(.5)) * resolution_);
      }
      if (centers.empty())
        throw std::invalid_argument("cloud has no finite in-bounds points");
      std::sort(centers.begin(), centers.end(),
                [](const Vec3& a, const Vec3& b) {
                  for (int k = 0; k < 3; ++k)
                    if (a[k] != b[k]) return a[k] < b[k];
                  return false;
                });
      centers.erase(
          std::unique(centers.begin(), centers.end(),
                      [](const Vec3& a, const Vec3& b) { return a == b; }),
          centers.end());
      std::vector<Sphere> spheres;
      spheres.reserve(centers.size());
      for (const auto& c : centers)
        spheres.push_back({c, std::sqrt(3.) * .5 * resolution_});
      map_ = std::make_unique<ObstacleIndex>(std::move(spheres));
      if (seed_.empty())
        report("WAITING_FOR_SEED");
      else
        compute();
    } catch (const std::exception& e) {
      report(std::string("INVALID_MAP: ") + e.what());
    }
  }
  void pathCallback(const nav_msgs::msg::Path& msg) {
    clear();
    seed_.clear();
    if (msg.header.frame_id != frame_) {
      report("INVALID_SEED: frame mismatch");
      return;
    }
    if (msg.poses.size() > options_.max_controls || msg.poses.size() < 2) {
      report("INVALID_SEED: size");
      return;
    }
    for (const auto& p : msg.poses) {
      if (!p.header.frame_id.empty() && p.header.frame_id != frame_) {
        seed_.clear();
        report("INVALID_SEED: pose frame mismatch");
        return;
      }
      seed_.emplace_back(p.pose.position.x, p.pose.position.y,
                         p.pose.position.z);
    }
    if (!map_) {
      report("WAITING_FOR_MAP");
      return;
    }
    compute();
  }
  void compute() {
    const auto r = generate(seed_, *map_, bounds_, options_);
    if (!r) {
      report(std::string(statusName(r.status)) + ": " + r.message);
      return;
    }
    std_msgs::msg::Float64MultiArray coefficients;
    coefficients.layout.dim.resize(2);
    const auto n = r.corridor.centerControls().size();
    coefficients.layout.dim[0].label = "quintic_bezier_controls_v2";
    coefficients.layout.dim[0].size = n;
    coefficients.layout.dim[0].stride = n * 11;
    coefficients.layout.dim[1].label = "u0_u1_cx_cy_cz_m00_m10_m20_m01_m11_m21";
    coefficients.layout.dim[1].size = 11;
    coefficients.layout.dim[1].stride = 11;
    for (std::size_t i = 0; i < n; ++i) {
      coefficients.data.push_back(r.corridor.knots()[i / 6]);
      coefficients.data.push_back(r.corridor.knots()[i / 6 + 1]);
      const auto& c = r.corridor.centerControls()[i];
      const auto& m = r.corridor.shapeControls()[i];
      for (int k = 0; k < 3; ++k) coefficients.data.push_back(c[k]);
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 3; ++k) coefficients.data.push_back(m(k, j));
    }
    visualization_msgs::msg::MarkerArray output;
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_;
    marker.header.stamp = now();
    marker.ns = "direct_hermite_corridor";
    marker.id = 0;
    marker.type = marker.LINE_LIST;
    marker.action = marker.ADD;
    marker.pose.orientation.w = 1;
    marker.scale.x = .015;
    marker.color.r = .65;
    marker.color.g = .2;
    marker.color.b = 1;
    marker.color.a = .8;
    nav_msgs::msg::Path reference;
    reference.header = marker.header;
    const int samples =
        std::min(2000, std::max(80, int(r.corridor.pieces() * 4)));
    auto point = [](const Vec3& p) {
      geometry_msgs::msg::Point out;
      out.x = p.x();
      out.y = p.y();
      out.z = p.z();
      return out;
    };
    for (int i = 0; i <= samples; ++i) {
      const auto s = r.corridor.section(double(i) / samples);
      geometry_msgs::msg::PoseStamped pose;
      pose.header = reference.header;
      pose.pose.position = point(s.center);
      pose.pose.orientation.w = 1;
      reference.poses.push_back(pose);
      for (int j = 0; j < 24; ++j) {
        const double a = j * 2 * std::acos(-1.) / 24,
                     b = (j + 1) * 2 * std::acos(-1.) / 24;
        marker.points.push_back(point(s.point({std::cos(a), std::sin(a)})));
        marker.points.push_back(point(s.point({std::cos(b), std::sin(b)})));
      }
    }
    output.markers.push_back(std::move(marker));
    markers_->publish(output);
    coefficients_->publish(coefficients);
    reference_->publish(reference);
    std::ostringstream summary;
    summary << "SUCCESS ms=" << r.metrics.total_ms << " controls=" << n
            << " constraints=" << r.metrics.constraints;
    report(summary.str());
  }
  std::string frame_;
  double resolution_;
  Bounds bounds_;
  Options options_;
  std::unique_ptr<ObstacleIndex> map_;
  std::vector<Vec3> seed_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr coefficients_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr reference_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_;
};
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<DirectCorridorNode>());
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
