// SPDX-License-Identifier: MIT
// GCOPTER mockamap + original Informed RRT* -> direct quintic Hermite corridor
// demo.
#include <ompl/util/RandomNumbers.h>
#include <unistd.h>

#include <Eigen/Eigenvalues>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <iomanip>
#include <nav_msgs/msg/path.hpp>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sstream>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "gcopter/direct_search_map.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/voxel_map.hpp"

using namespace gcopter::direct;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
using Marker = visualization_msgs::msg::Marker;
using Markers = visualization_msgs::msg::MarkerArray;
namespace {
double milliseconds(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
geometry_msgs::msg::Point point(const Vec3& p) {
  geometry_msgs::msg::Point m;
  m.x = p.x();
  m.y = p.y();
  m.z = p.z();
  return m;
}
void writePoints(const fs::path& path, const std::vector<Vec3>& p) {
  std::ofstream out(path);
  out << std::setprecision(17);
  for (const auto& v : p) out << v.transpose() << '\n';
  if (!out) throw std::runtime_error("snapshot write failed");
}
}  // namespace

class DirectDemo final : public rclcpp::Node {
 public:
  DirectDemo() : Node("direct_demo") {
    frame_ = declare_parameter<std::string>("frame", "odom");
    resolution_ = declare_parameter<double>("resolution", .25);
    const auto b = declare_parameter<std::vector<double>>(
        "bounds", {-25, -25, 0, 25, 25, 5});
    if (b.size() != 6)
      throw std::invalid_argument("bounds needs lower xyz, upper xyz");
    bounds_ = {Vec3(b[0], b[1], b[2]), Vec3(b[3], b[4], b[5])};
    options_.robot_radius = declare_parameter<double>("robot_radius", .2);
    options_.safety_margin = declare_parameter<double>("safety_margin", .05);
    options_.min_radius = declare_parameter<double>("min_radius", .03);
    options_.max_radius = declare_parameter<double>("max_radius", 1.5);
    options_.control_spacing = declare_parameter<double>("control_spacing", .4);
    options_.max_offset = declare_parameter<double>("max_offset", 1.5);
    options_.inflation_iterations =
        declare_parameter<int>("inflation_iterations", 2);
    options_.certificate_depth = declare_parameter<int>("certificate_depth", 9);
    options_.budget_ms = declare_parameter<double>("corridor_budget_ms", 1000.);
    reference_clearance_ = declare_parameter<double>("reference_clearance", .2);
    search_timeout_ = declare_parameter<double>("search_timeout", .5);
    search_range_ = declare_parameter<double>("search_range", 2.);
    height_mode_ = declare_parameter<std::string>("height_mode", "fixed");
    height_ = declare_parameter<double>("height", 2.5);
    if (!bounds_.lower.allFinite() || !bounds_.upper.allFinite() ||
        (bounds_.upper.array() <= bounds_.lower.array()).any() ||
        !std::isfinite(resolution_) || resolution_ <= 0 ||
        !std::isfinite(reference_clearance_) || reference_clearance_ < 0 ||
        !std::isfinite(search_timeout_) || search_timeout_ <= 0 ||
        !std::isfinite(search_range_) || search_range_ <= 0 ||
        !std::isfinite(height_) ||
        (height_mode_ != "fixed" && height_mode_ != "arrow"))
      throw std::invalid_argument(
          "invalid demo geometry/search/height parameters");
    // Validate all core options before creating publishers or accepting
    // endpoints.
    const auto option_check =
        generate({(bounds_.lower + bounds_.upper) * .5,
                  (bounds_.lower + bounds_.upper) * .5 + Vec3(.01, 0, 0)},
                 ObstacleIndex({}), bounds_, options_);
    if (option_check.status == Status::InvalidInput)
      throw std::invalid_argument(option_check.message);
    const Vec3 dimensions =
        ((bounds_.upper - bounds_.lower) / resolution_).array().ceil().matrix();
    if (dimensions.prod() > 10000000)
      throw std::invalid_argument("map exceeds ten million voxels");
    dims_ = dimensions.cast<int>();
    const auto seed = declare_parameter<int>("search_seed", 7);
    if (seed <= 0) throw std::invalid_argument("search_seed must be positive");
    ompl::RNG::setSeed(seed);
    const std::string root = declare_parameter<std::string>(
        "output_dir", "/tmp/gcopter_direct_demo");
    const auto stamp =
        std::chrono::system_clock::now().time_since_epoch().count();
    session_ = fs::path(root) / ("run_" + std::to_string(stamp) + "_" +
                                 std::to_string(getpid()));
    fs::create_directories(session_);
    declare_parameter<std::string>("session_dir", session_.string());
    const auto qos = rclcpp::QoS(1).reliable().transient_local();
    corridor_pub_ = create_publisher<Markers>("~/corridor", qos);
    selection_pub_ = create_publisher<Markers>("~/selection", qos);
    seed_pub_ = create_publisher<nav_msgs::msg::Path>("~/seed", qos);
    reference_pub_ = create_publisher<nav_msgs::msg::Path>("~/reference", qos);
    controls_pub_ =
        create_publisher<std_msgs::msg::Float64MultiArray>("~/controls", qos);
    status_pub_ = create_publisher<std_msgs::msg::String>("~/status", qos);
    metrics_pub_ = create_publisher<std_msgs::msg::String>("~/metrics", qos);
    map_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/voxel_map", rclcpp::QoS(1),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          onMap(msg);
        });
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/move_base_simple/goal", 10,
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          select(msg->header.frame_id, posePoint(msg->pose), false);
        });
    start_sub_ =
        create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr
                       msg) {
              select(msg->header.frame_id, posePoint(msg->pose.pose), true);
            });
    click_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
        "/clicked_point", 10,
        [this](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
          select(msg->header.frame_id,
                 Vec3(msg->point.x, msg->point.y, msg->point.z), false);
        });
    reset_sub_ = create_subscription<std_msgs::msg::Empty>(
        "~/reset", 10, [this](std_msgs::msg::Empty::ConstSharedPtr) {
          endpoints_.clear();
          clearOutput();
          status(map_ ? "WAITING_FOR_START" : "WAITING_FOR_MAP");
        });
    status("WAITING_FOR_MAP");
    RCLCPP_INFO(get_logger(), "session_dir=%s", session_.c_str());
  }

 private:
  Vec3 posePoint(const geometry_msgs::msg::Pose& pose) const {
    double z = height_;
    if (height_mode_ == "arrow") {
      const double inset = options_.robot_radius + options_.safety_margin +
                           reference_clearance_ + resolution_;
      z = bounds_.lower.z() + inset +
          std::clamp(std::abs(pose.orientation.z), 0., 1.) *
              (bounds_.upper.z() - bounds_.lower.z() - 2 * inset);
    }
    return {pose.position.x, pose.position.y, z};
  }
  Marker baseMarker(const std::string& name, int id, int type) const {
    Marker m;
    m.header.frame_id = frame_;
    m.header.stamp = now();
    m.ns = name;
    m.id = id;
    m.type = type;
    m.action = Marker::ADD;
    m.pose.orientation.w = 1;
    return m;
  }
  void status(const std::string& text) {
    std_msgs::msg::String out;
    out.data = text;
    status_pub_->publish(out);
    RCLCPP_INFO(get_logger(), "%s", text.c_str());
    Markers markers;
    Marker clear;
    clear.action = Marker::DELETEALL;
    markers.markers.push_back(clear);
    for (std::size_t i = 0; i < endpoints_.size(); ++i) {
      auto m = baseMarker("selection", i, Marker::SPHERE);
      m.pose.position = point(endpoints_[i]);
      m.scale.x = m.scale.y = m.scale.z = .45;
      m.color.a = 1;
      m.color.r = i == 1 ? .95 : .1;
      m.color.g = i == 0 ? .95 : .1;
      markers.markers.push_back(m);
    }
    auto label = baseMarker("status", 0, Marker::TEXT_VIEW_FACING);
    label.pose.position =
        point(endpoints_.empty() ? Vec3(0, 0, bounds_.upper.z() + .5)
                                 : endpoints_.front() + Vec3(0, 0, 1));
    label.scale.z = .4;
    label.color.r = label.color.g = label.color.b = label.color.a = 1;
    label.text = text;
    markers.markers.push_back(label);
    selection_pub_->publish(markers);
  }
  void clearOutput() {
    Markers m;
    Marker clear;
    clear.action = Marker::DELETEALL;
    m.markers.push_back(clear);
    corridor_pub_->publish(m);
    nav_msgs::msg::Path p;
    p.header.frame_id = frame_;
    p.header.stamp = now();
    seed_pub_->publish(p);
    reference_pub_->publish(p);
    controls_pub_->publish(std_msgs::msg::Float64MultiArray());
    metrics_pub_->publish(std_msgs::msg::String());
  }
  void onMap(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
    // mockamap republishes an immutable map at 1 Hz. Identical payloads must
    // not rebuild the BVH, reset selection or erase a displayed result.
    if (last_map_ && msg->header.frame_id == last_map_->header.frame_id &&
        msg->width == last_map_->width && msg->height == last_map_->height &&
        msg->point_step == last_map_->point_step &&
        msg->row_step == last_map_->row_step &&
        msg->is_bigendian == last_map_->is_bigendian &&
        msg->fields == last_map_->fields && msg->data == last_map_->data)
      return;
    endpoints_.clear();
    clearOutput();
    map_.reset();
    last_map_.reset();
    const auto start = Clock::now();
    try {
      if (msg->header.frame_id != frame_ || msg->is_bigendian ||
          msg->width == 0 || msg->height == 0 || msg->point_step == 0 ||
          std::uint64_t(msg->row_step) * msg->height != msg->data.size() ||
          std::uint64_t(msg->width) * msg->point_step != msg->row_step ||
          std::uint64_t(msg->width) * msg->height > 4000000)
        throw std::runtime_error("map frame/layout/size invalid");
      for (const std::string name : {"x", "y", "z"}) {
        auto f =
            std::find_if(msg->fields.begin(), msg->fields.end(),
                         [&](const auto& field) { return field.name == name; });
        if (f == msg->fields.end() ||
            f->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
            f->count != 1 || std::uint64_t(f->offset) + 4 > msg->point_step)
          throw std::runtime_error("map requires FLOAT32 xyz");
      }
      voxel_map::VoxelMap voxels(dims_, bounds_.lower, resolution_);
      sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x"), y(*msg, "y"),
          z(*msg, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        Vec3 p(*x, *y, *z);
        if (p.allFinite() && (p.array() >= bounds_.lower.array()).all() &&
            (p.array() < bounds_.upper.array()).all())
          voxels.setOccupied(p);
      }
      std::vector<Sphere> spheres;
      std::vector<Vec3> centers;
      const auto& cells = voxels.getVoxels();
      const double radius = std::sqrt(3.) * .5 * resolution_;
      for (std::size_t k = 0; k < cells.size(); ++k)
        if (cells[k]) {
          const int z = k / (dims_.x() * dims_.y()),
                    rest = k - z * dims_.x() * dims_.y();
          const Vec3 p = voxels.posI2D(
              Eigen::Vector3i(rest % dims_.x(), rest / dims_.x(), z));
          spheres.push_back({p, radius});
          centers.push_back(p);
        }
      if (spheres.empty()) throw std::runtime_error("empty occupied map");
      map_ = std::make_unique<ObstacleIndex>(std::move(spheres));
      ++map_version_;
      map_ms_ = milliseconds(start);
      writePoints(session_ / ("map_" + std::to_string(map_version_) + ".xyz"),
                  centers);
      last_map_ = std::move(msg);
      status("WAITING_FOR_START");
    } catch (const std::exception& e) {
      map_.reset();
      status(std::string("INVALID_MAP: ") + e.what());
    }
  }
  void select(const std::string& frame, const Vec3& p, bool explicit_start) {
    if (!map_) {
      status("WAITING_FOR_MAP");
      return;
    }
    clearOutput();
    if (explicit_start || endpoints_.size() == 2) endpoints_.clear();
    SearchMap search(
        *map_, bounds_,
        options_.robot_radius + options_.safety_margin + reference_clearance_);
    if (frame != frame_ || search.query(p)) {
      status("INVALID_POINT: choose free space with reference clearance");
      return;
    }
    if (endpoints_.size() == 1 && (p - endpoints_.front()).norm() < 1e-5) {
      status("INVALID_POINT: start equals goal");
      return;
    }
    endpoints_.push_back(p);
    if (endpoints_.size() == 1) {
      status("WAITING_FOR_GOAL");
      return;
    }
    plan(search);
  }
  nav_msgs::msg::Path pathMessage(const std::vector<Vec3>& points) const {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_;
    path.header.stamp = now();
    for (const auto& p : points) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position = point(p);
      pose.pose.orientation.w = 1;
      path.poses.push_back(pose);
    }
    return path;
  }
  void plan(const SearchMap& search) {
    const auto begin = Clock::now();
    const auto id = ++request_;
    const fs::path dir = session_ / ("request_" + std::to_string(id));
    fs::create_directories(dir);
    status("SEARCHING request=" + std::to_string(id));
    std::vector<Vec3> seed;
    const double cost = sfc_gen::planPath(
        endpoints_[0], endpoints_[1], bounds_.lower, bounds_.upper, &search,
        search_timeout_, seed,
        [&](const Vec3& a, const Vec3& b) { return search.segmentFree(a, b); },
        search_range_);
    const double search_ms = milliseconds(begin);
    std::string outcome = "SEARCH_FAILED";
    Result result;
    if (std::isfinite(cost) && seed.size() >= 2 &&
        (seed.front() - endpoints_[0]).norm() < 1e-7 &&
        (seed.back() - endpoints_[1]).norm() < 1e-7) {
      bool edges_ok = true;
      for (std::size_t k = 1; k < seed.size(); ++k)
        edges_ok &= search.segmentFree(seed[k - 1], seed[k]);
      if (edges_ok) {
        status("GENERATING request=" + std::to_string(id));
        result = generate(seed, *map_, bounds_, options_);
        outcome = result ? "PASS" : statusName(result.status);
      } else
        outcome = "SEARCH_EDGE_COLLISION";
    }
    double length = 0;
    int turns = 0;
    double lowz = 1e9, highz = -1e9;
    for (std::size_t k = 0; k < seed.size(); ++k) {
      lowz = std::min(lowz, seed[k].z());
      highz = std::max(highz, seed[k].z());
      if (k) length += (seed[k] - seed[k - 1]).norm();
      if (k > 1) {
        Vec3 a = seed[k - 1] - seed[k - 2], b = seed[k] - seed[k - 1];
        if (a.norm() > 1e-8 && b.norm() > 1e-8 &&
            a.normalized().dot(b.normalized()) < std::cos(std::acos(-1.) / 6))
          ++turns;
      }
    }
    double min_axis = 0, median_minor = 0, median_major = 0;
    const auto rendering_start = Clock::now();
    if (result) {
      const auto& c = result.corridor;
      const int samples = std::min(800, std::max(120, int(c.pieces() * 4)));
      constexpr int angles = 24;
      std::vector<Vec3> reference;
      std::vector<double> minor, major;
      auto wire = baseMarker("wire", 0, Marker::LINE_LIST);
      wire.scale.x = .018;
      wire.color.r = .75;
      wire.color.g = .35;
      wire.color.b = 1;
      wire.color.a = .85;
      auto mesh = baseMarker("surface", 1, Marker::TRIANGLE_LIST);
      mesh.scale.x = mesh.scale.y = mesh.scale.z = 1;
      mesh.color = wire.color;
      mesh.color.a = .14;
      std::array<Vec3, angles> previous{};
      for (int i = 0; i <= samples; ++i) {
        const auto s = c.section(double(i) / samples);
        reference.push_back(s.center);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(
            s.shape.transpose() * s.shape);
        minor.push_back(std::sqrt(eigen.eigenvalues()[0]));
        major.push_back(std::sqrt(eigen.eigenvalues()[1]));
        std::array<Vec3, angles> ring{};
        for (int j = 0; j < angles; ++j) {
          const double a = 2 * std::acos(-1.) * j / angles;
          ring[j] = s.point({std::cos(a), std::sin(a)});
        }
        for (int j = 0; j < angles; ++j) {
          int k = (j + 1) % angles;
          if (i % 3 == 0 || i == samples) {
            wire.points.push_back(point(ring[j]));
            wire.points.push_back(point(ring[k]));
          }
          if (i)
            for (const auto& p : {previous[j], previous[k], ring[k],
                                  previous[j], ring[k], ring[j]})
              mesh.points.push_back(point(p));
        }
        previous = ring;
      }
      std::sort(minor.begin(), minor.end());
      std::sort(major.begin(), major.end());
      min_axis = minor.front();
      median_minor = minor[minor.size() / 2];
      median_major = major[major.size() / 2];
      Markers markers;
      markers.markers = {std::move(mesh), std::move(wire)};
      corridor_pub_->publish(markers);
      seed_pub_->publish(pathMessage(seed));
      reference_pub_->publish(pathMessage(reference));
      std_msgs::msg::Float64MultiArray controls;
      controls.layout.dim.resize(2);
      controls.layout.dim[0].label = "quintic_bezier_controls_v2";
      controls.layout.dim[0].size = c.centerControls().size();
      controls.layout.dim[0].stride = c.centerControls().size() * 11;
      controls.layout.dim[1].label = "u0_u1_cx_cy_cz_m00_m10_m20_m01_m11_m21";
      controls.layout.dim[1].size = 11;
      controls.layout.dim[1].stride = 11;
      std::ofstream coefficients(dir / "controls.csv");
      coefficients << std::setprecision(17)
                   << "u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21\n";
      for (std::size_t k = 0; k < c.centerControls().size(); ++k) {
        coefficients << c.knots()[k / 6] << ',' << c.knots()[k / 6 + 1] << ',';
        controls.data.push_back(c.knots()[k / 6]);
        controls.data.push_back(c.knots()[k / 6 + 1]);
        const auto& center = c.centerControls()[k];
        const auto& m = c.shapeControls()[k];
        for (int j = 0; j < 3; ++j) {
          if (j) coefficients << ',';
          coefficients << center[j];
          controls.data.push_back(center[j]);
        }
        for (int col = 0; col < 2; ++col)
          for (int row = 0; row < 3; ++row) {
            coefficients << ',' << m(row, col);
            controls.data.push_back(m(row, col));
          }
        coefficients << '\n';
      }
      controls_pub_->publish(controls);
      writePoints(dir / "reference.xyz", reference);
    }
    writePoints(dir / "seed.xyz", seed);
    const double render_ms = milliseconds(rendering_start),
                 total_ms = milliseconds(begin);
    auto jsonPoint = [](const Vec3& p) {
      std::ostringstream s;
      s << std::setprecision(17) << '[' << p.x() << ',' << p.y() << ',' << p.z()
        << ']';
      return s.str();
    };
    std::ostringstream metric;
    metric << std::setprecision(12) << "{\"request\":" << id
           << ",\"map_version\":" << map_version_ << ",\"status\":\"" << outcome
           << "\",\"start\":" << jsonPoint(endpoints_[0])
           << ",\"goal\":" << jsonPoint(endpoints_[1])
           << ",\"bounds_lower\":" << jsonPoint(bounds_.lower)
           << ",\"bounds_upper\":" << jsonPoint(bounds_.upper)
           << ",\"resolution\":" << resolution_
           << ",\"robot_radius\":" << options_.robot_radius
           << ",\"safety_margin\":" << options_.safety_margin
           << ",\"reference_clearance\":" << reference_clearance_
           << ",\"control_spacing\":" << options_.control_spacing
           << ",\"min_radius\":" << options_.min_radius
           << ",\"max_radius\":" << options_.max_radius
           << ",\"search_timeout\":" << search_timeout_
           << ",\"search_range\":" << search_range_
           << ",\"corridor_budget_ms\":" << options_.budget_ms
           << ",\"obstacles\":" << map_->obstacles().size()
           << ",\"map_build_ms\":" << map_ms_ << ",\"search_ms\":" << search_ms
           << ",\"corridor_ms\":" << result.metrics.total_ms
           << ",\"render_export_ms\":" << render_ms
           << ",\"total_ms\":" << total_ms << ",\"seed_points\":" << seed.size()
           << ",\"seed_length\":" << length << ",\"turns_over_30deg\":" << turns
           << ",\"height_span\":" << (seed.empty() ? 0 : highz - lowz)
           << ",\"representation\":\"quintic_hermite_v2\""
           << ",\"nodes\":" << result.metrics.nodes
           << ",\"max_center_shift\":" << result.metrics.max_center_shift
           << ",\"optimization_ms\":" << result.metrics.optimization_ms
           << ",\"certificate_ms\":" << result.metrics.certificate_ms
           << ",\"controls\":" << result.metrics.controls
           << ",\"constraints\":" << result.metrics.constraints
           << ",\"refinements\":" << result.metrics.refinements
           << ",\"min_axis\":" << min_axis
           << ",\"median_minor\":" << median_minor
           << ",\"median_major\":" << median_major << '}';
    std::ofstream file(dir / "metrics.json");
    file << metric.str() << '\n';
    std_msgs::msg::String metrics;
    metrics.data = metric.str();
    metrics_pub_->publish(metrics);
    std::ostringstream label;
    label << outcome << " request=" << id << "\nSearch " << std::fixed
          << std::setprecision(1) << search_ms << " ms; corridor "
          << result.metrics.total_ms << " ms; total " << total_ms << " ms";
    if (result)
      label << "\nMin semiaxis " << std::setprecision(3) << min_axis << " m";
    else
      label << "\n" << result.message;
    status(label.str());
  }
  std::string frame_, height_mode_;
  double resolution_, height_, reference_clearance_, search_timeout_,
      search_range_, map_ms_ = 0;
  Bounds bounds_;
  Eigen::Vector3i dims_;
  Options options_;
  fs::path session_;
  int map_version_ = 0, request_ = 0;
  std::unique_ptr<ObstacleIndex> map_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr last_map_;
  std::vector<Vec3> endpoints_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Publisher<Markers>::SharedPtr corridor_pub_, selection_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr seed_pub_, reference_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr controls_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_, metrics_pub_;
};
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<DirectDemo>());
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
