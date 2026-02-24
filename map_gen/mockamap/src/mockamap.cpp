#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "maps.hpp"

#define ROS_INFO(...) RCLCPP_INFO(rclcpp::get_logger("mockamap"), __VA_ARGS__)

void
optimizeMap(mocka::Maps::BasicInfo& in)
{
  std::vector<int>* temp = new std::vector<int>;

  pcl::KdTreeFLANN<pcl::PointXYZ>     kdtree;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  cloud->width  = in.cloud->width;
  cloud->height = in.cloud->height;
  cloud->points.resize(cloud->width * cloud->height);

  for (int i = 0; i < static_cast<int>(cloud->width); i++)
  {
    cloud->points[i].x = in.cloud->points[i].x;
    cloud->points[i].y = in.cloud->points[i].y;
    cloud->points[i].z = in.cloud->points[i].z;
  }

  kdtree.setInputCloud(cloud);
  double radius = 1.75 / in.scale;

  for (int i = 0; i < static_cast<int>(cloud->width); i++)
  {
    std::vector<int>   pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    if (kdtree.radiusSearch(cloud->points[i], radius, pointIdxRadiusSearch,
                            pointRadiusSquaredDistance) >= 27)
    {
      temp->push_back(i);
    }
  }
  for (int i = static_cast<int>(temp->size()) - 1; i >= 0; i--)
  {
    in.cloud->points.erase(in.cloud->points.begin() +
                           temp->at(i));
  }
  in.cloud->width -= temp->size();

  pcl::toROSMsg(*in.cloud, *in.output);
  in.output->header.frame_id = "odom";
  ROS_INFO("finish: number of points after optimization %d", in.cloud->width);
  delete temp;
}

int
main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("mockamap_node");

  auto pcl_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("mock_map", 1);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  sensor_msgs::msg::PointCloud2  output;

  int seed;
  int sizeX;
  int sizeY;
  int sizeZ;
  double scale;
  double update_freq;
  int type;

  node->declare_parameter<int>("seed", 4546);
  node->declare_parameter<double>("update_freq", 1.0);
  node->declare_parameter<double>("resolution", 0.38);
  node->declare_parameter<int>("x_length", 100);
  node->declare_parameter<int>("y_length", 100);
  node->declare_parameter<int>("z_length", 10);
  node->declare_parameter<int>("type", 1);

  node->get_parameter("seed", seed);
  node->get_parameter("update_freq", update_freq);
  node->get_parameter("resolution", scale);
  node->get_parameter("x_length", sizeX);
  node->get_parameter("y_length", sizeY);
  node->get_parameter("z_length", sizeZ);
  node->get_parameter("type", type);

  scale = 1 / scale;
  sizeX = sizeX * scale;
  sizeY = sizeY * scale;
  sizeZ = sizeZ * scale;

  mocka::Maps::BasicInfo info;
  info.nh_private = node.get();
  info.sizeX      = sizeX;
  info.sizeY      = sizeY;
  info.sizeZ      = sizeZ;
  info.seed       = seed;
  info.scale      = scale;
  info.output     = &output;
  info.cloud      = &cloud;

  mocka::Maps map;
  map.setInfo(info);
  map.generate(type);

  rclcpp::Rate loop_rate(update_freq);
  while (rclcpp::ok())
  {
    pcl_pub->publish(output);
    rclcpp::spin_some(node);
    loop_rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
