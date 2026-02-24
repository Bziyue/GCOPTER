cmake_minimum_required(VERSION 3.8)
project(mockamap)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common kdtree)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

add_executable(mockamap_node
  src/mockamap.cpp
  src/maps.cpp
  src/perlinnoise.cpp
)

ament_target_dependencies(mockamap_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_link_libraries(mockamap_node
  ${PCL_LIBRARIES}
)

install(TARGETS mockamap_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
