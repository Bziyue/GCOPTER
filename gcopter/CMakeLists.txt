cmake_minimum_required(VERSION 3.8)
project(gcopter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(
  include
  ${OMPL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(global_planning src/global_planning.cpp)
ament_target_dependencies(global_planning
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

target_link_libraries(global_planning
  ${OMPL_LIBRARIES}
)

install(TARGETS global_planning
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/start_global_planning_launch.sh
  scripts/trajectory_dynamics_analysis.py
  scripts/swarm_min_distance_analysis.py
  scripts/run_trajectory_dynamics_analysis.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
