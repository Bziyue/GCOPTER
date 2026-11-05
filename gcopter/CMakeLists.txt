cmake_minimum_required(VERSION 3.16)
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
find_package(nav_msgs REQUIRED)
find_package(SweepCorridor REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

include_directories(
  include
  ${OMPL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(global_planning src/global_planning.cpp)
target_link_libraries(global_planning
  rclcpp::rclcpp ${std_msgs_TARGETS} ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS} ${visualization_msgs_TARGETS}
  ${OMPL_LIBRARIES}
)
target_link_directories(global_planning PRIVATE ${OMPL_LIBRARY_DIRS})

add_library(demo_comparison STATIC src/demo_comparison.cpp)
target_include_directories(demo_comparison PRIVATE tools)
target_link_libraries(demo_comparison PUBLIC SweepCorridor::SweepCorridor)
add_executable(direct_corridor_demo src/direct_corridor_demo.cpp src/demo_generation.cpp)
target_link_libraries(direct_corridor_demo PRIVATE demo_comparison SweepCorridor::SweepCorridor
  ${OMPL_LIBRARIES} rclcpp::rclcpp ${std_msgs_TARGETS} ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS} ${visualization_msgs_TARGETS} ${nav_msgs_TARGETS})
target_link_directories(direct_corridor_demo PRIVATE ${OMPL_LIBRARY_DIRS})
add_library(gcopter_corridor_panel SHARED src/corridor_panel.cpp src/ghost_map_display.cpp)
target_link_libraries(gcopter_corridor_panel PRIVATE rviz_common::rviz_common
  pluginlib::pluginlib Qt6::Widgets rclcpp::rclcpp ${std_msgs_TARGETS} ${visualization_msgs_TARGETS})
pluginlib_export_plugin_description_file(rviz_common plugins/corridor_panel.xml)
install(TARGETS gcopter_corridor_panel LIBRARY DESTINATION lib)
option(GCOPTER_BUILD_CORRIDOR_BENCHMARK "Build offline FIRI volume comparison tools" OFF)
if(GCOPTER_BUILD_CORRIDOR_BENCHMARK)
  add_executable(corridor_firi_benchmark tools/corridor_firi_benchmark.cpp)
  target_link_libraries(corridor_firi_benchmark PRIVATE SweepCorridor::SweepCorridor ${OMPL_LIBRARIES})
  target_link_directories(corridor_firi_benchmark PRIVATE ${OMPL_LIBRARY_DIRS})
endif()
include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_executable(direct_search_test tests/direct_search_test.cpp)
  target_link_libraries(direct_search_test PRIVATE SweepCorridor::SweepCorridor ${OMPL_LIBRARIES})
  target_link_directories(direct_search_test PRIVATE ${OMPL_LIBRARY_DIRS})
  add_test(NAME direct_search_test COMMAND direct_search_test)
  if(GCOPTER_BUILD_CORRIDOR_BENCHMARK)
    add_executable(corridor_volume_test tests/corridor_volume_test.cpp)
    target_link_libraries(corridor_volume_test PRIVATE SweepCorridor::SweepCorridor)
    add_test(NAME corridor_volume_test COMMAND corridor_volume_test)
  endif()
endif()

install(TARGETS global_planning direct_corridor_demo
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
