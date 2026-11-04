include(CMakeFindDependencyMacro)
# Both header packages are installed together, including in isolated ROS prefixes.
find_dependency(SweepCorridor CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/../../.." NO_DEFAULT_PATH)
