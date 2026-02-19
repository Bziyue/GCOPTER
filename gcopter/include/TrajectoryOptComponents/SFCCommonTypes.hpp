#ifndef SPLINE_TRAJECTORY_SFC_COMMON_TYPES_HPP
#define SPLINE_TRAJECTORY_SFC_COMMON_TYPES_HPP

#include <vector>
#include <Eigen/Eigen>

namespace gcopter
{
using PolyhedronV = Eigen::Matrix3Xd;
using PolyhedronH = Eigen::MatrixX4d;
using PolyhedraV = std::vector<PolyhedronV>;
using PolyhedraH = std::vector<PolyhedronH>;
}

#endif
