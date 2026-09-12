#include "gcopter/geo_utils.hpp"
#ifdef BENCH_MINCO
#include "gcopter/gcopter.hpp"
using Optimizer = gcopter::GCOPTER_PolytopeSFC;
using Output = Trajectory<5>;
using Corridor = Optimizer::PolyhedraH;
#else
#include "gcopter/spline_sfc_optimizer.hpp"
using Optimizer = gcopter::SplineSFCOptimizer;
using Output = Optimizer::Polynomial;
using Corridor = gcopter::PolyhedraH;
#endif
#include <chrono>
#include <iomanip>
#include <iostream>

/** @brief Solve fixed overlapping box corridors without ROS or map input.
 * @param argc Optional trial count.
 * @param argv Arguments; the default is 21 trials for each perturbation.
 * @return Nonzero if setup, solve, or endpoint validation fails. */
int main(int argc, char **argv)
{
    const int repeats = argc > 1 ? std::stoi(argv[1]) : 21;
    std::cout << "perturbation,trial,prepare_us,solve_us,cost,duration,pieces\n" << std::setprecision(17);
    for (int perturbation : {-1,0,1})
    {
        Corridor corridor;
        for (int i = 0; i < 3; ++i)
        {
            Eigen::MatrixX4d box(6,4);
            box << -1,0,0,-1.0+2*i, 1,0,0,-3.0-2*i,
                    0,-1,0,-2, 0,1,0,-2, 0,0,-1,-2, 0,0,1,-2;
            // Halfspaces use n.dot(p) + offset <= 0.
            corridor.push_back(box);
        }
        Eigen::Matrix3d start = Eigen::Matrix3d::Zero(), end = start;
        end.col(0) << 6, 0.4 + 0.1*perturbation, 0.2;
        Eigen::VectorXd bounds(5), weights(5), physical(6);
        bounds << 4,2.1,1.05,2,12;
        weights << 1e4,1e4,1e4,1e4,1e5;
        physical << 0.61,9.8,0.7,0.8,0.01,0.0001;
        for (int trial = 0; trial < repeats; ++trial)
        {
            const auto begin = std::chrono::steady_clock::now();
            Optimizer optimizer;
            if (!optimizer.setup(20,start,end,corridor,INFINITY,1e-2,16,bounds,weights,physical)) return 1;
            const auto prepared = std::chrono::steady_clock::now();
            Output output;
            const double cost = optimizer.optimize(output,1e-5);
            const auto solved = std::chrono::steady_clock::now();
#ifdef BENCH_MINCO
            const double duration = output.getTotalDuration();
            const int pieces = output.getPieceNum();
            const Eigen::Vector3d endpoint = output.getPos(duration);
#else
            const double duration = output.endTime();
            const int pieces = output.numSegments();
            const Eigen::Vector3d endpoint = output.evaluate(duration);
#endif
            if (!std::isfinite(cost) || pieces < 1 || (endpoint-end.col(0)).norm() > 1e-7) return 2;
            std::cout << perturbation << ',' << trial << ','
                      << std::chrono::duration<double,std::micro>(prepared-begin).count() << ','
                      << std::chrono::duration<double,std::micro>(solved-prepared).count() << ','
                      << cost << ',' << duration << ',' << pieces << '\n';
        }
    }
}
