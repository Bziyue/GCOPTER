// Imported regression from diff-planner-swarm 7e41f0d; include path adapted.
#include "gcopter/lbfgs.hpp"

#include <cassert>
#include <cmath>
#include <limits>

namespace
{
  /** @brief 不可计算的初值、残缺梯度和非法调用不能被误报为收敛。 */
  void CheckInvalidInitialEvaluation()
  {
    for (int scenario = 0; scenario < 4; ++scenario)
    {
      Eigen::VectorXd x = Eigen::VectorXd::Ones(2);
      double cost = 0.0;
      const auto evaluate = [](void *data, const Eigen::VectorXd &,
                               Eigen::VectorXd &g)
      {
        const int kind = *static_cast<int *>(data);
        if (kind == 0)
        {
          g.setZero();
          return std::numeric_limits<double>::infinity();
        }
        if (kind == 1) g.setConstant(std::numeric_limits<double>::quiet_NaN());
        if (kind == 2) g.setZero(1);
        if (kind == 3) g(0) = 0.0; // 第二维未写入，必须由 NaN 哨兵识别。
        return 1.0;
      };
      lbfgs::lbfgs_parameter_t parameters;
      const int code = lbfgs::lbfgs_optimize(x, cost, evaluate, nullptr, nullptr,
                                           &scenario, parameters);
      assert(code == lbfgs::LBFGSERR_INVALID_FUNCVAL);
    }
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    double cost = 0.0;
    assert(lbfgs::lbfgs_optimize(x, cost, nullptr, nullptr, nullptr, nullptr, {}) ==
           lbfgs::LBFGSERR_INVALIDPARAMETERS);
  }

  /** @brief 有限初值越过对数定义域后须缩步恢复，最后回调与返回变量、代价一致。 */
  void CheckDomainRecovery()
  {
    struct Context
    {
      int invalid{0};
      double last_x{0.0};
      double last_cost{0.0};
    } context;
    const auto evaluate = [](void *data, const Eigen::VectorXd &x,
                             Eigen::VectorXd &g)
    {
      auto &record = *static_cast<Context *>(data);
      record.last_x = x(0);
      if (x(0) <= 0.0)
      {
        ++record.invalid;
        g.setZero(1);
        return std::numeric_limits<double>::infinity();
      }
      g(0) = 2.0 * (x(0) - 0.02) - 0.001 / x(0);
      record.last_cost = std::pow(x(0) - 0.02, 2) - 0.001 * std::log(x(0));
      return record.last_cost;
    };
    Eigen::VectorXd x = Eigen::VectorXd::Constant(1, 0.1);
    double cost = 0.0;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.past = 0;
    parameters.g_epsilon = 1.0e-10;
    assert(lbfgs::lbfgs_optimize(x, cost, evaluate, nullptr, nullptr, &context,
                                parameters) ==
           lbfgs::LBFGS_CONVERGENCE);
    assert(context.invalid > 0);
    assert(std::abs(x(0) - (0.04 + std::sqrt(0.0096)) / 4.0) < 1.0e-8);
    assert(x(0) == context.last_x && cost == context.last_cost);
  }

  /** @brief 两次有界搜索均失败时恢复接受点的代价，不把外部拒绝点缓存冒充接受点。 */
  void CheckRollbackAndRetryBound()
  {
    struct Context
    {
      int calls{0};
      double last_x{0.0};
    } context;
    const auto evaluate = [](void *data, const Eigen::VectorXd &x,
                             Eigen::VectorXd &g)
    {
      auto &record = *static_cast<Context *>(data);
      ++record.calls;
      record.last_x = x(0);
      g.setConstant(1, 1.0);
      return x(0) == 0.0 ? 7.0 : std::numeric_limits<double>::infinity();
    };
    Eigen::VectorXd x = Eigen::VectorXd::Zero(1);
    lbfgs::lbfgs_parameter_t parameters;
    parameters.max_linesearch = 4;
    double cost = 0.0;
    const int code = lbfgs::lbfgs_optimize(x, cost, evaluate, nullptr, nullptr,
                                         &context, parameters);
    assert(code < 0 && x(0) == 0.0 && cost == 7.0);
    assert(context.calls == 1 + 2 * parameters.max_linesearch);
    assert(context.last_x != x(0));
  }

  /** @brief 重启必须重新查询动态步长界，并向 progress 报告两次搜索的总评价数。 */
  void CheckRestartBoundAndProgress()
  {
    struct Context
    {
      int evaluations{0};
      int bounds{0};
      int reports{0};
    } context;
    const auto evaluate = [](void *data, const Eigen::VectorXd &x,
                             Eigen::VectorXd &g)
    {
      ++static_cast<Context *>(data)->evaluations;
      g = x;
      return 0.5 * x.squaredNorm();
    };
    const auto bound = [](void *data, const Eigen::VectorXd &,
                          const Eigen::VectorXd &)
    {
      // 两个界均在二次目标的有效域内；首次过紧的界使曲率条件无法满足。
      return ++static_cast<Context *>(data)->bounds == 1 ? 0.01 : 2.0;
    };
    const auto progress = [](void *data, const Eigen::VectorXd &,
                             const Eigen::VectorXd &, double, double,
                             int iteration, int evaluations)
    {
      ++static_cast<Context *>(data)->reports;
      assert(iteration == 1 && evaluations == 2);
      return 0;
    };
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    double cost = 0.0;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.max_linesearch = 1;
    const int code = lbfgs::lbfgs_optimize(x, cost, evaluate, bound, progress,
                                         &context, parameters);
    assert(code == lbfgs::LBFGS_CONVERGENCE && x(0) == 0.0 && cost == 0.0);
    assert(context.bounds == 2 && context.evaluations == 3 && context.reports == 1);
  }

  /** @brief g_epsilon 为零时，迭代中精确零梯度必须在零方向搜索前退出。 */
  void CheckExactStationaryIterate()
  {
    const auto evaluate = [](void *, const Eigen::VectorXd &x, Eigen::VectorXd &g)
    {
      g = x;
      return 0.5 * x.squaredNorm();
    };
    Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
    double cost = 0.0;
    lbfgs::lbfgs_parameter_t parameters;
    parameters.g_epsilon = 0.0;
    parameters.past = 0;
    parameters.max_iterations = 4;
    assert(lbfgs::lbfgs_optimize(x, cost, evaluate, nullptr, nullptr, nullptr,
                                parameters) ==
           lbfgs::LBFGS_CONVERGENCE);
    assert(x(0) == 0.0 && cost == 0.0);
  }
}

/** @brief 运行独立求解器的定义域、梯度、恢复和有限重试回归。 @return 断言通过返回零。 */
int main()
{
  CheckInvalidInitialEvaluation();
  CheckDomainRecovery();
  CheckRollbackAndRetryBound();
  CheckRestartBoundAndProgress();
  CheckExactStationaryIterate();
}
