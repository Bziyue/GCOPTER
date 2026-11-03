// SPDX-License-Identifier: MIT
// GCOPTER-derived optimizer; Copyright (c) 2021 Zhepei Wang and Fei Gao.
// License text: third_party/GCOPTER/LICENSE. Project-maintained refactor: 2026-09-18.
#ifndef LBFGS_HPP
#define LBFGS_HPP

#include <Eigen/Eigen>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lbfgs
{
    // ----------------------- Data Type Part -----------------------

    /** @brief 有限括区间内的选点策略；接受条件由同一线搜索内核决定。 */
    enum class LineSearchInterpolation { Bisection, Cubic };

    /** @brief 通用 L-BFGS 参数；梯度容差与目标/变量尺度有关，默认不以代价停滞退出。 */
    struct lbfgs_parameter_t
    {
        /**
         * The number of corrections to approximate the inverse hessian matrix.
         *  The L-BFGS routine stores the computation results of previous m
         *  iterations to approximate the inverse hessian matrix of the current
         *  iteration. This parameter controls the size of the limited memories
         *  (corrections). The default value is 128. Values less than 3 are
         *  not recommended. Large values will result in excessive computing time.
         */
        int mem_size = 128;

        /**
         * @brief 真实梯度无穷范数的绝对容差，默认 1e-5；零仅接受精确零梯度。
         * 不除以变量范数，避免坐标平移导致伪收敛。容差单位随目标/变量的尺度变化，
         * 调用者应先合理缩放问题；非光滑目标的此条件不是广义驻点证书。
         */
        double g_epsilon = 1.0e-5;

        /**
         * Distance for delta-based convergence test.
         *  This parameter determines the distance, in iterations, to compute
         *  the rate of decrease of the cost function. If the value of this
         *  parameter is zero, the library does not perform the delta-based
         *  stopping test. The default value is 0 (disabled).
         */
        int past = 0;

        /** @brief 调用方提供的相对函数评价误差估计；零仅保留 double 算术余量，不是普适误差界。 */
        double function_relative_error = 0.0;
        /** @brief 一次求解最多接受的噪声尾部步数；零关闭近似接受，默认两步。 */
        int max_noise_steps = 2;

        /**
         * Delta for convergence test.
         *  This parameter determines the minimum rate of decrease of the
         *  cost function. The library stops iterations when the following 
         *  condition is met:
         *      |f' - f| / max(1, |f|) < delta,
         *  where f' is the cost value of past iterations ago, and f is
         *  the cost value of the current iteration.
         *  The default value is 1.0e-6.
         */
        double delta = 1.0e-6;

        /**
         * The maximum number of iterations.
         *  The lbfgs_optimize() function terminates an minimization process with
         *  ::LBFGSERR_MAXIMUMITERATION status code when the iteration count
         *  exceedes this parameter. Setting this parameter to zero continues an
         *  minimization process until a convergence or error. The default value
         *  is 0.
         */
        int max_iterations = 0;

        /**
         * The maximum number of trials for the line search.
         *  This parameter limits attempted steps, including expansions without a
         *  representable displacement, and bounds function/gradient evaluations. An iteration can retry once after a failed search,
         *  so it can use at most twice this many evaluations. The default is 64.
         */
        int max_linesearch = 64;

        /**
         * The minimum step of the line search routine.
         *  The default value is 1.0e-20. This value need not be modified unless
         *  the exponents are too large for the machine being used, or unless the
         *  problem is extremely badly scaled (in which case the exponents should
         *  be increased).
         */
        double min_step = 1.0e-20;

        /**
         * The maximum step of the line search.
         *  The default value is 1.0e+20. This value need not be modified unless
         *  the exponents are too large for the machine being used, or unless the
         *  problem is extremely badly scaled (in which case the exponents should
         *  be increased).
         */
        double max_step = 1.0e+20;

        /**
         * A parameter to control the accuracy of the line search routine.
         *  The default value is 1.0e-4. This parameter should be greater
         *  than zero and smaller than 1.0.
         */
        double f_dec_coeff = 1.0e-4;

        /**
         * A parameter to control the accuracy of the line search routine.
         *  The default value is 0.9. If the function and gradient
         *  evaluations are inexpensive with respect to the cost of the
         *  iteration (which is sometimes the case when solving very large
         *  problems) it may be advantageous to set this parameter to a small
         *  value. A typical small value is 0.1. This parameter should be
         *  greater than the f_dec_coeff parameter and smaller than 1.0.
         */
        double s_curv_coeff = 0.9;

        /**
         * @brief Li–Fukushima 谨慎更新系数，默认 1e-6，允许非负有限值。
         * 新曲率必须满足正性、有限性及相对步长/梯度阈值；拒绝时保留旧有效历史。
         * 浮点、非光滑、定义域失败或有限预算下不由此提供全局收敛保证。
         */
        double cautious_factor = 1.0e-6;

        /**
         * The machine precision for floating-point values. The default is 1.0e-16. 
         *  This parameter must be a positive value set by a client program to
         *  estimate the machine precision.
         */
        double machine_prec = 1.0e-16;

        /** @brief 选点策略；默认二分，Cubic 使用保护三次插值，共用弱/近似 Wolfe 条件。 */
        LineSearchInterpolation interpolation = LineSearchInterpolation::Bisection;
    };

    /**
     * Return values of lbfgs_optimize().
     *  Roughly speaking, a negative value indicates an error.
     */
    enum
    {
        /** L-BFGS reaches convergence. */
        LBFGS_CONVERGENCE = 0,
        /** L-BFGS satisfies stopping criteria. */
        LBFGS_STOP,
        /** The iteration has been canceled by the monitor callback. */
        LBFGS_CANCELED,
        /** 噪声尾部预算耗尽；保留接受点，未证明梯度收敛。 */
        LBFGS_NOISE_LIMIT,

        /** Unknown error. */
        LBFGSERR_UNKNOWNERROR = -1024,
        /** Invalid number of variables specified. */
        LBFGSERR_INVALID_N,
        /** Invalid parameter lbfgs_parameter_t::mem_size specified. */
        LBFGSERR_INVALID_MEMSIZE,
        /** Invalid parameter lbfgs_parameter_t::g_epsilon specified. */
        LBFGSERR_INVALID_GEPSILON,
        /** Invalid parameter lbfgs_parameter_t::past specified. */
        LBFGSERR_INVALID_TESTPERIOD,
        /** Invalid parameter lbfgs_parameter_t::delta specified. */
        LBFGSERR_INVALID_DELTA,
        /** Invalid parameter lbfgs_parameter_t::min_step specified. */
        LBFGSERR_INVALID_MINSTEP,
        /** Invalid parameter lbfgs_parameter_t::max_step specified. */
        LBFGSERR_INVALID_MAXSTEP,
        /** Invalid parameter lbfgs_parameter_t::f_dec_coeff specified. */
        LBFGSERR_INVALID_FDECCOEFF,
        /** Invalid parameter lbfgs_parameter_t::s_curv_coeff specified. */
        LBFGSERR_INVALID_SCURVCOEFF,
        /** Invalid parameter lbfgs_parameter_t::machine_prec specified. */
        LBFGSERR_INVALID_MACHINEPREC,
        /** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
        LBFGSERR_INVALID_MAXLINESEARCH,
        /** The function value became NaN or Inf. */
        LBFGSERR_INVALID_FUNCVAL,
        /** The line-search step became smaller than lbfgs_parameter_t::min_step. */
        LBFGSERR_MINIMUMSTEP,
        /** The line-search step became larger than lbfgs_parameter_t::max_step. */
        LBFGSERR_MAXIMUMSTEP,
        /** Line search reaches the maximum, assumptions not satisfied or precision not achievable.*/
        LBFGSERR_MAXIMUMLINESEARCH,
        /** The algorithm routine reaches the maximum number of iterations. */
        LBFGSERR_MAXIMUMITERATION,
        /** Relative search interval width is at least lbfgs_parameter_t::machine_prec. */
        LBFGSERR_WIDTHTOOSMALL,
        /** A logic error (negative line-search step) occurred. */
        LBFGSERR_INVALIDPARAMETERS,
        /** The current search direction increases the cost function value. */
        LBFGSERR_INCREASEGRADIENT,
        /** 评价前后检测到取消；恢复最后接受点，不再尝试缩步。 */
        LBFGSERR_CANCELED,
        /** 已无可表示的新试探点，非梯度收敛。 */
        LBFGSERR_PRECISION,

    };

    /**
     * Callback interface to provide cost function and gradient evaluations.
     *
     *  The lbfgs_optimize() function call this function to obtain the values of cost
     *  function and its gradients when needed. A client program must implement
     *  this function to evaluate the values of the cost function and its
     *  gradients, given current values of variables.
     *  
     *  @param  instance    The user data sent for lbfgs_optimize() function by the client.
     *  @param  x           The current values of variables.
     *  @param  g           The gradient vector. The callback function must compute
     *                      the gradient values for the current variables.
     *  @retval double      The value of the cost function for the current variables.
     */
    typedef double (*lbfgs_evaluate_t)(void *instance,
                                       const Eigen::VectorXd &x,
                                       Eigen::VectorXd &g);

    /**
     * Callback interface to provide an upper bound at the beginning of the current line search.
     *
     *  The lbfgs_optimize() function call this function to obtain the values of the
     *  upperbound of the stepsize to search in, provided with the beginning values of
     *  variables before the line search, and the current step vector (can be descent direction). 
     *  A client program can implement this function for more efficient linesearch. Any step 
     *  larger than this bound should not be considered. For example, it has a very large or even 
     *  inf function value. Note that the function value at the provided bound should be FINITE!
     *  If it is not used, just set it nullptr.
     *  
     *  @param  instance    The user data sent for lbfgs_optimize() function by the client.
     *  @param  xp          The values of variables before current line search.
     *  @param  d           The step vector. It can be the descent direction.
     *  @retval double      The upperboud of the step in current line search routine,
     *                      such that (stpbound * d) is the maximum reasonable step.
     */
    typedef double (*lbfgs_stepbound_t)(void *instance,
                                        const Eigen::VectorXd &xp,
                                        const Eigen::VectorXd &d);

    /**
     * Callback interface to monitor the progress of the minimization process.
     *
     *  The lbfgs_optimize() function call this function for each iteration. Implementing
     *  this function, a client program can store or display the current progress
     *  of the minimization process. If it is not used, just set it nullptr.
     *
     *  @param  instance    The user data sent for lbfgs_optimize() function by the client.
     *  @param  x           The current values of variables.
     *  @param  g           The current gradient values of variables.
     *  @param  fx          The current value of the cost function.
     *  @param  step        The line-search step used for this iteration.
     *  @param  k           The iteration count.
     *  @param  ls          The number of evaluations called for this iteration.
     *  @retval int         Zero to continue the minimization process. Returning a
     *                      non-zero value will cancel the minimization process.
     */
    typedef int (*lbfgs_progress_t)(void *instance,
                                    const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &g,
                                    const double fx,
                                    const double step,
                                    const int k,
                                    const int ls);

    /**
     * Callback data struct
     */
    struct callback_data_t
    {
        void *instance = nullptr;
        lbfgs_evaluate_t proc_evaluate = nullptr;
        lbfgs_stepbound_t proc_stepbound = nullptr;
        lbfgs_progress_t proc_progress = nullptr;
    };

  /** @brief 单次同步求解统计；评价数只计实际 f/g 回调，重启仍属同一求解。 */
  struct lbfgs_statistics_t
  {
    int evaluations{0};
    int iterations{0};
    int invalid_evaluations{0};
    int line_search_restarts{0};
    int rejected_curvature_pairs{0};
    int noise_steps{0};
  };

  /**
   * @brief 同步评价的取消与接受通知；空成员禁用对应功能，全部指针仅在 minimize 期间借用。
   * @note canceled 在每次完整评价前后检查，不能抢占回调；accepted 在初值和接受步调用，
   *       此时外部评价缓存与 x/f/g 一致。evaluation 是从一开始的实际回调编号。
   */
  struct lbfgs_control_t
  {
    void *instance{nullptr};
    bool (*canceled)(void *){nullptr};
    void (*accepted)(void *, const Eigen::VectorXd &, const Eigen::VectorXd &, double, int){nullptr};
  };

  namespace detail
  {
    /** @brief 校验求解控制量的范围与有限性。 @param p 求解参数。 @return 零或对应负错误码。 */
    inline int Validate(const lbfgs_parameter_t &p)
    {
      if (p.mem_size <= 0) return LBFGSERR_INVALID_MEMSIZE;
      if (!std::isfinite(p.g_epsilon) || p.g_epsilon < 0) return LBFGSERR_INVALID_GEPSILON;
      if (p.past < 0) return LBFGSERR_INVALID_TESTPERIOD;
      if (!std::isfinite(p.delta) || p.delta < 0) return LBFGSERR_INVALID_DELTA;
      if (!std::isfinite(p.min_step) || p.min_step < 0) return LBFGSERR_INVALID_MINSTEP;
      if (!std::isfinite(p.max_step) || p.max_step <= 0 || p.max_step < p.min_step)
        return LBFGSERR_INVALID_MAXSTEP;
      if (!(p.f_dec_coeff > 0 && p.f_dec_coeff < 1)) return LBFGSERR_INVALID_FDECCOEFF;
      if (!(p.s_curv_coeff > p.f_dec_coeff && p.s_curv_coeff < 1)) return LBFGSERR_INVALID_SCURVCOEFF;
      if (!std::isfinite(p.machine_prec) || p.machine_prec <= 0) return LBFGSERR_INVALID_MACHINEPREC;
      if (p.max_linesearch <= 0) return LBFGSERR_INVALID_MAXLINESEARCH;
      if (!std::isfinite(p.function_relative_error) || p.function_relative_error < 0 || p.max_noise_steps < 0)
        return LBFGSERR_INVALIDPARAMETERS;
      if (p.max_iterations < 0 || !std::isfinite(p.cautious_factor) || p.cautious_factor < 0 ||
          (p.interpolation != LineSearchInterpolation::Bisection &&
           p.interpolation != LineSearchInterpolation::Cubic))
        return LBFGSERR_INVALIDPARAMETERS;
      return 0;
    }

    /**
     * @brief 完整评价真实目标与梯度，以 NaN 哨兵识别未写入的分量。
     * @param evaluate 同步借用的可调用对象；必须完整写入梯度，异常原样传播。
     * @param x 当前变量，不施加额外坐标变换或梯度裁剪。
     * @param[out] g 同维梯度；非法结果不供曲率更新使用。
     * @param[out] value 目标值，可非有限以表示定义域外试探。
     * @param[in,out] stats 本次独占统计，调用前递增评价数。
     * @param control 可空同步控制，借用至本次调用结束。
     * @return 有效为 1，非法试探为 0，取消为 -1；取消不计入非法评价。
     */
    template <class Evaluate>
    int EvaluatePoint(Evaluate &evaluate, const Eigen::VectorXd &x,
                       Eigen::VectorXd &g, double &value, lbfgs_statistics_t &stats,
                       const lbfgs_control_t *control = nullptr)
    {
      const auto canceled = [&]() {
        return control && control->canceled && control->canceled(control->instance);
      };
      if (canceled()) return -1;
      g.setConstant(x.size(), std::numeric_limits<double>::quiet_NaN());
      ++stats.evaluations;
      value = evaluate(x, g);
      if (canceled()) return -1;
      const bool valid = x.allFinite() && std::isfinite(value) && g.size() == x.size() && g.allFinite();
      if (!valid) ++stats.invalid_evaluations;
      return valid;
    }

    struct LinePoint
    {
      double step;
      double value;
      double slope;
    };

    /**
     * @brief 在有限括区间中选择内点；三次模型失效时依次退回导数割线和二分。
     * @param lo 下界点，具有有限值和负方向导数。
     * @param hi 上界点，可因定义域失败而具有非有限值。
     * @param method 插值策略；两种策略使用相同的 Wolfe 接受与失败条件。
     * @param precision 目标评价相对舍入尺度估计；抵消区间仅用导数割线或二分。
     * @return 区间内候选步长，插值限定在区间的 10%–90% 内；精度耗尽由搜索层拒绝。
     * @note 系数按最大斜率尺度归一化，避免判别式平方溢出；模型不是接受条件。
     */
    inline double Interpolate(const LinePoint &lo, const LinePoint &hi,
                              LineSearchInterpolation method, double precision)
    {
      const double width = hi.step - lo.step;
      const double midpoint = lo.step + 0.5 * width;
      if (method == LineSearchInterpolation::Bisection) return midpoint;

      double proposed = std::numeric_limits<double>::quiet_NaN();
      const double noise = 8 * precision * std::max({1.0, std::abs(lo.value), std::abs(hi.value)});
      if (std::isfinite(hi.value) && std::isfinite(hi.slope) &&
          std::abs(hi.value - lo.value) > noise)
      {
        const double secant = (hi.value - lo.value) / width;
        const double scale = std::max({std::abs(secant), std::abs(lo.slope), std::abs(hi.slope)});
        if (std::isfinite(scale) && scale > 0)
        {
          const double u = lo.slope / scale, v = hi.slope / scale, z = secant / scale;
          const double a = u + v - 2 * z, b = 3 * z - 2 * u - v;
          const double discriminant = b * b - 3 * a * u;
          if (discriminant >= 0)
          {
            // 稳定求解三次模型的一阶导数；同时覆盖 b 两种符号下的抵消。
            const double root = std::sqrt(discriminant);
            const double q = -b - std::copysign(root, b);
            const auto take_minimum = [&](double t) {
              if (t > 0 && t < 1 && 6*a*t + 2*b > 0) proposed = lo.step + t*width;
            };
            if (a == 0)
            {
              if (b > 0) take_minimum(-u/(2*b));
            }
            else
            {
              take_minimum(q/(3*a));
              if (q != 0) take_minimum(u/q);
            }
          }
        }
      }
      if (!std::isfinite(proposed) && lo.slope < 0 && hi.slope > 0 && std::isfinite(hi.slope))
        proposed = lo.step - lo.slope / (hi.slope - lo.slope) * width;
      return std::isfinite(proposed)
                 ? std::clamp(proposed, lo.step + 0.1 * width, hi.step - 0.1 * width)
                 : midpoint;
    }

    /**
     * @brief 同一目标上的有界弱/近似 Wolfe 搜索，允许定义域外试探缩步。
     * @param[in,out] x 输出最后试探点，失败由外层恢复接受点。
     * @param[in,out] value 输入起点代价，输出最后试探代价。
     * @param[out] g 最后试探梯度。
     * @param[in,out] step 初始步长和最后试探步长，调用方须先约束初值。
     * @param direction 有限且严格下降的搜索方向。
     * @param xp 有限接受点，调用期间只读。
     * @param gp 与 xp 对应的有限同维梯度。
     * @param minimum 最小步长。
     * @param maximum 最大步长，不越过调用方声明的有效域上界。
     * @param evaluate 同步借用的真实目标/梯度回调。
     * @param p Wolfe、区间精度和评价次数限制。
     * @param[in,out] stats 当前求解的累计计数。
     * @param control 可空评价控制；取消优先于定义域失败。
     * @param allow_noise 是否允许噪声尾部；minimize 在首次严格下降前禁用。
     * @return 正数为本次搜索评价数；负数为数值、方向、取消或搜索边界失败。
     */
    template <class Evaluate>
    int LineSearch(Eigen::VectorXd &x, double &value, Eigen::VectorXd &g, double &step,
                   const Eigen::VectorXd &direction, const Eigen::VectorXd &xp,
                   const Eigen::VectorXd &gp, double minimum, double maximum,
                   Evaluate &evaluate, const lbfgs_parameter_t &p, lbfgs_statistics_t &stats,
                   const lbfgs_control_t *control = nullptr, bool allow_noise = true)
    {
      if (!(step > 0) || step < minimum || step > maximum) return LBFGSERR_INVALIDPARAMETERS;
      const double initial_slope = gp.dot(direction), initial_value = value;
      if (!std::isfinite(initial_slope) || !(initial_slope < 0)) return LBFGSERR_INCREASEGRADIENT;
      const double decrease = p.f_dec_coeff * initial_slope;
      const double curvature = p.s_curv_coeff * initial_slope;
      // 只允许有限次、梯度确实改善的尾部修整；评价误差须由调用方显式给出。
      const double roundoff = (8.0 * std::numeric_limits<double>::epsilon() + p.function_relative_error) *
                              std::max(1.0, std::abs(initial_value));
      LinePoint lo{0, initial_value, initial_slope};
      LinePoint hi{maximum, std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::quiet_NaN()};
      bool bracketed = false, touched_maximum = false;

      const int evaluations_before = stats.evaluations;
      for (int trial = 1; trial <= p.max_linesearch; ++trial)
      {
        x = xp + step * direction;
        if ((x.array() == xp.array()).all())
        {
          // 未括区时先扩大到可表示位移；括区后不得反复评价同一个接受点。
          if (bracketed || step >= maximum) return LBFGSERR_PRECISION;
          const double larger = std::min(2 * step, maximum);
          if (larger == step) return LBFGSERR_PRECISION;
          step = larger;
          continue;
        }
        const int evaluation = EvaluatePoint(evaluate, x, g, value, stats, control);
        if (evaluation < 0) return LBFGSERR_CANCELED;
        bool valid = evaluation > 0;
        const double slope = valid ? g.dot(direction) : std::numeric_limits<double>::quiet_NaN();
        if (valid && !std::isfinite(slope))
        {
          ++stats.invalid_evaluations;
          valid = false;
        }
        if (!valid)
        {
          hi = {step, std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()};
          bracketed = true;
        }
        else
        {
          // 先作差再减线性项，避免巨大常数把 Armijo 下降量直接舍入掉。
          const long double predicted = -static_cast<long double>(step) * initial_slope;
          const long double residual = static_cast<long double>(value) - initial_value -
                                       static_cast<long double>(step) * decrease;
          const bool ordinary = residual <= 0 && slope >= curvature;
          const bool noise_tail = allow_noise && stats.noise_steps < p.max_noise_steps &&
                                  p.f_dec_coeff < 0.5 && predicted <= 64.0L * roundoff;
          const bool approximate = noise_tail && residual <= roundoff && slope >= curvature &&
              slope <= (2*p.f_dec_coeff-1)*initial_slope &&
              g.cwiseAbs().maxCoeff() <= 0.9 * gp.cwiseAbs().maxCoeff();
          if (ordinary || approximate)
          {
            if (!ordinary) ++stats.noise_steps;
            return stats.evaluations - evaluations_before;
          }
          if (residual <= (noise_tail ? roundoff : 0.0) && slope < curvature)
            lo = {step, value, slope};
          else
          {
            hi = {step, value, slope};
            bracketed = true;
          }
        }

        if (trial == p.max_linesearch)
          return valid ? LBFGSERR_MAXIMUMLINESEARCH : LBFGSERR_INVALID_FUNCVAL;
        if (bracketed && hi.step - lo.step < p.machine_prec * hi.step)
          return LBFGSERR_WIDTHTOOSMALL;
        const double next = bracketed ? Interpolate(lo, hi, p.interpolation, p.machine_prec) : 2 * step;
        if (next < minimum || next == step) return LBFGSERR_MINIMUMSTEP;
        step = next;
        if (step > maximum)
        {
          if (touched_maximum) return LBFGSERR_MAXIMUMSTEP;
          touched_maximum = true;
          step = maximum;
        }
      }
      return LBFGSERR_MAXIMUMLINESEARCH;
    }

    /** @brief 单次求解独占的曲率环形缓冲；只在合法插入后覆盖旧历史。 */
    class LimitedMemory
    {
    public:
      LimitedMemory(int n, int capacity)
          : s_(n, capacity), y_(n, capacity), ys_(capacity), alpha_(capacity),
            pair_s_(n), pair_y_(n), capacity_(capacity) {}

      void Clear() { count_ = end_ = 0; scale_ = 1.0; }

      /**
       * @brief 更新有效曲率并计算预条件负梯度；被拒绝的新曲率不破坏既有历史。
       * @param x 当前接受变量。
       * @param xp 前一接受变量。
       * @param g 当前真实梯度。
       * @param gp 前一真实梯度。
       * @param cautious 非负谨慎更新系数，保持 Li–Fukushima 过滤语义。
       * @param[out] direction 下一搜索方向；算术退化时清空历史并回到真实负梯度。
       * @param[in,out] stats 本次独占统计，记录拒绝曲率对的数量。
       */
      void Direction(const Eigen::VectorXd &x, const Eigen::VectorXd &xp,
                     const Eigen::VectorXd &g, const Eigen::VectorXd &gp, double cautious,
                     Eigen::VectorXd &direction, lbfgs_statistics_t &stats)
      {
        pair_s_ = x - xp;
        pair_y_ = g - gp;
        const double ys = pair_y_.dot(pair_s_), yy = pair_y_.squaredNorm();
        const double threshold = pair_s_.squaredNorm() * gp.norm() * cautious;
        const double scale = yy > 0 ? ys / yy : 0.0;
        if (std::isfinite(ys) && std::isfinite(yy) && yy > 0 && std::isfinite(threshold) &&
            ys > std::max(0.0, threshold) && std::isfinite(scale) && scale > 0)
        {
          s_.col(end_) = pair_s_;
          y_.col(end_) = pair_y_;
          ys_(end_) = ys;
          scale_ = scale;
          if (++end_ == capacity_) end_ = 0;
          count_ = std::min(count_ + 1, capacity_);
        }
        else
          ++stats.rejected_curvature_pairs;

        direction = -g;
        if (count_ > 0 && x.size() == 1)
        {
          // 一维 BFGS 的逆 Hessian 更新恒等于最新有效割线 s/y；
          // 旧曲率项精确消去，无需遍历历史。过滤失败时仍保留前一次有效尺度。
          direction *= scale_;
        }
        else if (count_ > 0)
        {
          int index = end_;
          // 环形游标用比较回绕，避免在每对历史的两次遍历中做整数取模。
          for (int i = 0; i < count_; ++i)
          {
            if (--index < 0) index = capacity_ - 1;
            alpha_(index) = s_.col(index).dot(direction) / ys_(index);
            direction += (-alpha_(index)) * y_.col(index);
          }
          direction *= scale_;
          for (int i = 0; i < count_; ++i)
          {
            const double beta = y_.col(index).dot(direction) / ys_(index);
            direction += (alpha_(index) - beta) * s_.col(index);
            if (++index == capacity_) index = 0;
          }
        }
        const double slope = g.dot(direction);
        if (!direction.allFinite() || !std::isfinite(slope) || slope >= 0)
        {
          Clear();
          direction = -g;
        }
      }

    private:
      Eigen::MatrixXd s_, y_;
      Eigen::VectorXd ys_, alpha_, pair_s_, pair_y_;
      int capacity_, count_{0}, end_{0};
      double scale_{1.0};
    };

    inline bool Recoverable(int code)
    {
      return code == LBFGSERR_MAXIMUMLINESEARCH || code == LBFGSERR_INVALID_FUNCVAL ||
             code == LBFGSERR_MINIMUMSTEP || code == LBFGSERR_WIDTHTOOSMALL;
    }

    inline double InitialStep(const Eigen::VectorXd &direction)
    {
      const double norm = direction.norm();
      return 1.0 / (std::isfinite(norm) ? norm : direction.stableNorm());
    }
  } // namespace detail

  /**
   * @brief 使用可内联的真实 f/g 回调求解非凸无约束目标；所有工作区由本次调用独占。
   * @param[in,out] x 有限初值；有限初值评价后的失败恢复最后接受变量。
   * @param[out] f 返回 x 的目标值；参数非法时不修改，初值评价失败时可非有限。
   * @param evaluate 借用的可调用对象 double(const VectorXd &, VectorXd &)，须完整写入同维真实梯度。
   * @param p 求解参数；mem_size 是实际历史容量，不施加隐藏上限。
   * @param[out] output 可空、调用期间独占的统计指针；不取得所有权，每次调用重置。
   * @param stepbound 可空步长上界回调；正有限值或 +Inf，每次重启重新查询。
   * @param progress 可空进度回调；ls 包含本次迭代两次搜索的全部评价数。
   * @param instance 借用上下文，仅传递给 stepbound/progress；可空性由这些回调约定。
   * @param control 可空同步控制；评价前后取消返回负 CANCELED，初值/接受通知不增加评价。
   * @return CONVERGENCE 为梯度准则，STOP 为代价停滞，NOISE_LIMIT 为噪声尾部预算耗尽；均不保证轨迹可行。
   * @note 单次线搜索失败最多重启一次，使用同一目标/预算且仍需 Wolfe；不保证全局最优。
   *       成功最后一次评价与 x/f 一致；失败不恢复回调的外部缓存。异常原样传播。
   *       独立调用可并发，共享回调上下文或 output 的线程安全由调用者负责。
   */
  template <class Evaluate>
  int minimize(Eigen::VectorXd &x, double &f, Evaluate &&evaluate,
               const lbfgs_parameter_t &p = {}, lbfgs_statistics_t *output = nullptr,
               lbfgs_stepbound_t stepbound = nullptr, lbfgs_progress_t progress = nullptr,
               void *instance = nullptr, const lbfgs_control_t *control = nullptr)
  {
    lbfgs_statistics_t local;
    auto &stats = output ? *output : local;
    stats = {};
    if (x.size() <= 0) return LBFGSERR_INVALID_N;
    if (!x.allFinite()) return LBFGSERR_INVALIDPARAMETERS;
    if (const int error = detail::Validate(p); error != 0) return error;

    const int n = x.size();
    Eigen::VectorXd g(n), gp(n), xp(n), direction(n), past_values(p.past);
    detail::LimitedMemory memory(n, p.mem_size);
    const int initial = detail::EvaluatePoint(evaluate, x, g, f, stats, control);
    if (initial < 0) return LBFGSERR_CANCELED;
    if (initial == 0) return LBFGSERR_INVALID_FUNCVAL;
    const auto accepted = [&]() {
      if (control && control->accepted)
        control->accepted(control->instance, x, g, f, stats.evaluations);
    };
    accepted();
    const auto stationary = [&]()
    {
      return g.cwiseAbs().maxCoeff() <= p.g_epsilon;
    };
    if (stationary()) return LBFGS_CONVERGENCE;
    if (p.past > 0) past_values(0) = f;
    direction = -g;
    double suggested_step = detail::InitialStep(direction);

    for (;;)
    {
      xp = x;
      gp = g;
      const double previous_f = f;
      const int evaluations_before = stats.evaluations;
      int code = LBFGSERR_UNKNOWNERROR;
      double step = suggested_step;
      for (int attempt = 0; attempt < 2; ++attempt)
      {
        double maximum = p.max_step;
        if (stepbound)
        {
          const double bound = stepbound(instance, xp, direction);
          if (std::isnan(bound) || bound <= 0) { code = LBFGSERR_INVALIDPARAMETERS; break; }
          maximum = std::min(maximum, bound);
        }
        if (maximum < p.min_step) { code = LBFGSERR_MINIMUMSTEP; break; }
        step = std::max(p.min_step, std::min(suggested_step, 0.5 * maximum));
        code = detail::LineSearch(x, f, g, step, direction, xp, gp, p.min_step, maximum,
                                  evaluate, p, stats, control, stats.iterations > 0);
        if (code >= 0 || attempt != 0 || !detail::Recoverable(code)) break;

        x = xp;
        g = gp;
        f = previous_f;
        memory.Clear();
        direction = -g;
        suggested_step = detail::InitialStep(direction);
        ++stats.line_search_restarts;
      }
      if (code < 0)
      {
        x = xp;
        g = gp;
        f = previous_f;
        return code;
      }

      accepted();
      const int iteration = ++stats.iterations;
      if (progress && progress(instance, x, g, f, step, iteration, stats.evaluations - evaluations_before))
        return LBFGS_CANCELED;
      if (stationary()) return LBFGS_CONVERGENCE;
      if (stats.noise_steps > 0 && stats.noise_steps >= p.max_noise_steps) return LBFGS_NOISE_LIMIT;
      if (p.past > 0)
      {
        const int index = iteration % p.past;
        if (iteration >= p.past &&
            std::abs(past_values(index) - f) / std::max(1.0, std::abs(f)) < p.delta)
          return LBFGS_STOP;
        past_values(index) = f;
      }
      if (p.max_iterations > 0 && iteration >= p.max_iterations) return LBFGSERR_MAXIMUMITERATION;
      memory.Direction(x, xp, g, gp, p.cautious_factor, direction, stats);
      suggested_step = 1.0;
    }
  }

  /**
   * @brief GCOPTER 兼容入口，共用 minimize 内核，无独立的旧求解路径。
   * @param[in,out] x 有限初始/返回变量，失败时恢复最后接受点。
   * @param[out] f 返回目标值，参数非法时不修改。
   * @param evaluate 非空、同步借用的真实目标/完整梯度回调。
   * @param stepbound 可空步长上界回调，重启后重新查询。
   * @param progress 可空进度回调，非零返回中止。
   * @param instance 借用上下文，可空性由回调约定，不取得所有权。
   * @param p 求解参数。
   * @param[out] statistics 可空的本次独占统计，遵循 minimize 契约。
   * @return 与 minimize 相同；空 evaluate 返回 INVALIDPARAMETERS。
   */
  inline int lbfgs_optimize(Eigen::VectorXd &x, double &f, lbfgs_evaluate_t evaluate,
                            lbfgs_stepbound_t stepbound, lbfgs_progress_t progress, void *instance,
                            const lbfgs_parameter_t &p, lbfgs_statistics_t *statistics = nullptr)
  {
    if (!evaluate)
    {
      if (statistics) *statistics = {};
      return LBFGSERR_INVALIDPARAMETERS;
    }
    auto objective = [&](const Eigen::VectorXd &variables, Eigen::VectorXd &gradient)
    { return evaluate(instance, variables, gradient); };
    return minimize(x, f, objective, p, statistics, stepbound, progress, instance);
  }

  /**
   * @brief 兼容既有直接线搜索调用；与 minimize 共用选点/接受实现。
   * @param[in,out] x 最后试探变量；失败时本接口不回退。
   * @param[in,out] f 输入起点代价，输出最后试探代价。
   * @param[out] g 最后试探梯度。
   * @param[in,out] step 已位于 [minimum,maximum] 的正初始/返回步长。
   * @param direction 有限严格下降方向。
   * @param xp 有限搜索起点。
   * @param gp 起点的有限同维梯度。
   * @param minimum 非负步长下界。
   * @param maximum 有限正步长上界。
   * @param callback 借用的上下文，proc_evaluate 非空。
   * @param p 求解参数，interpolation 控制选点但不改变 Wolfe 条件。
   * @param[in,out] evaluations 可空借用计数器，累计本次实际回调数。
   * @return 正数为搜索评价数，负数表示失败；不保证外部缓存有效。
   */
  inline int line_search_lewisoverton(Eigen::VectorXd &x, double &f, Eigen::VectorXd &g,
                                      double &step, const Eigen::VectorXd &direction,
                                      const Eigen::VectorXd &xp, const Eigen::VectorXd &gp,
                                      double minimum, double maximum, const callback_data_t &callback,
                                      const lbfgs_parameter_t &p, int *evaluations = nullptr)
  {
    if (!callback.proc_evaluate || xp.size() <= 0 || gp.size() != xp.size() ||
        direction.size() != xp.size() || !xp.allFinite() || !gp.allFinite() ||
        !direction.allFinite() || !std::isfinite(f) || !std::isfinite(minimum) ||
        minimum < 0 || !std::isfinite(maximum) || maximum <= 0 || minimum > maximum)
      return LBFGSERR_INVALIDPARAMETERS;
    if (const int error = detail::Validate(p); error != 0) return error;
    auto objective = [&](const Eigen::VectorXd &variables, Eigen::VectorXd &gradient)
    { return callback.proc_evaluate(callback.instance, variables, gradient); };
    lbfgs_statistics_t stats;
    const int result = detail::LineSearch(x, f, g, step, direction, xp, gp, minimum, maximum,
                                          objective, p, stats);
    if (evaluations) *evaluations += stats.evaluations;
    return result;
  }

    /**
     * Get string description of an lbfgs_optimize() return code.
     *
     *  @param err          A value returned by lbfgs_optimize().
     */
    inline const char *lbfgs_strerror(const int err)
    {
        switch (err)
        {
        case LBFGSERR_PRECISION:
            return "No representable line-search displacement.";
        case LBFGS_NOISE_LIMIT:
            return "Function-noise progress limit; gradient tolerance not established.";
        case LBFGSERR_CANCELED:
            return "Evaluation canceled.";
        case LBFGS_CONVERGENCE:
            return "Success: reached convergence (g_epsilon).";

        case LBFGS_STOP:
            return "Success: met stopping criteria (past f decrease less than delta).";

        case LBFGS_CANCELED:
            return "The iteration has been canceled by the monitor callback.";

        case LBFGSERR_UNKNOWNERROR:
            return "Unknown error.";

        case LBFGSERR_INVALID_N:
            return "Invalid number of variables specified.";

        case LBFGSERR_INVALID_MEMSIZE:
            return "Invalid parameter lbfgs_parameter_t::mem_size specified.";

        case LBFGSERR_INVALID_GEPSILON:
            return "Invalid parameter lbfgs_parameter_t::g_epsilon specified.";

        case LBFGSERR_INVALID_TESTPERIOD:
            return "Invalid parameter lbfgs_parameter_t::past specified.";

        case LBFGSERR_INVALID_DELTA:
            return "Invalid parameter lbfgs_parameter_t::delta specified.";

        case LBFGSERR_INVALID_MINSTEP:
            return "Invalid parameter lbfgs_parameter_t::min_step specified.";

        case LBFGSERR_INVALID_MAXSTEP:
            return "Invalid parameter lbfgs_parameter_t::max_step specified.";

        case LBFGSERR_INVALID_FDECCOEFF:
            return "Invalid parameter lbfgs_parameter_t::f_dec_coeff specified.";

        case LBFGSERR_INVALID_SCURVCOEFF:
            return "Invalid parameter lbfgs_parameter_t::s_curv_coeff specified.";

        case LBFGSERR_INVALID_MACHINEPREC:
            return "Invalid parameter lbfgs_parameter_t::machine_prec specified.";

        case LBFGSERR_INVALID_MAXLINESEARCH:
            return "Invalid parameter lbfgs_parameter_t::max_linesearch specified.";

        case LBFGSERR_INVALID_FUNCVAL:
            return "The function value became NaN or Inf.";

        case LBFGSERR_MINIMUMSTEP:
            return "The line-search step became smaller than lbfgs_parameter_t::min_step.";

        case LBFGSERR_MAXIMUMSTEP:
            return "The line-search step became larger than lbfgs_parameter_t::max_step.";

        case LBFGSERR_MAXIMUMLINESEARCH:
            return "Line search reaches the maximum try number, assumptions not satisfied or precision not achievable.";

        case LBFGSERR_MAXIMUMITERATION:
            return "The algorithm routine reaches the maximum number of iterations.";

        case LBFGSERR_WIDTHTOOSMALL:
            return "Relative search interval width is at least lbfgs_parameter_t::machine_prec.";

        case LBFGSERR_INVALIDPARAMETERS:
            return "A logic error (negative line-search step) occurred.";

        case LBFGSERR_INCREASEGRADIENT:
            return "The current search direction increases the cost function value.";

        default:
            return "(unknown)";
        }
    }

} // namespace lbfgs

#endif
