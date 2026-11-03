# 本仓库的 SplineOptimizer 接入协议

对应 diff-planner-swarm `7e41f0d` 四头快照，来源及哈希见 `third_party/SplineTrajectory/SOURCE.md`。

优化器自己持有工作区。旧的外部 Context / Workspace、EvaluateSpec、makeEvaluateSpec 接口已移除；每个并发求解任务需要自己的优化器实例。

```cpp
using Spline = SplineTrajectory::QuinticSplineND<3>;
SplineTrajectory::SplineOptimizer<Spline> optimizer;
SplineTrajectory::SplineProblem<Spline> problem;
// 填写 durations、waypoints（N+1 行）、boundary、start_time。
SplineTrajectory::OptimizerOptions options;
options.energy_weight = 1;
options.integration_steps = 16;
optimizer.prepare(problem, options);
auto x = optimizer.initialGuess();
Eigen::VectorXd gradient(optimizer.dimension());
// Objective 按需声明 duration / integral / coefficient / parameter / decision 成员。
auto result = optimizer.evaluate(x, gradient, objective);
// 优化结束后显式恢复接受点；polynomial() 是借用视图。
optimizer.updateAccepted(x);
auto output = optimizer.copyPolynomial();
```

各阶段的代价返回 double，梯度累加到传入的预分配缓冲区：

| 成员 | 输入 | 梯度 |
|---|---|---|
| duration | `const std::vector<double>&` | `Eigen::Ref<Eigen::VectorXd>` |
| integral | `const IntegralPointInfo&, const SampleState<3>&` | `SampleGradient<3>&` |
| coefficient | `const ParameterView<Spline>&` | `CoefficientGradient<Spline>&` |
| parameter | `const ParameterView<Spline>&` | `ParameterGradient<Spline>&` |
| decision | `const DecisionView&` | `Eigen::Ref<Eigen::VectorXd>` |

`integral` 声明 `kDerivativeOrder` 可限制所需导数阶数；GCOPTER 的平坦性惩罚用 3 阶。
`coefficient` 在样条伴随前执行，适合 Bézier 控制点代价：`backwardAdd` 同时返回系数偏导和固定系数时的独立时长偏导，随后优化器统一传播到时长与内路点。
`parameter` 在样条伴随后执行；`decision` 最后执行，用于原有空间参数化的范数惩罚。

参数化包含 `time / space / auxiliary`。空间映射采用 `dimension(index)`、`toPhysical`、`toUnconstrained`、`backwardInto`；index 是含首尾的全局路点索引。GCOPTER 保留原有多面体空间映射与时间映射，三头迁移本身不改变走廊几何。

失败评估返回非有限代价及状态，交由迁入的 L-BFGS 回溯；不能伪造有限大数作为成功结果。接受点同步失败或优化失败时主节点不发布旧轨迹。

梯度检查调用自由函数 `SplineTrajectory::checkGradients(optimizer, x, objective)`。`relative_error` 为相对误差，返回状态使用绝对误差阈值，二者不要混用。
