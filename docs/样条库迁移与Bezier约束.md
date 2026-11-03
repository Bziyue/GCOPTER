# 样条库迁移与 Bézier 轨迹约束

## 第一阶段：迁移最新四头

2026-11-03，从本机 diff-planner-swarm 的干净提交 `7e41f0d1d1771aadc3e36e447d321e3bea4e59ff` 原样迁入 SplineTrajectory、SplineOptimizer、SplineConvexHull 和修改版 L-BFGS。四头逐字节核对通过；许可证、来源路径和 SHA256 见 `third_party/SplineTrajectory/SOURCE.md`。迁移前实验基线独立保存于 `0b82fdb`。

主节点与对照实验均采用新版优化器自持工作区、SplineProblem/OptimizerOptions、具名代价阶段和 Polynomial 输出。主节点保留 Quintic 最小 jerk、原多面体路点映射与原动力学模型。旧 Context/Spec 调用已移除，接口文档同步更新。优化结束显式同步接受点，主节点失败时清空输出，实验仍记录终止状态与最终候选供诊断。

此阶段保留原积分点位置约束，以便把库迁移与下一阶段约束变化分开。L-BFGS 换用了源仓库的无效候选回溯、恢复和曲率处理；不承诺与旧版本逐次迭代相同。

### 验证

- ROS `global_planning` 与 `corridor_firi_benchmark` 编译通过。单进程编译：2 GiB 限额触发受控退出后，在 3 GiB 虚拟地址空间上限内完成；未无限制重试。
- 独立测试 `library_regression / native_optimizer / lbfgs_recovery` 全通过；Release 仍启用测试断言。
- 三次、五次、七次：含显式时间权重及 Bézier 转换时间依赖的全链式梯度相对误差分别 4.28e-10、2.66e-10、1.99e-9。
- 原生多面体空间映射、积分平坦性惩罚的完整梯度相对误差 1.44e-9；测试初值代价尺度很大，绝对误差 65.8，不能据此声称绝对误差也低。盒内轨迹优化成功并通过 1001 点检查。
- 六条冻结输入，两种走廊梯度检查均通过，最大相对误差 3.10e-8；输入向量哈希一致。每次求解 800 迭代、16 积分步、权重 1e5、预热一次和计时一次，峰值 RSS 28.3 MiB，运行虚拟内存限制 512 MiB。
- 本次短预算样本 Hermite 1/6、FIRI 0/6 通过既有采样可行性标准；多数达到迭代上限，三个 Hermite 个案线搜索停止。**这是库迁移回归，不是已解决约束可行性或得到性能提升的结论。**
- 空旷控制用例两种走廊都通过采样可行性，时长、长度与 jerk 能量一致到 1e-4 相对容差。

原始紧凑记录：`docs/validation/library_migration/phase1`、`phase1_smoke`；本机 CSV 和构建日志在 `.scratch/library_migration`。地图哈希保留于 manifest，未将大地图复制进 Git。

```sh
prlimit --as=3221225472 --cpu=180 cmake --build .scratch/trajectory_comparison/build -j1
prlimit --as=536870912 --cpu=60 ctest --test-dir .scratch/trajectory_comparison/build --output-on-failure -j1
python3 gcopter/experiments/trajectory/run.py --inputs gcopter/experiments/trajectory/inputs.json --output .scratch/library_migration/phase1 --repeats 1 --check-gradients
```
