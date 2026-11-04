# C3 自由边界膨胀

这里比较固定端点的原 C3（`4a210a1`）与当前六变量 C3。实验直接调用正式头文件库；没有独立的生成器副本或端部补丁。

每个控制块由三维中心位移和二维 SPD 形状矩阵组成。所有中心共同优化，原始起终点通过局部截面反求保持在走廊中。目标为

```text
E(center, shape) - 0.1 * sum(w_i * log det S_i) - log V
    - (0.01 / N) * sum(log(progress_slack)) + barriers
V = pi * integral dot(cross(M0, M1), c') du
```

`V` 使用五点 Gauss 精确积分，含全局重叠的重复体积；最终比较使用独立的三维并集 QMC。中心和形状的弯曲权重均为 `3e-5`。参考标架和原路径扩张盒固定，分离平面按原流程更新两轮。端部不增加种子点或 knot，也不指定延伸距离。

`progress_slack` 是已有局部进度锥的 Lorentz 行列式 `t²-||r||²`。其固定权重在障碍参数下降后仍保留，避免优化把局部纵向间距压到接近零。该项作用于整条走廊，直接复用约束梯度、Hessian 和精确线搜索，不增加变量或约束遍历。

支撑、形状和偏移项使用解析梯度/Hessian。体积采用精确梯度与正半定秩一曲率，起终点包含项采用隐函数梯度与 Gauss–Newton 曲率；因此整体是稀疏拟牛顿法，不是全局凸问题。半带宽为 23，秩一更新通过 Sherman–Morrison 求解。两端闭端面不等式使用小型活动集求解，允许原点恰好位于端面。每个接受步通过完整截面 Jacobian 认证。

## 重放

从仓库根目录执行：

```bash
python3 SweepCorridor/experiments/free_boundary/build.py --targets tests,benchmark,baseline,measure
prlimit --as=4294967296 -- taskset -c 2 .scratch/free_boundary/tests
python3 SweepCorridor/experiments/free_boundary/run.py \
  --output .scratch/free_boundary/final_development --repeats 7
python3 SweepCorridor/experiments/free_boundary/run.py \
  --output .scratch/free_boundary/final_regression --cases "$(seq -s, 7 66)" --repeats 3
python3 SweepCorridor/experiments/free_boundary/report.py
```

构建与运行串行执行，固定 CPU 2，进程地址空间限制 4 GiB，数值库线程数为 1。基线头文件从 Git 的 `4a210a1` 提取。输入快照及其 SHA256 见 `docs/validation/inflation_round1/inputs.json`；它们引用本地内容寻址地图和冻结种子路径，不重新运行搜索。移到另一台机器时，需要复制这些快照并更新清单路径，内容哈希应保持一致。

每种方法先预热一次，再记录完整重复序列。`benchmark.cpp` 在计时外重新检查连续支撑、全截面 Jacobian、起终点包含及 C² 接缝。`measure.cpp` 使用独立反求与 8 次随机数字平移，每次 65,536 个 Sobol 样本；不允许未决根。失败案例保留在汇总中。

`boundary_bending` 是原路径长度归一化参数下的边界二阶导数能量。报告另计算纵向边界曲线的 `integral kappa² ds`，以区分曲面形状与纵向参数分配的影响；它不是整个曲面的薄壳弯曲能量。`core_union` 仅统计原起终点所在截面之间的空间。阶段诊断可设置 `SWEEP_PROFILE=1`，常规计时不启用内部阶段时钟。

`geometry_metrics.py` 还计算原起终点区间侧壁的 `integral (k1²+k2²) dA`，其中 `k1,k2` 是主曲率。它不依赖纵向速度或截面角度的重新参数化，也不包括端盖边缘。默认每段 24 点 Gauss、48 个角度；尖锐位置需要增加采样确认积分稳定性。

当前固定预算为每轮 `8+3+3` 步、两轮分离面更新。报告保存 `termination`，达到预算不等同于数值收敛。测试仅涉及走廊生成与查询，没有运行轨迹优化。

历史手工延伸、追加封端和特化预算代码已从源码移除，旧代码及结果保存在 `docs/validation/inflation_round1/`。当前结果见 [自由边界实验](../../../docs/C3自由边界自然膨胀实验.md)。
