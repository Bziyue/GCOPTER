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
python3 gcopter/experiments/trajectory/run.py --inputs gcopter/experiments/trajectory/inputs.json --output .scratch/library_migration/phase1 --repeats 1 --check-gradients --constraint sampled
```


## 第二阶段：细分 Bézier 位置/速度约束

第一阶段已先独立提交为 `f90671d`，随后才引入以下约束变化。迁入的四个库头未再修改，约束适配放在 `gcopter/include/TrajectoryOptAdapters/BezierPenaltyCost.hpp`。

### 原生 GCOPTER 的接入

`SplineSFCOptimizer` 的目标现在分为时长、积分、系数和决策变量四个阶段：

1. `duration`：总时间代价。
2. `integral`：经平坦性映射计算的机体角速度、倾角、推力惩罚；位置与速度积分权重置零，避免重复计算这两项。
3. `coefficient`：位置与速度的细分 Bézier 控制点惩罚。
4. `decision`：原多面体空间映射的范数惩罚。

最小 jerk 能量由优化器自身累加，仍使用五次样条。`setup` 新增最后一个可选参数 `bezierSubdivisionDepth`，默认 2：每个源段四片，每片 6 个位置控制点、5 个速度控制点。速度直接从幂基导数转换，避免在很大的世界坐标中对位置控制点做差分。

多面体沿用 setup 中固定的 `hPolyIdx_`，一段及其细分片的所有控制点受同一个凸区域约束。归一化平面、工作区与控制点梯度缓冲区只在 setup 构造；每次迭代复用。没有增加逐控制点的拓扑校验；迁入库内部原有的数值和维度检查保留。

### 约束充分性与软惩罚边界

对于一片轨迹，

\[
p(s)=\sum_{i=0}^{d}B_i^d(s)b_i,\quad B_i^d(s)\ge0,\quad\sum_iB_i^d(s)=1.
\]

若同一片的所有位置控制点满足同一凸区域的所有平面 `nᵀb_i + d <= 0`，则该片整条曲线位于该凸区域。速度曲线同理：所有速度控制点在半径 `v_max` 的球内，足以保证片内速度上界。细分不改变曲线，只缩小用于判定的凸包。

但这里使用有限权重的软惩罚，优化器可能以非零控制点残差停止。**不能把采用 Bézier 表示等同于已有硬安全证书**。角速度、倾角和推力也没有由速度凸包自动获得连续时间保证，它们仍通过积分惩罚及独立采样审计处理。没有照搬参考工程的 A/J 限制，因为 GCOPTER 本次问题使用的是平坦性动力学限值；最小 jerk 能量也不是 jerk 上界。

### 梯度传播

设一段时长为 `T`，局部幂基系数为 `c`；转换和细分矩阵为 `A_l(T)`，控制点 `b_l=A_l(T)c`。每源段该阶控制点数量为 `m`，代价定义为

\[
J=\frac{T}{m}\sum_l\phi(b_l).
\]

控制点梯度、固定系数时的独立时长偏导分别为

\[
g_{b_l}=\frac{T}{m}\nabla\phi(b_l),\qquad
\partial_TJ=\frac{1}{m}\sum_l\phi(b_l)
+\sum_lg_{b_l}^{\mathsf T}\frac{\partial A_l}{\partial T}c.
\]

系数偏导为 `Σ A_lᵀ g_b_l`。第一项时间偏导在代价循环中累加；第二项及系数偏导由 `hull.backwardAdd` 累加。随后新版 `SplineOptimizer` 一次样条伴随传播到物理时长、内路点及边界，再通过时间映射和空间映射传播到 L-BFGS 变量。漏掉任一时间项都会产生错误梯度。

这里的 `T/m` 是控制点软代价的归一化权重，不是声称它等于原曲线罚项积分。原生多面体使用原 smoothed-L1 平面惩罚；同初值实验使用米制正部平方惩罚，因此两种后端的数值目标不能混称为相同。

### 同初值实验：凸多面体与非凸扫掠走廊的区别

运行器支持三种模式：

| 参数 | FIRI 位置 | Hermite 位置 | 速度 |
|---|---|---|---|
| `--constraint sampled` | 原积分点多面体并集罚项 | 原积分点隐式截面罚项 | 积分 |
| `--constraint bezier`（默认） | 固定细分片对应凸多面体的控制点罚项 | 控制点与曲线积分各半权重 | Bézier 导数控制点 |
| `--constraint bezier-controls` | 同上 | 仅控制点，作为消融 | Bézier 导数控制点 |

实验的 FIRI 关联在初始轨迹上一次确定：对一片全部控制点，选最大平面残差最小的多面体，随后固定。初始关联不保证可行，后续优化必须减少残差；没有让不同控制点各自选择不同的多面体。这与原生 GCOPTER 基于 corridor overlap 的分段初始化不同，仍不能把本实验称为原生 GCOPTER 的完整性能排名。

Hermite 扫掠走廊一般非凸。控制点分别位于其中不能推出曲线也位于其中，即使细分到四片也一样。最终二进制的纯控制点消融复现了这个问题：原输入 case_03 的控制点最大残差约 3 mm，实际曲线最大径向残差约 26 mm。原输入 case_00 也有约 2.8 mm 对 10.7 mm 的差距。局部隐式截面选根还可能存在分支切换。默认因此保留原曲线积分位置项，控制点项与积分项各取原位置权重的一半；这仍是软约束与采样审计，**不构成非凸扫掠走廊的连续包含证明**。

### 回归与实测

四个 CTest 均通过；新增测试覆盖活跃位置/速度罚项的完整梯度（相对误差 9.41e-10）、端点安全但中段越界、细分收紧上界，以及 de Casteljau 重建与原多项式一致。ROS 全部目标重新编译通过，并已安装到本仓库 `install/gcopter`；现有 `direct_search_test`、`corridor_volume_test` 通过。原生盒内轨迹及完整梯度检查通过。空旷控制用例两种走廊仍得到匹配的可行轨迹。

六个输入仍各预热一次、计时一次，800 次迭代上限、16 个积分间隔、位置总权重 1e5。下表是本机单次结果，不是重复统计的稳定性能结论；比较的是迁移后同一库下的 sampled 与最终默认 Bézier/混合模式。两边初始变量哈希保持一致。

| 输入 | FIRI sampled / Bézier (ms) | Hermite sampled / 混合 (ms) | FIRI 采样可行 | Hermite 采样可行 |
|---|---:|---:|---|---|
| case_00 | 168.2 / 124.5 | 876.5 / 2016.0 | 未通过 → 未通过 | 通过 → 通过 |
| case_01 | 337.3 / 174.2 | 1180.0 / 2129.3 | 未通过 → 未通过 | 未通过 → 未通过 |
| case_02 | 135.4 / 102.8 | 839.6 / 1950.5 | 未通过 → 未通过 | 未通过 → 未通过 |
| case_03 | 376.2 / 180.4 | 878.8 / 3659.8 | 未通过 → 未通过 | 未通过 → 未通过 |
| case_04 | 158.6 / 106.8 | 806.4 / 1922.9 | 未通过 → 通过 | 未通过 → 未通过 |
| case_05 | 77.4 / 57.0 | 423.2 / 965.4 | 未通过 → 通过 | 未通过 → 未通过 |

FIRI 采样可行数量从 0/6 到 2/6，Hermite 仍是 1/6。多数达到迭代上限，若干 Hermite 线搜索停止；不能把残差未通过的候选当作可发布安全轨迹。两种目标的可行性、质量和总时间必须联合看，不能仅比较优化器返回成功或更短耗时。

最终六例最大梯度相对误差为 3.10e-8，峰值 RSS 约 28.3 MiB，虚拟内存上限仍为 512 MiB。FIRI 本轮耗时减少，但 Hermite 混合模式明显变慢，且没有提高总体采样可行率，**不作为已经完成性能优化的结论**。新控制点残差字段与实际曲线审计字段分别记录；sampled 模式的控制点字段为未启用时的零值，不能用于安全判断。

紧凑原始结果位于 `docs/validation/library_migration/phase2`、`phase2_smoke`、`phase2_ablation`。消融仅使用原 case_00 与 case_03，manifest 保存了对应请求路径；不可将消融目录内部的 case_01 当作原六例的 case_01。所有测试、编译和地图实验串行执行。

```sh
python3 gcopter/experiments/trajectory/run.py \
  --inputs gcopter/experiments/trajectory/inputs.json \
  --output .scratch/library_migration/phase2 --repeats 1 --check-gradients \
  --constraint bezier --bezier-depth 2
```

下一步若要同时提高非凸走廊的速度和安全性，需要构造随进度分片的可认证凸内域，或对轨迹与扫掠走廊联合做区间包含判定；不能仅增加控制点数量就当作解决了非凸包含问题。本次没有更改走廊生成几何，也没有对原六个用例的失败结果做隐藏或过滤。

本机现有 ROS CTest 的 OMPL/ODE 依赖需要设置库路径（首次未设置时 direct_search_test 因缺少 libode.so.8 未启动，补齐路径后通过）：

```sh
LD_LIBRARY_PATH="$PWD/.scratch/ompl/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  prlimit --as=536870912 --cpu=60 ctest --test-dir build/gcopter --output-on-failure -j1
```
