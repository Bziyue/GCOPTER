# SweepCorridor 迁移与同输入验证

本轮以 `92da988` 的整理版本为参照，建立可独立迁移的 [SweepCorridor](../SweepCorridor) 头文件包。方案先提交在 `d139196`。

## 最终接口

```cpp
#include <SweepCorridor.hpp>
namespace sc = SweepCorridor;

sc::Options options;
sc::ObstacleIndex map(obstacles);
auto c3 = sc::Generator<3>(options).generate(seed, map, bounds);
auto h5 = sc::Generator<5>(options).generate(seed, map, bounds);

if (c3) {
    sc::Query<3> query(c3.corridor);
    auto section = query.section(u);
    auto constraint = query.evaluate(u, position);
}
```

两个次数共用相同的 `Corridor`、`Query`、`Result` 接口。查询返回中心、截面形状、一二阶导数、约束梯度与 Hessian，并保留截面平面残差。`Query` 使用预计算幂基和 Horner 求值，拥有缓存数据；每个查询线程持有独立对象。接口中的 u 由调用者提供，本次未把轨迹优化后端的全局截面反求迁入库。

输出只保存节点和原生次数的 Bézier 控制量。C3 每段四个控制量，H5 每段六个。H5 生成过程中的 Hermite 节点状态属于临时求解数据，不随最终走廊保存。默认算法常量留在内部，公共配置描述几何、安全、资源和显式计时。

## 实现与清理

- 独立包为 C++17/Eigen，CMake 导出 `SweepCorridor::SweepCorridor` INTERFACE target，没有 ROS、GCOPTER、OMPL、文件读写或轨迹优化依赖。
- C3 带状优化与 H5 节点优化分别保留，共用几何、障碍和证书基础设施。
- H5 固定 smooth 流程、全截面 Jacobian 和默认步长策略；保留第一轮 8 次、后续 12 次等实际求解预算，没有把各阶段粗暴统一。
- 删除 ShapeStage、任意形状后处理回调、投影初始化分支、诊断求导内核和历史调参模式。支撑平面直接通过结果返回。
- 删除原 `direct_corridor` 下重复的核心实现及历史实验源码；历史结果继续保留，源码可从 `92da988` 恢复。
- `direct_corridor` 仅作为 ROS/旧 GCOPTER 工具适配层。保留的轨迹查询适配头移到 `gcopter/include/gcopter/corridor_evaluator.hpp`，不进入迁移包。
- ROS 演示直接选择 `Generator<3>` 或 `Generator<5>`，使用相同查询接口绘制原生次数控制量，不再对 C3 升阶。现有启动命令仍有效。

## 同输入回归

冻结地图和种子路径，预热一次、正式重复七次；串行、CPU 2、单数值线程，进程地址空间限制 4 GiB。障碍索引构建不计入生成耗时。

| 方案 | 输入数量 | 清理前 / ms | SweepCorridor / ms | 变化 |
| --- | ---: | ---: | ---: | ---: |
| C3 | 67 | 1902.697 | 1781.222 | −6.38% |
| H5 smooth | 7 | 253.848 | 256.190 | +0.92% |

表中是**每个案例七次生成中位数之和**，不是单次请求平均值。C3 使用原有生成器内部 total 口径；H5 使用生成调用 wall 口径，包含统一输出转换。两行不能用来比较 C3 与 H5 谁更快，因为案例集合和口径不同。原始逐次记录均保留。

- C3 65 组成功、2 组预期初始化失败，前后状态一致。
- C3 成功输出的几何系数、支撑平面逐字节相同；Newton 步数、回退和预测步决策统计一致。
- H5 七组输出的几何系数、支撑平面逐字节相同，Newton 步数一致。
- 每个成功输出都复核连续支撑与全截面 Jacobian 证书。
- 因输出几何相同，体积、边界平滑性和自交情况也保持相同；没有重新采样体积来制造统计差异。
- 新程序运行时峰值 RSS：C3 最大约 32.8 MiB，H5 最大约 30.9 MiB。4 GiB 是地址空间限制，不是实测 RSS。

本轮 C3 有提速；H5 未体现提速，七例合计约慢 0.9%。这是代码组织和编译变化下的实测结果，没有降低安全标准或更改优化预算。

## 接口、安装与集成验证

- 多个翻译单元同时包含头文件，编译和链接通过。
- C3/H5 生成、连续证书、非法输入、种子碰撞与失败空输出测试通过。
- Horner 与原生 Bézier 的值、一二阶导数对照通过。
- 两种次数的约束梯度和 Hessian 有限差分检查通过。
- 安装后的独立消费工程仅通过 `find_package(SweepCorridor)` 和 Eigen 构建，生成及查询成功。
- 旧工具适配、ROS 节点、RViz 演示及保留的 H5/FIRI 轨迹对照程序编译通过，保留的轨迹查询适配器测试也通过。本次没有运行轨迹优化。
- C3、H5 各自通过 Perlin 和迷宫无窗口交互测试：返回 PASS 和非空曲面。测试进程已关闭，未启动 RViz 窗口。

原始日志、系数哈希、逐案例时序、源码快照及 ROS 测试结果见 [验证归档](validation/sweep_corridor/results.json)。

## 迁移到其他仓库

复制整个 `SweepCorridor/`，或仅复制其 `include/` 并保留 LICENSE。推荐：

```cmake
add_subdirectory(third_party/SweepCorridor)
target_link_libraries(your_target PRIVATE SweepCorridor::SweepCorridor)
```

也可以安装后 `find_package(SweepCorridor REQUIRED)`。作为子目录使用时不会自动创建测试目标；独立构建时可运行测试。旧的 `gcopter::direct_corridor` 仅是当前仓库的适配接口，新仓库不需要复制它。
