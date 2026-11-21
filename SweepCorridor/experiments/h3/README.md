# H3 局部 Hermite 实验

从 `ros2@7d4fc603607f131a1d282247c3e523c11da5d06e` 建立的独立实验。主库及 RViz2 默认使用 C3，RViz2 可通过 `corridor:=h3` 切换到 H3。实验程序输出相同的四控制点三次 Bézier 几何格式，但几何连续性约定为 **C¹**。

RViz2 默认把占据栅格作为实体体素交给 H3，搜索、种子检查和分离面共用同一种几何。在同一 BVH 查询域内，粗网格膨胀允许相邻段共享经过整段验证的分离面。细网格通过精确插入节点得到。当前默认与上一代实体体素 H3 的逐例对照见[核心体积与生成耗时](../../../docs/validation/h3_core/README.md)。早期球障碍对照见[分层膨胀实验](../../../docs/H3实体体素与分层膨胀.md)。

每个 Hermite 节点有三个中心偏移和三个对称形状变量，相邻段共享九通道一阶导数。全局膨胀采用六维块三对角求解，细网格再耦合优化节点值与导数。每个十五维局部步只影响相邻两段。第二次联合扫描访问全部节点。随后按障碍压力选段换面，按核心截面的可行面积余量补选节点，执行一轮局部修正。自由端中心块可在没有障碍压力时单独更新。

- `model.hpp`、`block_tridiagonal.hpp` 和 `problem.hpp` 实现两节点 Hermite 模型、积分、块矩阵与支撑约束组装。
- `cap_volume.hpp` 计算自由端体积和移动锚点的边界导数。`anchors.hpp` 和 `solve.hpp` 处理锚点反解、端面主动集及全局线搜索。
- `joint_jets.hpp` 与 `end_blocks.hpp` 分别处理节点与端部更新。线搜索保留完整扫掠体积的多项式，直接计算试探步的自由端扣减量。
- `jets.hpp` 提出自然样条导数并按整段证书裁剪。`wall_quality.hpp` 对中间半径回退后的最终平滑候选核对物理侧壁曲率和核心体积。`volume_jets.hpp` 用于仅释放导数的历史消融。
- `inflation_cover.hpp`、`local_refinement.hpp` 和 `generator.hpp` 负责粗细转移、分离面共享、局部换面、初始化重试与在线生成。点和线段到原始障碍的距离由主库的 `ObstacleIndex` 提供。

## 障碍输入与内缩半径

生成器接受同一个 `SweepCorridor::ObstacleIndex`，调用方按原始地图的几何含义构造索引。

| 原始输入 | C++ 入口 | 几何含义 |
|---|---|---|
| 障碍点坐标 | `ObstacleIndex::fromPoints(points)` | 每个采样点自身半径为零，点间空间保持空闲 |
| 占据体素中心和边长 | `ObstacleIndex::fromOccupiedVoxels(centers, resolution)` | 占据整个轴对齐立方体 |
| 带半径的障碍球 | `ObstacleIndex(std::vector<Sphere>{...})` | 每个障碍独立指定球心和物理半径 |

以上长度单位均为米。三种输入都直接传原始几何。H3 通过 `options.robot_radius` 为机器人球体留空间，`safety_margin` 表示附加净空，分离面认证还预留 `certificate_margin`。把占据体素的中心当作点输入，会改变障碍的物理范围。

```cpp
namespace sc = SweepCorridor;
auto obstacles = sc::ObstacleIndex::fromPoints(points);  // std::vector<sc::Vec3>
sc::Options options;
options.robot_radius = 0.20;
auto result = sc::experiments::h3::generate(seed, obstacles, bounds, options, settings);
```

## 构建和运行

在仓库根目录执行。两个脚本都固定到 CPU 2、限制地址空间为 4 GiB，并且串行运行。

```bash
python3 SweepCorridor/experiments/h3/build.py
taskset -c 2 prlimit --as=4294967296 .scratch/h3/tests
python3 SweepCorridor/experiments/h3/run.py \
  --output .scratch/h3/my_comparison \
  --cases 0,1,2,3,4,5,6,67 \
  --modes c3,h3_jets,h3_mid --repeats 7 --samples 65536
```

每种方法预热一次；`--repeats` 是正式重复次数，`--samples` 是每组随机平移的样本数，共八组。`--samples 0` 只运行生成和独立证书检查。地图、种子及 SHA256 来自 `docs/validation/free_boundary/inputs.json`，该清单引用既有冻结输入快照。

| 实验名称 | 导数更新 / 求解预算 | 用途 |
|---|---|---|
| `c3` | 冻结提交中的原生 C² C3 | 主基线 |
| `h3` | 固定初始共享导数，28 步 | 诊断导数冻结的形状损失 |
| `h3_jets` | 每轮膨胀后更新，28 步 | 旧 H3 的算法配置 |
| `h3_previous` | 从 `d1e0ad6` 重建的旧 H3 | 冻结代码对照（先构建 `--targets previous`） |
| `h3_joint_baseline` | 从 `fd7a301` 重建的共享导数 H3 | 端部耦合实验的基线 |
| `h3_adaptive_baseline` | 从 `08451b5` 重建的局部刷新与端部耦合 H3 | 分层膨胀实验的直接基线，先构建 `--targets adaptive` |
| `h3_new` | 移动锚点核心体积、截面逐点 Jacobian、局部换面与面积修正 | 当前 H3 默认 |
| `h3_pre_core` | 上一代实体体素、粗细网格 H3 | 本轮几何对照 |
| `h3_core_full` / `h3_core_pressure` | 全体积权重 / 仅按压力选择局部节点 | 核心目标与面积选点消融 |
| `h3_core_span_scale` / `h3_core_no_share` | 切换到单段尺度 / 停用经过认证的平面共享 | 独立验证分离面尺度与复用 |
| `h3_points` | 与 `h3_new` 相同的求解器，输入零半径障碍点 | 独立检查点障碍与机器人的净空 |
| `h3_spheres` | 与 `h3_new` 相同的求解器，输入旧版外接球 | 单独测量求解策略的收益 |
| `h3_voxel_single` / `h3_voxel_fine` | 上一代 H3 逐段建面或从细网格开始 | 分层膨胀实验的历史配置 |
| `h3_world` | 更新导数，采用九维世界 Hessian 回传 | 六维组装的独立消融（先构建 `--targets world`） |
| `h3_mid` | 收紧障碍参数前及每轮后更新，28 步 | 平滑性与耗时取舍 |
| `h3_more` | 两轮，每轮 12+3+3 步 | 迭代预算消融 |
| `h3_three` | 三轮，每轮 4+2+2 步 | 覆盖刷新预算消融 |
| `h3_polish` | 默认候选后增加 3 步再膨胀 | 平滑后余量消融 |

`generate()` 默认使用 `AfterPass` 导数更新，最后进行共享导数联合修正及局部刷新。主膨胀正则权重保持原值，局部形状修正使用三倍弯曲正则。实验设置集中在 `OptimizerSettings`，不进入主库配置。阶段计时由 `SWEEP_PROFILE=1` 单独开启，正式速度比较关闭阶段时钟。结果保留每次终止原因；`SUCCESS` 表示得到已认证几何，不能解读成优化器达到收敛阈值。

数值结果、环境记录与图见 [H3 实验报告](../../../docs/H3三次Hermite走廊实验.md)。本实验只比较走廊生成，不运行轨迹优化。

## RViz2 点选

构建 ROS 演示后，从任意目录启动：

```bash
/home/zdp/ForCodex/GCOPTER/gcopter/tools/run_direct_demo.sh corridor:=h3 scene:=perlin
```

迷宫使用 `scene:=maze`；`map_seed:=2048` 可更换地图。默认固定高度为 2.5 m，通过 `height:=2.0` 调整。顶部 **2D Goal Pose** 依次选起点、终点，再次点选可开始新路径。

H3 入口对应实验的 `h3_new`：主膨胀后联合优化节点值和共享导数。原地图生成器、Informed RRT* 搜索和同输入 FIRI 对照均保留；界面标注 **H3 · C¹**，显示 H3 和 FIRI 各自生成耗时。红色为 H3，蓝色为 FIRI，可在左侧分别切换。H3 模式只生成走廊，隐藏轨迹优化控件。默认 `corridor:=c3` 保持原轨迹演示功能。

渲染和导出直接读取 H3 的三次 Bézier 系数，不重新拟合，也不转换成 C²。每次输出 `metrics.json` 记录 `shape_mode=h3`、`continuity=1`、`representation=hermite_cubic_v3`、`jet_updates`、`joint_steps`、`end_steps`、`refreshed_spans`、`budget_skips`、`coarse_start_used`、`shared_obstacle_spans`、`intermediate_radius_used`、`area_selected_spans` 和 `core_smoothed`。耗时分开记录路径搜索、走廊生成、FIRI 和渲染导出。

ROS 集成测试（Perlin、迷宫各一条点选路径）：

```bash
cd /home/zdp/ForCodex/GCOPTER
source /opt/ros/lyrical/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=95 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 2 prlimit --as=4294967296 -- \
  python3 gcopter/tests/c3_demo_smoke.py --corridor h3
```

完整重建、主库验证、68 例回归、组装消融、阶段计时和报告归档可通过以下命令串行复现：

```bash
python3 SweepCorridor/experiments/h3/validate.py
```

上面的脚本复现首轮 H3 实验。在 `fd7a301` 上，共享导数体积实验的复现命令为：

```bash
python3 SweepCorridor/experiments/h3/validate_volume.py
python3 SweepCorridor/experiments/h3/analyze_volume.py
```

输入归档随仓库保存，脚本自动解压并核对 SHA256。结果见 [共享导数体积优化报告](../../../docs/H3共享导数体积优化.md)。命名消融保留独立配置，不随默认值变化；`h3_new_rebuild` 重新构造支持约束，`h3_new_halving` 改用逐次减半线搜索。其他 `h3_*` 模式用于报告中保留的预算、正则和仅导数实验，不接入 ROS 参数。

在 `08451b5` 上，端部耦合与局部覆盖实验也使用这 71 条输入，可运行以下命令复现。

```bash
python3 SweepCorridor/experiments/h3/validate_adaptive.py
python3 SweepCorridor/experiments/h3/analyze_adaptive.py
```

该历史配置的第二轮扫描保留累计首轮收益达 99% 的节点，优先刷新端部和约 5% 的高压力内部段，最多两轮。端部中心与共享中心导数组成小块，不设固定外推距离。消融 `h3_adaptive_no_budget`、`h3_adaptive_no_interior`、`h3_adaptive_no_end`、`h3_adaptive_no_refresh` 和 `h3_adaptive_one_round` 分别检验预算、内部刷新、端部耦合、覆盖刷新和第二轮的贡献。结果与数学说明见 [局部覆盖与端部耦合报告](../../../docs/H3局部覆盖与端部耦合优化.md)。

完成 ROS 集成测试后，可以核对演示与离线生成的全部系数：

```bash
python3 SweepCorridor/experiments/h3/verify_volume_demo.py \
  --root .scratch/h3_inflation --output docs/validation/h3_inflation/rviz.json
```
