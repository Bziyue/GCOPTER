# 直接五次 Hermite 可微走廊

独立 C++17 / Eigen 模块，位于 GCOPTER `ros2` 的实验分支。输入有序种子折线、不可变球体障碍索引和有限地图边界，输出世界坐标下 C² 连续的椭圆截面扫掠。**不调用 FIRI，不生成顺序凸多面体，不依赖 ROS 或外部 LP 求解器。** ROS2 仅是可选消息适配器。

实现、数学证书、输出 v2 格式和边界见 [设计说明](../docs/五次Hermite走廊.md)，同地图同路径对照见 [Hermite 验证报告](../docs/Hermite走廊验证.md)。主生成器已替换为五次 Hermite → 等价 Bézier → 局部细分，联合优化截面偏移与椭圆主轴方向。

交互演示默认使用原生 C3；`shape_mode:=smooth` 运行 H5 对照。H5 核心库的默认 `Options` 已采用同一 smooth 配置：`frame_smoothing=1.5`、`section_smoothing=0.2`、`jet_smoothing=0.01`、`certify_jacobian=true`、`initial_iterations=8`。正常演示不再提供历史 `baseline`、`boundary-12` 模式。

数学化简、数值稳定性修复及同范围 FIRI 复测见[体积与速度优化](../docs/Hermite体积与速度优化.md)和[第二轮加速](../docs/Hermite第二轮加速.md)，球障碍度量分离与查询复用见[第三轮优化](../docs/Hermite第三轮优化.md)，当前的分组剔除、块求解缓存和六例复核见[最新加速报告](../docs/分组剔除与块求解加速.md)。

三次/五次最小能量样条、支持函数截面、连续约束交换的实验范围见[参数化与优化实验方案](../docs/走廊参数化与优化实验方案.md)，实测及失败案例见[首轮实验报告](../docs/走廊参数化首轮实验.md)。这些新参数化仍处于实验阶段。

Minkowski 已有完整三维实验生成器和 C² 查询接口，启用 `DIRECT_CORRIDOR_BUILD_MINKOWSKI_EXPERIMENT` 构建。四模板、平滑定向、0.6 m 间距在本轮两条代表性迷宫路径中，相比原 0.4 m 椭圆同时扩大体积并减少耗时；同为 0.6 m 时则以额外耗时换取更大体积。不规则地图尚未获益，默认仍为椭圆。完整六例、同间距对照及 26 条回放见 [Minkowski 三维实验](../docs/Minkowski三维走廊实验.md)。

## 独立构建与验证

在此 GCOPTER 检出根目录执行：

```bash
cmake -S direct_corridor -B build/direct_corridor -DCMAKE_BUILD_TYPE=Release
cmake --build build/direct_corridor -j1
ctest --test-dir build/direct_corridor --output-on-failure
build/direct_corridor/direct_corridor_benchmark 30
python3 direct_corridor/tools/example.py \
  --binary build/direct_corridor/direct_corridor_cli \
  --output .scratch/direct_corridor_example --plot
```

核心仅需 Eigen3、CMake 与 C++17；绘图额外需要 Python numpy/matplotlib。基准不包含 BVH 构建时间，该时间单独报告。地图只构建一次、同一地图多次生成时可复用 `ObstacleIndex`。

## C++ 使用

```cpp
#include <gcopter/direct_corridor.hpp>
using namespace gcopter::direct;
std::vector<Vec3> seed{{0,0,2}, {3,0,2}, {3,3,2.5}};
ObstacleIndex map({Sphere{Vec3(1,2,2), 0.2}});
Bounds bounds{Vec3(-5,-5,0), Vec3(10,10,5)};
Options options;
auto result = generate(seed, map, bounds, options);
if (!result) { /* 根据 status / message 处理失败；没有部分输出。 */ }
else {
  auto section = result.corridor.section(0.4);
  auto boundary = section.point(Eigen::Vector2d(1,0));
  auto derivatives = result.corridor.evaluate(0.4, boundary);
}
```

本地工程可 `add_subdirectory(.../direct_corridor)`，链接 `gcopter::direct_corridor`。独立安装后也支持 `find_package(gcopter_direct_corridor CONFIG REQUIRED)`。所有位置、半径为米；导数对归一化 `u∈[0,1]` 求导，不是弧长或时间导数。`evaluate()` 的梯度/Hessian 顺序是 `[u,x,y,z]`；**`q<=0` 还必须搭配截面平面条件**，不能把无限柱体误当成截面。

可选 `Options::local_range > 0` 将整个扫掠约束在沿种子线段扩张的轴对齐局部盒内，`local_progress` 控制长种子边的切分长度（默认 7 m）。范围默认关闭；它与 `max_radius`、`max_offset` 同时生效，不会自动取消这两个上限。若要复现 FIRI 相同 3 m 扩张范围并放开原 1.5 m 限制，参见[同范围体积与耗时对照](../docs/FIRI同范围扩张对照.md)及基准的 `matched-box` 模式。

## 命令行与结果

```bash
build/direct_corridor/direct_corridor_cli \
  seed.xyz obstacles.xyz 0.1732050808 0.2 \
  -25 -25 0 25 25 5 .scratch/output
```

参数依次为种子 xyz、障碍球心 xyz、统一障碍半径、机器人半径、地图 lower xyz、upper xyz、输出目录。每个输入文件每行三个有限数值，可有 `#` 注释；空障碍文件代表调用者明确给出的无障碍空间。

输出目录后可选 `safety_margin control_spacing budget_ms min_radius`。成功输出 `controls.csv`、`surface.csv`、`metrics.json`。`controls.csv` 每行 11 列：`u0,u1,cx,cy,cz,m00,m10,m20,m01,m11,m21`，连续六行组成一个五次 Bézier 区间。它们不是 B 样条控制点。`surface.csv` 只是显示采样，不是验收依据。失败返回非零；推荐每次使用独立输出目录，避免把旧文件当作本次结果。

球体列表必须覆盖需要排除的实体。对于边长 `r` 的占据体素，传入体素中心和半径 `sqrt(3)*r/2` 是保守覆盖；再由模块增加机器人半径与安全余量。不要重复膨胀。如果地图有 Unknown 策略，调用方应先按该策略把阻塞区域加入障碍，模块不推断观测状态。

## ROS2

要使用 **GCOPTER 原地图、原 Informed RRT* 搜索和 RViz2 交互选点**，见
[原始地图交互验证](../docs/原始地图交互验证.md)。构建后在仓库根目录执行
`gcopter/tools/run_direct_demo.sh`；该演示直接调用本模块，无需另起下述适配器。

这是同一仓库内独立的 `gcopter_direct_corridor` ament 包，不要求构建 GCOPTER 原有 OMPL 轨迹示例：

```bash
source /opt/ros/<distro>/setup.bash
colcon build --packages-select gcopter_direct_corridor \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DDIRECT_CORRIDOR_BUILD_ROS2=ON
source install/setup.bash
ros2 run gcopter_direct_corridor direct_corridor_node
```

节点默认坐标系 `map`，不执行隐式 TF：

| 方向 | 话题 | 类型 / 含义 |
|---|---|---|
| 输入 | `/direct_corridor/map` | `sensor_msgs/PointCloud2`，FLOAT32 xyz，占据点按 `voxel_resolution` 量化为体素，再以外接球覆盖 |
| 输入 | `/direct_corridor/seed` | `nav_msgs/Path`，有序种子，至少两点 |
| 输出 | `/direct_corridor/controls` | `std_msgs/Float64MultiArray`，每行十一列，与 CSV 一致 |
| 输出 | `/direct_corridor/reference` | `nav_msgs/Path`，中心线显示采样 |
| 输出 | `/direct_corridor/markers` | `visualization_msgs/MarkerArray`，椭圆环显示 |
| 输出 | `/direct_corridor/status` | 成功、等待或具体失败原因 |

参数包括 `bounds`（默认 `[-25,-25,0,25,25,5]`）、`voxel_resolution`、`robot_radius`、`safety_margin`、`control_spacing`、`min_radius`、`max_radius`、`max_offset`、`inflation_iterations`、`certificate_depth`、`budget_ms`。输入 QoS 为 reliable/volatile，输出为 reliable/transient-local。地图、路径更新先清除旧输出；新地图会重算已存种子。无效新路径不会复用旧种子。

适配器面向已给定的静态占据地图：未列出的体素视为可用，不包含观测/Unknown 管理。它使用单线程同步回调，不提供抢占式取消；核心提供协作式取消回调及预算检查。地图索引构建不计入核心预算。启动节点后，可在同一隔离 ROS domain 执行 `python3 direct_corridor/tools/ros2_smoke.py` 验证端到端输入与清除语义；该脚本会发布测试地图和路径，不应连接到运行中的机器人话题。

## 当前范围

- 支持多转弯、反向、竖直、不同高度与长路径，保留首末端点；中间折点允许平滑。
- 每节点优化平面内偏移和完整对称形状矩阵，椭圆主轴可旋转；`smooth` 模式先对参考方向作弧长滤波，再构造运输标架。
- 截面为可能倾斜的椭圆，不强制精确垂直于中心线；世界曲面本身 C²。
- `min_radius` 是原生节点优化的半轴下界；旋转混合和可选导数调整后的实际短半轴不必等于该下界。`smooth` 模式用整个截面的 Jacobian 证书验收局部不折回。
- 任意连续曲线的有限采样并不代表原曲线已知。这里接受的是**给定折线**，检查其闭线段，构造安全平滑近似；尖点、回折、净空不足或资源上限返回失败。
- 不保证扫掠全局不自交，也不保证整个管内世界坐标映射的一一性。不包含时间参数化、动力学、集群避碰或执行控制，未替换原 GCOPTER 轨迹优化器。

## C3 与 H5 库接口

安装包同时导出 `gcopter::direct_corridor`（H5）和
`gcopter::cubic_corridor`（C3）。C3 使用正式库，实验目录只保留测试与比较工具。

```cpp
#include <gcopter/cubic/cubic_generator.hpp>
using namespace gcopter::direct;

GeometryOptions geometry;
geometry.frame_smoothing = 1.5;
geometry.local_range = 3.0;
native::CubicOptions c3(geometry);
Options h5(geometry);
// native::generateNative(seed, map, bounds, c3);
// generate(seed, map, bounds, h5);
```

`GeometryOptions` 表示地图、机器人、几何尺度和资源上限。H5 的平滑、
Newton 迭代、取消回调和诊断参数属于 `Options`；C3 的固定求解常量与显式
诊断属于 `native::OptimizerSettings`。C3 不再接受 H5 配置对象，跨算法比较时
显式复制共同的几何输入。
