# H3 局部 Hermite 实验

从 `ros2@7d4fc603607f131a1d282247c3e523c11da5d06e` 建立的独立实验。主库及 RViz2 默认使用 C3，RViz2 可通过 `corridor:=h3` 切换到 H3。实验程序输出相同的四控制点三次 Bézier 几何格式，但几何连续性约定为 **C¹**。

H3 每个节点有三个中心偏移和三个对称形状变量。世界坐标中的九通道一阶导数由相邻段共享。主膨胀采用六维块三对角求解，之后联合优化节点值与共享导数，每个局部问题有 15 个变量，只改变相邻两段。第二次扫描按上一轮收益分配预算，再对受限区域刷新分离面，交替进行局部形状修正与六维端部中心块更新。初始化精确复用未优化的 C3 小截面参考；不调用 C3 的走廊膨胀器，初始化成本包含在总耗时中。

实现重点：

- `model.hpp`：两节点 Hermite 模型、精确弯曲能量、预计算的五点体积积分与体积线搜索多项式。
- `block_tridiagonal.hpp`：6×6 块三对角、尺度平衡、缓存 Schur 分解，多右端项复用。
- `problem.hpp`：单节点支撑约束直接在六维空间组装，细分约束只耦合两个节点；三组导数控制量只保留一组可变进度约束。
- `anchors.hpp` / `solve.hpp`：十二维局部反求梯度、自由端面主动集、体积秩一修正、可行线搜索。
- `jets.hpp`：共享导数提议，只更新相邻两段，复用原支持平面并重新认证这两段的全截面 Jacobian。
- `joint_jets.hpp`：节点值与共享导数的局部联合体积优化；复用完整支持分组，闭式弯曲能量和精确三次体积线搜索。
- `end_blocks.hpp`：固定形状下的仿射体积、中心与共享切向量的六维耦合更新、隐式锚点和缓存 Jacobian。
- `local_refinement.hpp`：复用 BVH 查询候选，只刷新高压力区域的分离面，再做小预算局部修正。
- `volume_jets.hpp`：Hermite 局部基函数和仅释放导数的诊断方案。
- `generator.hpp`：种子输入、覆盖构造、两轮膨胀和导数更新。

障碍 BVH、分离面、查询缓存、连续支持与 Jacobian 认证直接复用主库。共享冗余界新增一个编译期分支，H3 的固定形状导数偏置必须计入可达域证明；原生 C3 不执行该分支。

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
| `h3_joint_baseline` | 从 `fd7a301` 重建的共享导数 H3 | 本次直接基线（先构建 `--targets joint`） |
| `h3_new` | 主膨胀 + 收益预算扫描 + 最多两轮局部刷新与端部耦合 | 当前 H3 默认 |
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

渲染和导出直接读取 H3 的三次 Bézier 系数，不重新拟合，也不转换成 C²。每次输出 `metrics.json` 记录 `shape_mode=h3`、`continuity=1`、`representation=hermite_cubic_v3`、`jet_updates`、`joint_steps`、`end_steps`、`refreshed_spans` 和 `budget_skips`。耗时分开记录路径搜索、走廊生成、FIRI 和渲染导出。

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

当前分支的端部耦合与局部覆盖实验也使用这 71 条输入。完整复现命令为：

```bash
python3 SweepCorridor/experiments/h3/validate_adaptive.py
python3 SweepCorridor/experiments/h3/analyze_adaptive.py
```

默认第二轮扫描保留累计首轮收益达 99% 的节点；优先刷新端部和约 5% 的高压力内部 span，最多两轮。端部中心与共享中心导数组成小块，不设固定外推距离。消融 `h3_adaptive_no_budget`、`h3_adaptive_no_interior`、`h3_adaptive_no_end`、`h3_adaptive_no_refresh` 和 `h3_adaptive_one_round` 分别检验预算、内部刷新、端部耦合、覆盖刷新和第二轮的贡献。结果与数学说明见 [局部覆盖与端部耦合报告](../../../docs/H3局部覆盖与端部耦合优化.md)。

完成 ROS 集成测试后，可以核对演示与离线生成的全部系数：

```bash
python3 SweepCorridor/experiments/h3/verify_volume_demo.py \
  --root .scratch/h3_adaptive --output docs/validation/h3_adaptive/rviz.json
```
