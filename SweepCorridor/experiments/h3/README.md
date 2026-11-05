# H3 局部 Hermite 实验

从 `ros2@7d4fc603607f131a1d282247c3e523c11da5d06e` 建立的独立实验。主库及 RViz2 默认使用 C3，RViz2 可通过 `corridor:=h3` 切换到 H3。实验程序输出相同的四控制点三次 Bézier 几何格式，但几何连续性约定为 **C¹**。

H3 每个节点有三个中心偏移和三个对称形状变量。世界坐标中的九通道一阶导数由相邻段共享，在节点优化期间固定，在外层通过自然三次导数提议和连续安全限制更新。初始化精确复用未优化的 C3 小截面参考；不调用 C3 的走廊膨胀器，初始化成本包含在总耗时中。

实现重点：

- `model.hpp`：两节点 Hermite 模型、精确弯曲能量、预计算的五点体积积分与体积线搜索多项式。
- `block_tridiagonal.hpp`：6×6 块三对角、尺度平衡、缓存 Schur 分解，多右端项复用。
- `problem.hpp`：单节点支撑约束直接在六维空间组装，细分约束只耦合两个节点；三组导数控制量只保留一组可变进度约束。
- `anchors.hpp` / `solve.hpp`：十二维局部反求梯度、自由端面主动集、体积秩一修正、可行线搜索。
- `jets.hpp`：共享导数提议，只更新相邻两段，复用原支持平面并重新认证这两段的全截面 Jacobian。
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
| `h3_jets` | 每轮膨胀后更新，28 步 | H3 默认候选 |
| `h3_world` | 更新导数，采用九维世界 Hessian 回传 | 六维组装的独立消融（先构建 `--targets world`） |
| `h3_mid` | 收紧障碍参数前及每轮后更新，28 步 | 平滑性与耗时取舍 |
| `h3_more` | 两轮，每轮 12+3+3 步 | 迭代预算消融 |
| `h3_three` | 三轮，每轮 4+2+2 步 | 覆盖刷新预算消融 |
| `h3_polish` | 默认候选后增加 3 步再膨胀 | 平滑后余量消融 |

`generate()` 默认使用 `AfterPass` 导数更新。实验设置集中在 `OptimizerSettings`，不进入主库配置。阶段计时由 `SWEEP_PROFILE=1` 单独开启，正式速度比较关闭阶段时钟。结果保留每次终止原因；`SUCCESS` 表示得到已认证几何，不能解读成优化器达到收敛阈值。

数值结果、环境记录与图见 [H3 实验报告](../../../docs/H3三次Hermite走廊实验.md)。本实验只比较走廊生成，不运行轨迹优化。

## RViz2 点选

构建 ROS 演示后，从任意目录启动：

```bash
/home/zdp/ForCodex/GCOPTER/gcopter/tools/run_direct_demo.sh corridor:=h3 scene:=perlin
```

迷宫使用 `scene:=maze`；`map_seed:=2048` 可更换地图。默认固定高度为 2.5 m，通过 `height:=2.0` 调整。顶部 **2D Goal Pose** 依次选起点、终点，再次点选可开始新路径。

H3 入口对应实验的 `h3_jets`：每轮膨胀后更新共享导数。原地图生成器、Informed RRT* 搜索和同输入 FIRI 对照均保留；界面标注 **H3 · C¹**，显示 H3 和 FIRI 各自生成耗时。红色为 H3，蓝色为 FIRI，可在左侧分别切换。H3 模式只生成走廊，隐藏轨迹优化控件。默认 `corridor:=c3` 保持原轨迹演示功能。

渲染和导出直接读取 H3 的三次 Bézier 系数，不重新拟合，也不转换成 C²。每次输出 `metrics.json` 记录 `shape_mode=h3`、`continuity=1`、`representation=hermite_cubic_v3` 和 `jet_updates`。耗时分开记录路径搜索、走廊生成、FIRI 和渲染导出。

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
