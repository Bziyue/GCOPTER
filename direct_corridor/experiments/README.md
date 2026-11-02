# 参数化与计算优化实验

这些工具不安装共享库、不修改 ROS 接口，也不替换默认生成器。实验范围见[方案](../../docs/走廊参数化与优化实验方案.md)，首轮结果见[实验报告](../../docs/走廊参数化首轮实验.md)。

## 1. 完整生成器消融

在 GCOPTER 根目录执行：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 direct_corridor/experiments/run_pipeline.py \
  baseline schur decrement workset adaptive_nodes certificate_split certificate_adaptive \
  --inputs .scratch/accel_round3/inputs.json \
  --output .scratch/reproduce_parameterization/pipeline
```

`--inputs` 是保存请求目录的 JSON 数组。脚本从已提交基线 `abc0631` 读取源码，逐项生成隔离副本、编译实验库及匹配其私有头的测试程序，再调用已有 FIRI 对照工具。不同变体只影响子进程的 `LD_LIBRARY_PATH`。默认前三条路径、一次预热、五次重复、8×65,536 个体积样本。`--cases 6 --repeats 9` 可运行本轮块求解复核的设置。

| 变体 | 与基线的差异 |
|---|---|
| `schur` | 缓存消元后的传递矩阵与右端项，回代使用矩阵乘法 |
| `decrement` | 前两阶段停止阈值匹配障碍参数，末阶段精度不变 |
| `workset` | 选择近约束，解后补入遗漏的违反项；从全局内点重启，四轮后回到完整集合 |
| `adaptive_nodes` | 保留所有种子点，远离转角的均匀采样点减半；尚未按净空自适应 |
| `certificate_split` | 第二轮生成每个支持约束时，将其 Bézier 区间一分为二 |
| `certificate_adaptive` | 第二轮只在当前支持间隙 <5 cm 且细分可收紧上界 >5 mm 时细分 |

各变体都保留完整支持残差及连续非退化证书。工作集改变有限 μ 的障碍目标，节点变体改变离散化，不能描述为基线的严格等价提速。`sources.json` 记录源码和返回码，失败不会被标记为通过；查看 `tests.log`、`benchmark.log` 及独立审计字段。

所有构建/测试/对照串行：编译地址空间 2 GiB、CPU 120 s、墙钟 150 s；测试与对照进程 512 MiB。测试 CPU 30 s、墙钟 45 s；对照复用原 runner 的 CPU 25/30 s、墙钟 40 s 限制。输出目录必须是新的目录，以免混入旧结果。

## 2. 三次/五次最小能量样条

只读引用用户指定的参考头，不把参考仓库作为本项目的新运行依赖：

```bash
cmake -S direct_corridor/experiments -B .scratch/reproduce_parameterization/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPLINE_REFERENCE_INCLUDE=/home/zdp/WorkSpace/ubuntu2004/SwarmDevelop/diff-planner-swarm/include/DiffSwarm/planning/backend/spline
prlimit --as=2147483648 --cpu=120 \
  cmake --build .scratch/reproduce_parameterization/build -j1
timeout -k 2 40 prlimit --as=536870912 --cpu=30 \
  .scratch/reproduce_parameterization/build/spline_probe \
  .scratch/reproduce_parameterization/spline
```

`spline_probe` 比较 `CubicSplineND<3>` 和 `QuinticSplineND<3>`：一个偏移场、两个半宽场，固定截面标架；16/64/256/1024 段及一个窄口反例。包括 C² 接缝、插值残差和独立高斯积分检查。计时是固定拓扑的 `update()` 及 Bézier 转换，包含参考实现自身校验，不能当成整个走廊的生成速度。

三次固定端点速度，五次还固定端点加速度，因此二者能量可行域不同。负半宽结果是检测到的不可用参数化，不会导出为可执行走廊。

使用参考头的实际 SHA-256：

- `SplineTrajectory.hpp`：`6bfb00517539a826a586c60969ea8149973e7db3bfddbe630646ff6484f3cd72`
- `SplineConvexHull.hpp`：`221a30e2247f6c71218d7e02e7228e802a60c820574f11f24b442da9bbb09c33`

参考仓库读取时为 `7e41f0d1d1771aadc3e36e447d321e3bea4e59ff`；上游导入信息见其中 `third_party/SplineTrajectory/VERSION.md`。原 MIT 许可证、MINVO BSD-3-Clause 声明保留在参考仓库；本目录未复制这些头文件或 MINVO 数值表。`basis.json` 只作为本地运行时导出数据。

## 3. 固定截面、证书与体素实验

需要 NumPy，不需要 SciPy 或商用求解器：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  timeout -k 2 60 prlimit --as=536870912 --cpu=55 \
  python3 direct_corridor/experiments/geometry_probe.py \
  --spline .scratch/reproduce_parameterization/spline \
  --output .scratch/reproduce_parameterization/geometry \
  --cloud .scratch/interactive/rviz/run_1793605002598226995_3712673/map_1.xyz
```

省略 `--cloud` 仍可重现所有人工几何例及体素并集测试。真实地图和保存路径只在本机 `.scratch/interactive`，未随仓库分发。

脚本依次执行：

1. 最小能量样条的独立共享导数扰动检查、连续半宽极值检查。
2. C² 旋转标架与直接矩阵混合的面积比较。
3. 相同半空间下椭圆、四次超椭圆、二/四模板 Minkowski 和优化；每种枚举相同六个方向。包含梯度/Hessian 有限差分与混合面积积分收敛检查。
4. 相同随机多项式的 Bézier/MINVO 支持上界比较及基函数最小值检查。
5. 固定单段、两半径凸问题的控制约束、1/2 层细分与连续约束交换比较。此例的二阶锥范数退化为正标量半径，不代表完整三维 SOC 求解。
6. 精确体素盒支持函数、连续占据块合并，验证实体立方体、L 形和穿孔体的并集不变；可选读取满足格点契约的真实地图。

浮点多项式根和带容差的 Bernstein 细分是研究用数值 oracle，不是带向外舍入的区间证明。固定截面面积、两半径乘积和局部构造时间分别报告，不能替代经过连续认证的三维并集体积/生成时间。所有输出先经过本实验对应的数值检查，未通过会非零退出并保留已完成部分。

报告插图可用 `plot_results.py --spline <上面的样条输出目录> --summary docs/validation/parameterization_round4.json --output <图片路径.png>` 重建，额外需要 matplotlib。它只读取实验结果绘图，不运行新的优化。
