# 相同初值的走廊轨迹优化对照

该目录提供独立的实验后端。比较 FIRI 与当前默认的五次 Hermite 椭圆扫掠走廊，不包含可选 Minkowski 截面。

**2026-11-03 更新：** 后端已迁入 diff-planner-swarm `7e41f0d` 四头。运行器现在默认 `--constraint bezier --bezier-depth 2`：FIRI 细分片固定关联一个凸多面体；Hermite 使用控制点与积分点各半权重的位置惩罚；两者速度均使用 Bézier 导数凸包。其余平坦性动力学仍积分。生产 `SplineSFCOptimizer` 同时接入了位置/速度 Bézier 约束，但本实验仍不调用其独立初始化管线。

下文的积分点位置约束说明及旧报告对应 `--constraint sampled`。若要重跑旧目标请显式加该参数；新版 L-BFGS 可能改变迭代结果。`--constraint bezier-controls` 仅用于非凸走廊的纯控制点消融，**不能证明曲线位于扫掠走廊内**。新原理、梯度及实测记录见 `docs/样条库迁移与Bezier约束.md`。

截面查询默认 `--tube-mode fast`：保留原 Bernstein 浮点运算顺序，只计算一阶导数并使用有界精确缓存。`reference` 运行旧路径；`horner` 为可能改变非光滑优化终点的试验模式。`--profile` 单独启用细粒度计时，不用于正式速度排名；各计时字段的包含关系及六例等价验证见 `docs/截面求值与反求缓存加速.md`。

`--shape-mode` 默认 `baseline`，另提供 `hermite-free`（释放形状节点导数的五次 C²）和 `cubic-c2`（三次 C² 形状场）。两者保留中心线和节点，新增连续空间约束下的 refit；生成时间包含基线与 refit。六例体积、耗时、轨迹质量及梯度失败记录见 [截面样条对照](../../../docs/截面非零导数与C2三次样条对照.md)。

两边共用 vendored `SplineTrajectory::QuinticSplineND<3>` 最小 jerk 样条、IdentitySpatialMap、QuadInvTimeMap、L-BFGS 和 GCOPTER 平坦性动力学惩罚。只替换位置走廊约束。原生 `SFCOptimizer::setup` 会先从多面体构造新的 shortestPath 和分段，因此直接调用它不能保证相同初始轨迹；本实验不宣称测量了原生 GCOPTER 完整管线。

## 固定条件

- 冻结地图、种子路径、世界边界和体素分辨率；记录输入文件 SHA256。
- 体素采用半径 `sqrt(3)*resolution/2` 的闭球，加机器人 0.20 m、安全余量 0.05 m；走廊另加 0.002 m 证书余量。
- 两边沿用既有 solid/matched-box 对照：局部扩张范围 3 m、推进距离 7 m。FIRI 原生四次迭代，平面侵蚀后 shortcut；默认直接走廊间距 0.4 m，半轴上限由同一局部包围盒推导，不绑定 1.5 m。
- 初始位置取原始种子折线等弧长采样，轨迹分段间距 1 m、初始分配速度 2 m/s，起终点速度和加速度为零。两边共用这份前端路径产生的轨迹初值，不各自重新寻路。优化向量哈希必须一致；导出 `initial_waypoints.csv`。
- 能量权重 1、总时长权重 20。动力学限制及惩罚与仓库 YAML 一致：速度 4 m/s、机体角速度 2.1 rad/s、倾角 1.05 rad、推力 [2,12] N。
- L-BFGS memory=64、past=3、delta=1e-6、g_epsilon=0；迭代上限由参数指定，同一档中完全相同。

## 位置约束与梯度

FIRI 使用归一化 H 平面并集，违约量为

```
d_F(p) = min_i max_j (n_ij · p + b_ij)
J_F = w * max(0, d_F)^2
```

这不是原生 GCOPTER 的 PolytopeSpatialMap：并集的活动多面体、活动平面切换处可能不可微。它保留了空间变量与初值相同这一控制条件，但其收敛不能代替原生 FIRI 后端的收敛表现。

可微走廊是 `p=c(u)+M(u)z, ||z||<=1`。仅约束 `||M⁺(p-c)||<=1` 会得到无限柱体，因此必须处理截面平面。这里求解

```
f(u,p) = n(u) · (p-c(u)) = 0
f_u = n'(u) · (p-c(u)) - n(u) · c'(u)
du/dp = -n(u)/f_u
```

以相应样条段的归一化路径进度为选根提示，用有步长限制的 Newton，必要时在 knot 区间中寻找变号并二分。该提示不固定真实路径进度。内部根求解后使用隐函数梯度；越过端点时加入端面距离惩罚。

径向违约量是截面内的径向距离 `r*(1-1/sqrt(q))`，其中 `q=||M⁺(p-c)||²>1`。它不是到椭圆的最短欧氏距离；FIRI 的最大平面违约量也不是到多面体的精确欧氏距离。两边用相同的米制平方惩罚权重，但惩罚的数值性质不完全相同，所以质量比较以实际时长、长度、jerk 能量和可行性为主，不直接将两个含罚项目标当成同一目标。

**几何 C2 不等于全空间约束函数 C2。** 走廊未保证全局单射，截面反求可能有多根、分支跳变或 `f_u≈0`。此适配器是局部反求的实验实现，不是任意复杂轨迹上的全局光滑距离场。优化过程中出现的保守选根和不可微切换可能影响耗时和终止；稠密审计中的反求失败计入不通过。

## 判定与计时

优化之后逐段做至少 4 倍优化采样数、且时间间隔不超过 5 ms 的独立采样检查，同时重新查询原始闭球障碍 BVH 和世界边界。记录优化节点与稠密节点上的最大走廊违约量，不能把积分点通过当作全轨迹通过。

`sampled_feasible` 要求：障碍穿透不超过 1e-6 m；截面/端面及径向/平面违约不超过 5 mm；动力学限制容差 1%；无截面反求失败。它仅表示此容差下的采样检查通过，**不提供轨迹连续时间碰撞证书，也不表示优化收敛**。单独保留 L-BFGS 状态：1 是相对目标停止、-1008 是达到迭代上限、-1009 是线搜索区间过小。

`optimize_ms` 只包含 L-BFGS，包含轨迹更新、积分、梯度和截面反求；不包含建图、走廊生成、初值准备、梯度有限差分检查、导出或审计。`evaluations` 多计一次最终状态刷新。走廊生成时间每个输入记录一次，不作为三次中位数；轨迹优化先各预热一次，再交替顺序重复三次。每次都从同一初值重新开始。

## 运行

在仓库根目录运行；OMPL 路径按本机安装调整。

```sh
prlimit --as=2147483648 --cpu=120 cmake -S gcopter/experiments/trajectory \
  -B .scratch/trajectory_comparison/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/.scratch/ompl/usr"
prlimit --as=2147483648 --cpu=120 cmake --build .scratch/trajectory_comparison/build \
  --target trajectory_comparison -j1
python3 gcopter/experiments/trajectory/run.py \
  --inputs gcopter/experiments/trajectory/inputs.json \
  --output .scratch/trajectory_comparison/final_fast \
  --repeats 3 --check-gradients --constraint sampled
python3 gcopter/experiments/trajectory/run.py \
  --inputs gcopter/experiments/trajectory/inputs.json \
  --output .scratch/trajectory_comparison/final_dense \
  --repeats 3 --steps 128 --max-iterations 3200 --weight 1000000 \
  --check-gradients --cpu-seconds 120 --wall-seconds 130 --constraint sampled
```

运行器严格串行，每个子进程虚拟地址空间上限 512 MiB，CPU/wall 时间有上限，超时终止整个子进程组；`/usr/bin/time` 记录峰值 RSS。输入 JSON 引用本地已有的 RViz 冻结请求快照，地图未重复加入 Git；在其他机器重跑需要对应快照，并通过 manifest 中的 SHA256 核对。

每个 case 输出 `runs.jsonl`（含预热）、stderr、初始航点、两条稠密轨迹 CSV。详细 CSV 保留在 `.scratch`；版本化报告只收录紧凑指标和 manifest。

空旷直线控制用例（走廊约束不活跃时，两边应恢复相同的可行解）：

```sh
python3 gcopter/experiments/trajectory/smoke.py
python3 gcopter/experiments/trajectory/summarize.py .scratch/trajectory_comparison \
  docs/validation/trajectory_comparison docs/相同初值轨迹优化对照.md
```

生成报告配图（需要 matplotlib）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 prlimit --as=536870912 --cpu=30 \
  python3 gcopter/experiments/trajectory/plot.py \
  docs/validation/trajectory_comparison/final_dense.json \
  docs/figures/trajectory_comparison.png
```
