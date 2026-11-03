# 截面导数与 C² 三次样条实验

比较相同地图、种子路径和轨迹初值下的三个版本：

- `baseline`：现有五次 Hermite 截面，节点处形状一、二阶导数为零。
- `hermite-free`：五次 C² 形状场，节点导数参与变化。
- `cubic-c2`：三次 C² 形状场。

两个新版本保留基线中心线和节点，对形状矩阵做带连续空间约束的二次优化。因此总生成时间包括基线生成与 refit。它们使用局部基函数和每个控制矩阵的对称正定二维变换，并非完全自由的六分量矩阵优化。默认生产生成器不变。

数学构造、完整对比及失败算例见 [实验报告](../../../docs/截面非零导数与C2三次样条对照.md)。紧凑原始数据保存在 `docs/validation/shape_spline/results.json`；大体积输入和逐点输出放在 `.scratch`。

## 构建与测试

以下命令在仓库根目录依次执行。构建限制 3 GiB，运行限制 512 MiB，均为单进程串行。

```sh
cmake -S gcopter/experiments/trajectory -B .scratch/trajectory_comparison/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/.scratch/ompl/usr"
prlimit --as=3221225472 --cpu=600 cmake --build .scratch/trajectory_comparison/build -j1
prlimit --as=536870912 --cpu=60 ctest --test-dir .scratch/trajectory_comparison/build \
  --output-on-failure -j1
```

`test.cpp` 检查中心与节点不变、五次基对原曲线的精确再现、非零节点导数、C² 连接、三次升阶的四阶差分、解析梯度与 Hessian 向量积，并独立采样核对边界及球障碍。

## 同输入对比

```sh
python3 direct_corridor/experiments/shape_spline/run.py \
  --output .scratch/shape_spline/generation --repeats 3 --samples 65536
for mode in baseline hermite-free cubic-c2; do
  python3 gcopter/experiments/trajectory/run.py \
    --inputs gcopter/experiments/trajectory/inputs.json \
    --output ".scratch/shape_spline/trajectory/$mode" \
    --shape-mode "$mode" --repeats 3 --check-gradients
done
```

六份输入来自 `gcopter/experiments/trajectory/inputs.json` 引用的冻结 RViz 请求；输入哈希保存在结果中。复现需要这些本地快照。每个子进程最多 512 MiB 地址空间、55 s CPU、60 s 墙钟时间。体积使用同一网格上的八组平移 QMC 样本，每组 65536 点。三种形状复用相同的体积采样点。

本轮三次版本的 `case_01` 在初始轨迹目标梯度检查处失败。原失败记录保留；额外无梯度检查的诊断结果放在 `.scratch/shape_spline/diagnostic_cubic_case01/case_00`。该目录中的 case 0 对应原输入列表的 case 1，不能混为正常通过结果。复现诊断时，将原列表第二项单独写入一个输入 JSON，并运行：

```sh
python3 gcopter/experiments/trajectory/run.py \
  --inputs .scratch/shape_spline/case01_inputs.json \
  --output .scratch/shape_spline/diagnostic_cubic_case01 \
  --shape-mode cubic-c2 --repeats 3
python3 direct_corridor/experiments/shape_spline/run_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 prlimit --as=1073741824 --cpu=60 \
  python3 direct_corridor/experiments/shape_spline/summarize.py
```

`run_audit.py` 对保存的轨迹中局部反求残差大于 5 mm 的点，枚举其他截面根，区分“选根不合适”和“确实在整条走廊之外”。它不替代原始轨迹指标，也不做连续时间轨迹证明。`summarize.py` 汇总本轮固定目录中的结果，检查三种方法地图、路径、初值一致，生成报告配图和表格；还需要前轮基线逐字节对照记录 `.scratch/shape_spline/baseline_equivalence.json` 与最后重编译前后的几何一致性记录 `.scratch/shape_spline/geometry_replay_equivalence.json`。

## 集成五次生成阶段

`generateIntegratedShape` 在原生成器内部消费最后一次接受的连续平面覆盖，省去重新构造地图约束；原 `generate` 默认行为不变。它还直接使用三重复形状矩阵构造五次零导数初值。完整说明见[集成实验报告](../../../docs/五次非零导数融入生成流程.md)。

```sh
python3 direct_corridor/experiments/shape_spline/run.py --integrated \
  --output .scratch/integrated_shape/generation --repeats 3 --samples 65536
for mode in baseline hermite-free integrated-quintic; do
  python3 gcopter/experiments/trajectory/run.py \
    --inputs gcopter/experiments/trajectory/inputs.json \
    --output ".scratch/integrated_shape/trajectory/$mode" \
    --shape-mode "$mode" --repeats 3 --check-gradients
done
python3 direct_corridor/experiments/shape_spline/run_audit.py \
  --root .scratch/integrated_shape --integrated
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 prlimit --as=1073741824 --cpu=60 \
  python3 direct_corridor/experiments/shape_spline/summarize_integrated.py
```

所有命令依次运行，运行器继续限制每个实验进程 512 MiB、55 s CPU 和 60 s 墙钟时间。汇总读取前轮 `.scratch/shape_spline` 输出并生成逐字节核对记录 `previous_equivalence.json`，还收录未采用的 `warm_pilot` 小试；这两部分是本轮证据整理的输入，不是生成器的运行依赖。失败的梯度检查会保留原状态；若额外做诊断，独立结果放到 `.scratch/integrated_shape/diagnostic/<mode>/case_XX/case_00`，不得覆盖原记录。

## 快速局部更新与原基线

后续加速实验见[五次形状加速与基线对照](../../../docs/五次形状加速与基线对照.md)。`boundary-12` 保留两轮原生优化及原障碍参数序列，使用按需计算的可行步长，再做两遍局部标量扩张；`continuation` 额外在第二轮跳过最大障碍参数。默认生成器未切换到这些实验方案。

```sh
python3 direct_corridor/experiments/shape_spline/run.py --continuation \
  --output .scratch/fast_shape/generation --repeats 7 --samples 65536
for mode in baseline boundary-12 continuation; do
  python3 gcopter/experiments/trajectory/run.py \
    --inputs gcopter/experiments/trajectory/inputs.json \
    --output ".scratch/fast_shape/trajectory/$mode" \
    --shape-mode "$mode" --repeats 3 --check-gradients
done
python3 direct_corridor/experiments/shape_spline/run_audit.py --root .scratch/fast_shape --fast
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 prlimit --as=1073741824 --cpu=60 \
  python3 direct_corridor/experiments/shape_spline/summarize_fast.py
```

筛选入口 `--fast`、`--coordinate`、`--boundary`、`--coarse` 分别对应标量 Newton、缩短第一轮、解析边界步长、较稀形状节点的对照。这些开关互斥，默认仍运行最初的五次/三次实验。完整筛选记录在报告 JSON 中。汇总脚本还读取本轮小试目录、按需步长优化前后的几何等价记录，以及前轮基线输出；复现生成器本身不依赖这些整理材料。
