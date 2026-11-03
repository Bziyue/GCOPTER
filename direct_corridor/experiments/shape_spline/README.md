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
