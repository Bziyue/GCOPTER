# C3 自由边界实验数据

- `comparison.csv`：68 条输入的成功状态、生成时间、配对并集体积、中段体积和参数弯曲能量。
- `aggregate.json`：开发集与扩展回归的配对统计。失败案例保留在总数中，指标仅在共同成功路径间计算。
- `geometric_metrics.json`：原起终点之间侧壁的曲面弯曲能量、面积，以及纵向边界曲线的曲率能量。
- `results.json.gz`：完整重复时间、阶段诊断、探索配置、FIRI 辅助参考、数学测试与 ROS 检查记录。
- `geometry.tar.gz`：开发集、螺旋及迷宫转弯代表例的曲面、支撑平面、原种子路径，另含最新截图路径的 FIRI 平面。
- `inputs.json`：冻结路径与地图位置、边界及 SHA256。大点云复用原内容寻址快照，不重复存储。
- `artifacts.json`：源码、可执行文件与归档产物哈希。

`final_development` 与 `final_regression` 是正式结果：六变量自由边界、`3e-5` 弯曲权重、`0.01` 纵向间距正则。`balanced_*` 尚未加入持久的纵向间距正则，`development/holdout` 使用原 `1e-5` 弯曲权重。其他组属于早期诊断，不能混入正式耗时统计。探索配置只保留数值证据和构建脚本，正式库只有一套配置。

体积使用独立三维并集反求、8 次随机数字平移、每次 65,536 个 Sobol 样本。`paired_ci95` 是逐例配对差的 95% 区间半宽，不是跨地图泛化区间。曲面弯曲指标是数值积分；尖锐位置会对分辨率敏感，见曲率复核记录。地图中同源多条路径用于回归覆盖，未视为独立地图样本。

读取完整数据：

```python
import gzip, json
with gzip.open('docs/validation/free_boundary/results.json.gz', 'rt') as file:
    data = json.load(file)
```

复现入口在 `SweepCorridor/experiments/free_boundary/README.md`。上一轮手工延伸和追加封端源码已归档到 `../inflation_round1/experiment_sources.tar.gz`，当前源代码不再包含这些实现。
