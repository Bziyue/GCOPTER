# 第一轮空间膨胀实验数据

- `comparison.csv`：逐例耗时、并集体积、原路径区间体积和弯曲能量。
- `results.json.gz`：完整计时序列、失败状态、连续证书、输入摘要、构建记录和验证日志。
- `geometry.tar.gz`：8 条开发路径 × 3 个最终配置的完整曲面、原路径区间曲面、支撑平面，以及原始路径和 FIRI 平面。
- `inputs.json`：68 条冻结输入，含地图和路径的 SHA256；地图复用本地内容寻址快照，不重复提交大点云。
- `artifacts.json`：源码、可执行程序和数据产物的 SHA256。

结果分组：`final` 是同原始扩张盒下的完整消融；`selection` 是限制端部重试后的开发集；`holdout_bounded` 是对应的 60 条扩展回归。`holdout` 和 `retry_trace` 保留修正前的长尾与诊断，不能与最终配置混在一起计算耗时。`profile` 打开了阶段时钟，只用于定位成本；`final_equivalence` 验证趋零端部修正后的几何一致性。`edge` 是对应的世界边界案例。

读取原始数据：

```python
import gzip, json
with gzip.open("docs/validation/inflation_round1/results.json.gz", "rt") as file:
    results = json.load(file)
```

每次 QMC 结果中的 `samples` 是八个数字平移的总样本数。案例 8、17 使用四倍样本复核。原始输出路径指向本机实验目录；已归档的曲面可从 `geometry.tar.gz` 提取。

`experiment_sources.tar.gz` 保存本轮的历史实验实现；端部延伸、追加封端及其特化预算策略已从当前源码移除。
