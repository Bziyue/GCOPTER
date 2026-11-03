# 样条与优化内核来源

来源：`/home/zdp/WorkSpace/ubuntu2004/SwarmDevelop/diff-planner-swarm`

固定提交：`7e41f0d1d1771aadc3e36e447d321e3bea4e59ff`。读取时源工作区干净。四个头原样复制，未翻译或格式化。源库在上游导入版基础上的零梯度快速路径、数值修复及 L-BFGS 修改均保留；不宣称与原始上游一致。

| 本仓库文件 | 来源文件 | SHA256 |
|---|---|---|
| `gcopter/include/SplineTrajectory/SplineTrajectory.hpp` | `include/DiffSwarm/planning/backend/spline/SplineTrajectory.hpp` | `6bfb00517539a826a586c60969ea8149973e7db3bfddbe630646ff6484f3cd72` |
| `gcopter/include/SplineTrajectory/SplineOptimizer.hpp` | `include/DiffSwarm/planning/backend/spline/SplineOptimizer.hpp` | `350ebdc806a5399250d3aafc753f78097c3cf20b0c20d17858c9f2daf20f80dc` |
| `gcopter/include/SplineTrajectory/SplineConvexHull.hpp` | `include/DiffSwarm/planning/backend/spline/SplineConvexHull.hpp` | `221a30e2247f6c71218d7e02e7228e802a60c820574f11f24b442da9bbb09c33` |
| `gcopter/include/gcopter/lbfgs.hpp` | `include/DiffSwarm/planning/backend/optimizer/lbfgs.hpp` | `f42f908cea57fb2feabc676826e50bcc038e3459fca5739cb50091b0910c31be` |

更新时从上述来源同步四头，核对哈希，迁移调用方并运行数值及构建回归。Spline 三头保留 MIT 许可；MINVO 声明见本目录 THIRD_PARTY_NOTICES.md。L-BFGS 归属与 MIT 许可见 third_party/GCOPTER/LICENSE。
