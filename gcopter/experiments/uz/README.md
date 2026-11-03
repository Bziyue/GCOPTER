# 直接 u,z 轨迹实验

独立实验后端，未替换原世界坐标最小 jerk 后端。全局七次单调进度加五次圆盘内横向曲线，共 11 个变量。轨迹按构造位于已有走廊内；动力学仍为软惩罚。

构建目标 uz_test、uz_experiment 位于 ../trajectory/CMakeLists.txt。用本目录 run.py --inputs ... --output ... --repeats 3 --check-gradients 串行运行；每个几何片默认 16 阶 Gauss–Legendre，用 --quadrature 调整，默认进行九个时长倍数的初始化选择。

六个冻结地图用例的初始/最终梯度验证通过，动力学可行率为 0/6，不能发布为可用规划结果。完整原理及失败记录见 docs/直接uz轨迹参数化实验.md。
