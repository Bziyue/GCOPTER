# 历史实验归档

历史原始压缩包和过程日志保存在提交 `62645e2`，本地分支 `archive/pre-c3-cleanup-62645e2` 指向该提交。该提交已存在于 GitHub；归档分支本轮未推送。

当前目录保留报告、统计摘要、冻结输入及普通结果文件。移出的文件逐个与 Git blob 核对，路径、大小和 SHA256 见 `manifest.json`。需要复现时恢复单个文件：

```bash
git show 62645e2:docs/validation/c3_specialization/raw_csv.tar.gz > /tmp/raw_csv.tar.gz
```

历史报告的压缩包链接指向固定提交，不依赖当前分支。归档只缩小工作树，不重写 Git 历史。
