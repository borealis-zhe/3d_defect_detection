# Real3D-AD 多视图异常检测

本仓库保留方法实现、配置、测试和实验结论。正常点云经配准与多视图渲染，使用冻结DINOv2和可训练瓶颈/解码器计算重建误差，再映射到三维参考点。

## 目录

| 路径 | 内容 |
|---|---|
| `repro/` | 渲染、配准、特征重建、评分、选点与视角消融方法 |
| `scripts/` | 数据审计、分析计算及可视化实现；部分几何模块依赖其中的扫描方向估计 |
| `tests/` | 投影、可见性、标签匹配和融合契约测试 |
| `bilateral26_positions.json` | B双侧26方向配置 |
| `global26_positions.json` | C全方位26方向配置 |
| `docs/methods.md` | 方法代码索引与必要依赖 |
| `docs/conclusions.md` | 已完成实验结论、证据位置与局限 |

## 当前结果

12类、48正常训练对象、603测试对象，12类宏平均，单位为%。

| 方案 | 物体AUROC | 点AUROC | Point AP |
|---|---:|---:|---:|
| B 双侧26训练视图 | 93.37 | 97.96 | 63.47 |
| C 全方位26训练视图 | 94.23 | 98.27 | 64.46 |

测试使用相同入口、15视图与前表面参考点域。单种子且测试集已多次用于探索；结果不代表完整表面定位或官方协议复现。详见[结论](docs/conclusions.md)。

## 方法与验证

环境依赖见 `requirements-local.txt`，PyTorch/torchvision需与设备匹配。方法模块不内置SSH、密码或远程任务提交逻辑。公共训练方法可通过 `python -m repro.multiview_experiment --help` 查看接口（必选 `--scheme B` 或 `--scheme C`）。

本地验证：`python -m unittest discover -s tests -v`。

外部Dinomaly源码、原始数据、渲染元数据和权重不纳入Git；仅克隆仓库不能直接运行全流程。具体依赖见[方法说明](docs/methods.md)。

`.gitignore`排除数据、模型、环境、日志、结果和传输包。本地已有结果仍在`outputs/`中。远程启动包装、过程Markdown与计划已移出源码树；此次整理不修改服务器或已有模型。
