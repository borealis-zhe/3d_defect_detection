# 方法实现

## 当前多视图主线

| 方法 | 实现 | 关键约束 |
|---|---|---|
| B/C共享训练与评价 | `repro/multiview_experiment.py` | 冻结编码器，每类5000步、batch16、seed1；B/C独立数据、缓存及结果目录 |
| B双侧渲染 | `repro/bilateral_preprocess.py` | 主侧和对侧各13方向；每视图法线朝相机 |
| C全方位渲染 | `repro/global_preprocess.py` | 26全方位方向；部分视图为避免中心裁剪切边后移相机 |
| 配准后图像与标签映射 | `repro/aligned_preprocess.py` | 标签在姿态固定后读入；保持原始点ID |
| 原始归一化数据上的配准 | `repro/aligned_registration.py` | PCA候选、单向截尾ICP；读取已有归一化元数据 |
| 外侧相机估计 | `repro/camera_preprocess.py`、`scripts/scan_direction_audit.py` | 局部扫描到正常模板匹配；方向不确定性单独记录 |
| 全量外侧相机渲染及评分 | `repro/exterior_camera_experiment.py` | 纳入全部603对象；使用旧pose模型作相机对照，不是B/C模型入口 |

网络输入由224放大至448再中心裁剪至392；ViT-B/14得到28×28空间网格。两组教师/解码特征的余弦重建误差平均，放大为异常图后按点像素映射取值；当前B/C对15视图分数直接平均，物体取最大点分数。没有新增跨视图注意力、辅助头或默认可见性加权。

## 历史方法与诊断实现

- `repro/pose_experiment.py`：原27图姿态配准基线训练/评估；与B/C输入不同。
- `repro/camera_subset_experiment.py`：仅接纳可靠候选的历史相机子集对照，不能与603对象主结果混报。
- `repro/baseline.py`、`geometry.py`、`preprocess.py`：早期独立正交深度外观方案；与当前Open3D灰色外观不是同一渲染协议。
- `repro/bundle_ablate.py`、`ablate.py`：交接包冻结模型的逐视图分数融合。
- `repro/coverage_ablate.py`、`raw_coverage.py`：视角预算与覆盖率。
- `repro/full_surface.py`、`selection_study.py`：原始可见点域及不同选点方法。
- `repro/geometry_memory.py`、`geometry_density_audit.py`：正常几何记忆及密度敏感性。
- `repro/pose_registration.py`及其他`pose_*`模块：本地配准、几何审计与重渲染诊断。
- `scripts/`：保留可复算分析与绘图程序，过程文字已清理。

## 外部依赖与数据契约

B/C公共实现默认从项目相邻的`real3d-dinomaly-20261026/experiments/dinomaly_mv/runtime.py`加载外部运行时；可用`REAL3D_RUNTIME_ROOT`设置该外部项目根目录。运行时仍依赖其上游Dinomaly源码。本项目权重位置为`checkpoints/dinov2_vitb14_reg4_pretrain.pth`。

B/C预处理依赖`scripts/preprocess_pcd.py`的外部渲染实现（本仓库未内置）、原始PCD压缩包、`data/pcd_self_v1`归一化元数据及`data/pose_transforms`。现有默认布局仍对应原实验的数据位置，未借整理改变预处理算法。

B训练数据/缓存/结果分别为`data/bilateral26_v1`、`data/bilateral26_cache_v1`、`results/bilateral26_v1`；C为`data/global26_v2`、`data/global26_cache_v2`、`results/global26_v2`。共同测试图像为`data/exterior_camera_all_v1`。

交接包实验读取`data/handoff`，早期独立基线使用`third_party/Dinomaly`。以上数据、源码依赖和模型均被Git忽略。

所有移入repro的模块从项目根目录通过`python -m repro.<模块名>`使用，以保证包内引用正确。移动导致源码哈希改变，旧断点可能拒绝续训，这是原有一致性检查；不能删除校验来强制续跑。已完成权重与分数未修改。
