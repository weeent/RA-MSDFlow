# RA-MSDFlow

## 中文

这是 RA-MSDFlow 的 camera-ready 源码包。它包含数据准备、源域正常性模型训练、目标正常参考对齐、推理评估、公开 baseline 适配器、示例配置和必要单元测试。原始数据、特征缓存、模型权重、实验输出和第三方仓库均不随本仓库分发。

| 路径 | 内容 |
|---|---|
| `src/msdflow/` | 环境描述、模式库、正常性 Flow Matching、参考对齐、校准、评估和命令行入口 |
| `src/envfm/` | 数据清单、冻结特征编码器、Flow Matching 路径及评估基础组件 |
| `integrations/public_baselines/` | WT-Flow、MSFlow、RD++、ReFP-AD、GNL 的统一协议适配器 |
| `configs/` | 相对路径配置示例及公开 baseline 的固定提交记录 |
| `tests/` | 源码级测试，不包含实验侧验收脚本 |
| `scripts/fetch_public_baselines.py` | 按固定提交获取公开作者代码 |
| `REPRODUCE.md` | 数据准备、训练、参考对齐和主表复现说明 |

本工作使用 RobustAD、AeBAD-S 与 MVTec AD 2。请按数据集发布方要求下载数据；示例目录分别为 `datasets/RobustAD/`、`datasets/AeBAD/`、`datasets/mvtec_ad_2/`。

主表的公开源码 baseline 为 [WT-Flow](https://github.com/lil-wayne-0319/fmad)、[ReFP-AD](https://github.com/CLendering/ReFP-AD)、[MSFlow](https://github.com/cool-xuan/msflow)、[RD++](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) 和 [GNL/ADShift](https://github.com/mala-lab/ADShift)。固定提交见 `configs/baselines/public_baselines.lock.json`；第三方源码不包含在本仓库中。

最终 RA-MSDFlow 在目标正常参考上拟合 Shrinkage CORAL，并用冻结的源 MSD-Flow 正常性模型评分。历史入口名为 `reference-otcfm`，正式结果只读取其中的 `coral` 输出；附带的 `reference_otcfm` 输出不是最终方法。

## English

This is the camera-ready source package for RA-MSDFlow. It contains data preparation, source-normality training, target-normal reference alignment, inference and evaluation, public-baseline adapters, example configurations, and essential unit tests. Raw images, feature caches, model weights, experimental outputs, and third-party repositories are intentionally excluded.

| Path | Contents |
|---|---|
| `src/msdflow/` | Environment descriptors, mode bank, normality Flow Matching, reference alignment, calibration, evaluation, and CLI entry points |
| `src/envfm/` | Manifests, frozen feature encoder, Flow Matching path, and shared evaluation utilities |
| `integrations/public_baselines/` | Unified protocol adapters for WT-Flow, MSFlow, RD++, ReFP-AD, and GNL |
| `configs/` | Relative-path examples and pinned public-baseline revisions |
| `tests/` | Source-level tests; execution-side acceptance scripts are excluded |
| `scripts/fetch_public_baselines.py` | Fetches public author code at pinned revisions |
| `REPRODUCE.md` | Instructions for data preparation, training, reference alignment, and main-table reproduction |

The project uses RobustAD, AeBAD-S, and MVTec AD 2. Obtain each dataset from its publisher and place it, by default, in `datasets/RobustAD/`, `datasets/AeBAD/`, and `datasets/mvtec_ad_2/`. `configs/datasets/prepare_main.example.json` is included. It creates manifests containing machine-local dataset paths at runtime; generated manifests must not be committed.

The public-source main-table baselines are [WT-Flow](https://github.com/lil-wayne-0319/fmad), [ReFP-AD](https://github.com/CLendering/ReFP-AD), [MSFlow](https://github.com/cool-xuan/msflow), [RD++](https://github.com/tientrandinh/Revisiting-Reverse-Distillation), and [GNL/ADShift](https://github.com/mala-lab/ADShift). Their pinned revisions are recorded in `configs/baselines/public_baselines.lock.json`; their source code is not bundled here.

Final RA-MSDFlow fits Shrinkage CORAL on verified target-normal references and scores aligned features with a frozen source MSD-Flow normality model. The historical CLI name is `reference-otcfm`; only its `coral` output is the final method. Its accompanying `reference_otcfm` output is not part of RA-MSDFlow.
