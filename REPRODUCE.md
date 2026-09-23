# RA-MSDFlow：训练与复现 / Training and Reproduction

以下命令均在仓库根目录运行，以 RobustAD/PCB 的单类别闭环为例。需要 Python 3.10+，并先安装与本机 CUDA 驱动匹配的 PyTorch 与 torchvision。

All commands below run from the repository root and illustrate one RobustAD/PCB category. Use Python 3.10+ and install a PyTorch/torchvision build compatible with the local CUDA driver first.

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH=src
```

## 1. 数据、权重与公开 baseline / Data, weights, and public baselines

将公开数据放到 `datasets/RobustAD`、`datasets/AeBAD` 和 `datasets/mvtec_ad_2`，并将 WRN-50-2 权重置于 `weights/torchvision/wide_resnet50_2-95faca4d.pth`，或在配置中改用自己的相对路径。按固定提交拉取公开 baseline：

Place public datasets under `datasets/RobustAD`, `datasets/AeBAD`, and `datasets/mvtec_ad_2`. Put the WRN-50-2 weight at `weights/torchvision/wide_resnet50_2-95faca4d.pth`, or update the configuration with another relative path. Fetch public baselines at their pinned revisions:

```bash
python3 scripts/fetch_public_baselines.py --root .
```

按数据集发布方的说明准备数据。仓库不含图像或标注；第三方 baseline 的额外依赖见 `requirements-public-baselines.txt`。

Prepare datasets following their publishers' instructions. Images and annotations are not bundled; additional third-party baseline dependencies are listed in `requirements-public-baselines.txt`.

## 2. 数据准备 / Data preparation

```bash
python3 -m envfm.cli.prepare_all --config configs/datasets/prepare_main.example.json
```

该命令写出 `manifests/<dataset>/prepared_seed9826.jsonl` 与准备摘要。确认 normal train/val 只含正常样本、两者不共享 `base_id`，并检查审计标志。manifest 保存本机路径，应在实际训练机器重新生成。

The command writes `manifests/<dataset>/prepared_seed9826.jsonl` and a preparation summary. Verify that normal train/val contain normal samples only, share no `base_id`, and pass the audit flags. Manifests contain local paths and must be regenerated on the training machine.

## 3. 源域正常性模型训练 / Source normality-model training

示例配置默认对应 `robustad/PCB`。其他类别需同步修改数据集、类别、manifest、缓存、checkpoint 与输出目录。数据划分种子固定为 9826；目标参考抽样种子在下一节设置。

The examples target `robustad/PCB`. For another category, update the dataset, category, manifest, cache, checkpoint, and output paths together. The split seed is fixed at 9826; target-reference seeds are set in the next section.

```bash
python3 -m msdflow.cli.main train-descriptor --config configs/msdflow/descriptor.example.json
python3 -m msdflow.cli.main cache --config configs/msdflow/cache_train.example.json
python3 -m msdflow.cli.main cache --config configs/msdflow/cache_val.example.json
python3 -m msdflow.cli.main fit-conditions --config configs/msdflow/conditions.example.json
python3 -m msdflow.cli.main train-transport --config configs/msdflow/transport.example.json
python3 -m msdflow.cli.main train-normality --config configs/msdflow/normality.example.json
```

训练产物为 `inference_bundle.pt`。目标环境阶段保持编码器、环境描述、模式库和正常性流冻结。

The output is `inference_bundle.pt`. The encoder, environment descriptor, mode bank, and normality flow remain frozen during target-environment processing.

## 4. 目标参考对齐与 RA-MSDFlow 推理 / Target-reference alignment and RA-MSDFlow inference

复制 `configs/msdflow/reference_otcfm.example.json`，填写该类别的 `inference_bundle`、目标环境、reference 数量和输出目录：

Copy `configs/msdflow/reference_otcfm.example.json` and set the category-specific `inference_bundle`, target environment, reference count, and output directory:

```bash
python3 -m msdflow.cli.main reference-otcfm --config configs/msdflow/reference_otcfm.example.json
```

该历史入口会同时写出 `frozen_source`、`target_threshold_only`、`coral` 和 `reference_otcfm`。最终 RA-MSDFlow 仅使用 `coral`：以目标正常参考拟合 Shrinkage CORAL，用冻结 MSD-Flow 评分，并用五折 out-of-fold 参考分数校准 5% 工作点。目标异常不参与对齐、校准或模型选择。

This historical entry writes `frozen_source`, `target_threshold_only`, `coral`, and `reference_otcfm`. Final RA-MSDFlow uses only `coral`: Shrinkage CORAL is fitted on target-normal references, a frozen MSD-Flow produces scores, and five-fold out-of-fold reference scores calibrate the 5% operating point. Target anomalies never enter alignment, calibration, or model selection.

可按主表矩阵生成作业计划：RobustAD 使用 20-shot、3 reference seeds；AeBAD-S 使用 20-shot、3 seeds；MVTec AD 2 使用 2-shot、1 seed。先按实际设备调整示例配置，再运行：

Generate the main-table job plan as follows: RobustAD uses 20 shots and three reference seeds; AeBAD-S uses 20 shots and three seeds; MVTec AD 2 uses two shots and one seed. Adjust the example configuration for the available devices, then run:

```bash
python3 -m msdflow.cli.reference_otcfm_matrix --matrix configs/msdflow/reference_alignment_matrix.example.json --output-directory outputs/ra_main_plan
python3 -m msdflow.cli.reference_otcfm_matrix --matrix configs/msdflow/reference_alignment_matrix.example.json --output-directory outputs/ra_main_plan --execute
```

I-AUROC 使用连续原始图像分数；FPR/TPR 使用目标参考尾部校准后的 5% 工作点。每个 seed 内先平均环境与类别，再跨 seed 报告均值和标准差。没有像素 mask 的 PiledBags 不报告像素指标。

I-AUROC uses continuous raw image scores; FPR/TPR use the 5% operating point calibrated from target references. Average environments and categories within each seed, then report mean and standard deviation across seeds. Do not report pixel metrics for PiledBags, which has no pixel masks.

## 5. 公开 baseline、消融与最小核验 / Public baselines, ablation, and minimal checks

公开 baseline 使用相同的 source-normal 训练划分；目标正常参考只用于分数校准，不参与其模型训练。`method` 依次设为 `wtflow`、`msflow`、`rdpp`、`refp`、`gnl`。

Public baselines use the same source-normal split. Target-normal references calibrate scores only and do not train their models. Set `method` to `wtflow`, `msflow`, `rdpp`, `refp`, or `gnl`.

```bash
python3 -m msdflow.cli.run_public_baseline --config configs/msdflow/public_baseline.example.json
python3 -m msdflow.cli.summarize_public_baselines --root outputs/public_baselines --output-directory outputs/public_baselines/tables
python3 -m msdflow.cli.main coral-wtflow --config configs/msdflow/coral_wtflow_ablation.example.json
```

评分器消融在 RobustAD 上以同一对齐和参考划分比较冻结 WT-Flow 与 MSD-Flow；不重训 WT-Flow。两者必须共享 manifest、reference `base_id`、WRN 特征与目标查询。

The scorer ablation compares frozen WT-Flow and MSD-Flow on RobustAD under the same alignment and reference split; WT-Flow is not retrained. Both methods must share the manifest, reference `base_id`, WRN features, and target queries.

```bash
python3 -m msdflow.cli.main --help
python3 -m unittest discover -s tests -p 'test_msdflow*.py' -v
python3 -m unittest tests.test_reference_otcfm tests.test_public_baseline_protocol tests.test_coral_wtflow tests.test_summarize_public_baselines -v
```

在一个类别上确认 bundle、reference 与评价样本无交集，所有分数有限，且每张 reference 正好被 held out 一次，再运行完整矩阵。

For one category, first confirm that the bundle is valid, references and evaluation samples are disjoint, all scores are finite, and each reference is held out exactly once. Then run the full matrix.
