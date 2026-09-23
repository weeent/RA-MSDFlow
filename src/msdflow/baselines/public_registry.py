"""主表公开源码方法及固定提交。

仓库固定到具体 commit，避免作者后续更新让主表无法复现。这里只登记作者源码；
未公开作者实现的方法不会进入主表。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PublicBaselineSpec:
    """一个第三方 baseline 的来源与本地目录约定。"""

    key: str
    paper_name: str
    family: str
    repository_url: str
    commit: str
    local_directory: str
    main_table: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PUBLIC_BASELINES: dict[str, PublicBaselineSpec] = {
    "wtflow": PublicBaselineSpec(
        key="wtflow",
        paper_name="WT-Flow",
        family="flow_matching",
        repository_url="https://github.com/lil-wayne-0319/fmad.git",
        commit="f621f2b251637605a7d25593251d3be89a8c9734",
        local_directory="third_party/wt_flow_fmad",
        main_table=True,
        reason="与本文最直接的无条件 Flow Matching 对照。",
    ),
    "refp": PublicBaselineSpec(
        key="refp",
        paper_name="ReFP-AD",
        family="flow_and_energy",
        repository_url="https://github.com/CLendering/ReFP-AD.git",
        commit="0118c9899926e1394bd411aba23ee45f0b07407e",
        local_directory="third_party/ReFP-AD",
        main_table=True,
        reason="公开源码的强流式密度/能量模型对照。",
    ),
    "msflow": PublicBaselineSpec(
        key="msflow",
        paper_name="MSFlow",
        family="normalizing_flow",
        repository_url="https://github.com/cool-xuan/msflow.git",
        commit="e23975598bc7eb6f74604739d8f04d1519d704da",
        local_directory="third_party/msflow",
        main_table=True,
        reason="经典多尺度 normalizing-flow 工业异常检测基线。",
    ),
    "rdpp": PublicBaselineSpec(
        key="rdpp",
        paper_name="RD++",
        family="reverse_distillation",
        repository_url="https://github.com/tientrandinh/Revisiting-Reverse-Distillation.git",
        commit="7f2ceb7c87e602617b8600e1a498f7ef7f5247d6",
        local_directory="third_party/rdpp",
        main_table=True,
        reason="非流式强表征/蒸馏基线，用来判断收益是否只来自换检测器。",
    ),
    "gnl": PublicBaselineSpec(
        key="gnl",
        paper_name="GNL (ADShift)",
        family="domain_generalization",
        repository_url="https://github.com/mala-lab/ADShift.git",
        commit="56a87badd2e767dcb90b3b2ec00155d1e26c4bd1",
        local_directory="third_party/ADShift",
        main_table=True,
        reason="直接面向异常检测域移的公开源码基线。",
    ),
    "mmr": PublicBaselineSpec(
        key="mmr",
        paper_name="MMR",
        family="domain_generalization",
        repository_url="https://github.com/zhangzilongc/MMR.git",
        commit="22d1b171687509024e6f5b53b5a5e772d653d32b",
        local_directory="third_party/MMR",
        main_table=False,
        reason="作者实现主要围绕 AeBAD；跨三套主数据移植成本高，保留为补充实验。",
    ),
}


def main_table_specs() -> tuple[PublicBaselineSpec, ...]:
    """返回论文主表中真正需要执行的第三方方法。"""

    return tuple(spec for spec in PUBLIC_BASELINES.values() if spec.main_table)
