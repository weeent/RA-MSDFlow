"""RA-MSDFlow 的 __init__ 模块。"""

from importlib import import_module

__all__ = [
    "ConditionStandardizer",
    "IndustrialAnomalyDataset",
    "PhotometricDescriptor",
    "SampleRecord",
    "apply_photometric_transform",
    "audit_records",
    "build_aebad_manifest",
    "build_dataset_manifest",
    "build_mvtec_ad2_manifest",
    "build_mvtec_ad_manifest",
    "build_mpdd_manifest",
    "build_robustad_manifest",
    "build_visa_manifest",
    "expand_training_photometric_variants",
    "expand_evaluation_photometric_variants",
    "fit_condition_standardizer",
    "prepare_records_for_training",
    "read_records_jsonl",
    "sample_photometric_params",
    "stable_normal_split",
    "write_records_jsonl",
]


_EXPORT_MODULES = {
    "ConditionStandardizer": ".photometric_descriptor",
    "IndustrialAnomalyDataset": ".dataset",
    "PhotometricDescriptor": ".photometric_descriptor",
    "SampleRecord": ".records",
    "apply_photometric_transform": ".photometric_augment",
    "audit_records": ".manifest_builders",
    "build_aebad_manifest": ".manifest_builders",
    "build_dataset_manifest": ".manifest_builders",
    "build_mvtec_ad2_manifest": ".manifest_builders",
    "build_mvtec_ad_manifest": ".manifest_builders",
    "build_mpdd_manifest": ".manifest_builders",
    "build_robustad_manifest": ".manifest_builders",
    "build_visa_manifest": ".manifest_builders",
    "expand_training_photometric_variants": ".splits",
    "expand_evaluation_photometric_variants": ".splits",
    "fit_condition_standardizer": ".dataset",
    "prepare_records_for_training": ".prepare",
    "read_records_jsonl": ".records",
    "sample_photometric_params": ".photometric_augment",
    "stable_normal_split": ".splits",
    "write_records_jsonl": ".records",
}


def __getattr__(name: str):
    """执行 `__getattr__` 所需的处理。"""

    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name], __name__)
    return getattr(module, name)
