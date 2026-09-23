"""RA-MSDFlow 的 protocols 模块。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DomainAssignment:
    environment: str
    family: str
    is_primary: bool
    source_domain: str
    variant_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_ROBUSTAD: dict[tuple[str, str], tuple[str, str, bool]] = {
    ("MetalParts", "test0"): ("clean", "clean", True),
    ("MetalParts", "test1"): ("lighting", "photometric", True),
    ("MetalParts", "test2"): ("position", "geometric", False),
    ("MetalParts", "test3"): ("rotation", "geometric", False),
    ("MetalParts", "test4"): ("scale", "geometric", False),
    ("MetalParts", "test5"): ("background_1", "background", False),
    ("MetalParts", "test6"): ("background_2", "background", False),
    ("PCB", "test0"): ("clean", "clean", True),
    ("PCB", "test1"): ("lighting", "photometric", True),
    ("PCB", "test2"): ("white_balance", "photometric", True),
    ("PCB", "test3"): ("rotation", "geometric", False),
    ("PCB", "test4"): ("position", "geometric", False),
    ("PCB", "test5"): ("shadow", "photometric", False),
    ("PiledBags", "test0"): ("clean", "clean", True),
    ("PiledBags", "test1"): ("lighting", "photometric", True),
    ("PiledBags", "test2"): ("background", "background", False),
    ("PiledBags", "test3"): ("position_rotation", "geometric", False),
    ("PiledBags", "test4"): ("scale", "geometric", False),
    ("PiledBags", "test5"): ("shadow", "photometric", False),
}


def assign_domain(dataset: str, category: str, domain: str, variant_id: str = "clean") -> DomainAssignment:
    """执行 `assign_domain` 所需的处理。"""

    # 步骤 1：按当前协议处理。
    if variant_id in {"exposure", "white_balance", "gradient", "compound"}:
        return DomainAssignment(variant_id, "photometric", True, domain, variant_id)
    if dataset == "robustad":
        environment, family, primary = _ROBUSTAD.get(
            (category, domain), (domain, "other", False)
        )
        return DomainAssignment(environment, family, primary, domain, variant_id)
    if dataset == "aebad_s":
        mapping = {
            "same": ("clean", "clean", True),
            "illumination": ("lighting", "photometric", True),
            "background": ("background", "background", False),
            "view": ("view", "geometric", False),
        }
        environment, family, primary = mapping.get(domain, (domain, "other", False))
        return DomainAssignment(environment, family, primary, domain, variant_id)
    if dataset == "mvtec_ad2":
        if domain == "regular":
            values = ("clean", "clean", True)
        elif domain in {"overexposed", "underexposed"}:
            values = (domain, "photometric", True)
        elif domain == "mixed":
            values = ("mixed", "mixed", False)
        else:
            values = (domain, "other_shift", False)
        return DomainAssignment(*values, source_domain=domain, variant_id=variant_id)
    # 步骤 2：按当前协议处理。
    environment = "clean" if domain in {"regular", "same", "test0"} else domain
    family = "clean" if environment == "clean" else "other"
    return DomainAssignment(environment, family, environment == "clean", domain, variant_id)
