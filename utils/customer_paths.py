from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DERIVED_BASE = ROOT / "DataSets" / "Derived"
RAW_BASE = ROOT / "DataSets" / "Raw"


@dataclass(frozen=True)
class CustomerPaths:
    key: str
    raw_dir: Path
    derived_dir: Path

    @property
    def qa_dir(self) -> Path:
        return self.derived_dir / "QA"


DEFAULT_CUSTOMER = "northernlights"

_CUSTOMER_MAP = {
    "northernlights": {"raw": "NorthernLightsTest", "derived": "NorthernLights"},
    "stena": {"raw": "Stena", "derived": "Stena"},
}

_ALIASES = {
    "northernlightstest": "northernlights",
    "nl": "northernlights",
    "northern_lights": "northernlights",
    "stena": "stena",
    "saga": "stena",
}


def canonical_customers() -> List[str]:
    """Return the sorted list of canonical customer identifiers."""
    return sorted(_CUSTOMER_MAP.keys())


def customer_aliases() -> Iterable[str]:
    """Return all recognised aliases (canonical names included)."""
    yield from _CUSTOMER_MAP.keys()
    yield from _ALIASES.keys()


def resolve_customer(customer: str | None) -> CustomerPaths:
    """Translate a customer slug or alias into raw/derived directories."""
    key = (customer or DEFAULT_CUSTOMER).lower()
    key = _ALIASES.get(key, key)
    config = _CUSTOMER_MAP.get(key)
    if config is None:
        available = ", ".join(canonical_customers())
        raise ValueError(f"Unknown customer '{customer}'. Available options: {available}.")

    raw_dir = RAW_BASE / config["raw"]
    derived_dir = DERIVED_BASE / config["derived"]
    return CustomerPaths(key=key, raw_dir=raw_dir, derived_dir=derived_dir)


def ensure_customer_dirs(paths: CustomerPaths) -> None:
    """Create derived + QA subdirectories for the selected customer."""
    paths.derived_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = paths.qa_dir
    (qa_dir / "figures").mkdir(parents=True, exist_ok=True)
    (qa_dir / "tables").mkdir(parents=True, exist_ok=True)
    (qa_dir / "reports").mkdir(parents=True, exist_ok=True)


def describe_customers() -> str:
    """Textual summary of available customers and notable aliases."""
    lines = ["Available customers:"]
    for key in canonical_customers():
        raw = _CUSTOMER_MAP[key]["raw"]
        derived = _CUSTOMER_MAP[key]["derived"]
        alias_list = [alias for alias, target in _ALIASES.items() if target == key]
        alias_txt = f" (aliases: {', '.join(sorted(alias_list))})" if alias_list else ""
        lines.append(f"  - {key} -> raw: {raw}, derived: {derived}{alias_txt}")
    return "\n".join(lines)
