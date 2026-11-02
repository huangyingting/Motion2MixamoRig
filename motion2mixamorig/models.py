"""Model capabilities and installation contracts; standard library only."""

import json
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("model_catalog.json").read_text(encoding="utf-8"))
MODEL_SPECS = CATALOG["models"]
DEFAULT_MODEL = CATALOG["default"]


def get_model_spec(name: str) -> dict:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown extractor {name!r}; choose from {', '.join(MODEL_SPECS)}"
        ) from exc
