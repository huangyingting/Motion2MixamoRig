"""Motion extractors. Each one reads a video or a still and writes skeleton_motion.npz."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ..models import DEFAULT_MODEL, MODEL_SPECS, get_model_spec

if TYPE_CHECKING:
    from .base import MotionExtractor

DEFAULT_EXTRACTOR = DEFAULT_MODEL


def extractor_names() -> tuple[str, ...]:
    return tuple(MODEL_SPECS)


@lru_cache(maxsize=None)
def get_extractor(name: str) -> MotionExtractor:
    spec = get_model_spec(name)
    if spec["adapter"] == "gvhmr":
        from .gvhmr import GvhmrExtractor

        return GvhmrExtractor()
    from .external import ExternalExtractor

    return ExternalExtractor(name)
