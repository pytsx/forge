import logging
import os
from pathlib import Path

import structlog

from dollforge.adapters.baseline import EllipsoidReconstructor, SilhouetteSegmenter
from dollforge.adapters.blender import HeadlessBlender
from dollforge.adapters.contour import ContourDollSegmenter
from dollforge.adapters.grounded_sam2 import GroundedSam2Segmenter
from dollforge.adapters.matching import MultiSignalMatcher
from dollforge.adapters.templates import DollTemplateReconstructor
from dollforge.orchestration import Engine
from dollforge.service import Service
from dollforge.storage import Store


def create_service(root: Path | None = None) -> Service:
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level, structlog.processors.JSONRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = Store(root or Path(os.environ.get("DOLLFORGE_DATA", ".dollforge")))
    baseline = SilhouetteSegmenter()
    contour = ContourDollSegmenter()
    templates = DollTemplateReconstructor()
    segmenters = {contour.model_id: contour}
    if os.environ.get("DOLLFORGE_GDINO_CONFIG") and os.environ.get("DOLLFORGE_GDINO_CHECKPOINT"):
        segmenters["grounded_sam2_v1"] = GroundedSam2Segmenter.from_environment()
    return Service(store, Engine(
        store,
        baseline,
        EllipsoidReconstructor(),
        HeadlessBlender(),
        segmenters=segmenters,
        matcher=MultiSignalMatcher(),
        reconstructors={templates.model_id: templates},
    ))
