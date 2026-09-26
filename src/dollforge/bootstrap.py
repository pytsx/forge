import logging
import os
from pathlib import Path

import structlog

from dollforge.adapters.baseline import EllipsoidReconstructor, SilhouetteSegmenter
from dollforge.adapters.blender import HeadlessBlender
from dollforge.orchestration import Engine
from dollforge.service import Service
from dollforge.storage import Store


def create_service(root: Path | None = None) -> Service:
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level, structlog.processors.JSONRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = Store(root or Path(os.environ.get("DOLLFORGE_DATA", ".dollforge")))
    return Service(store, Engine(store, SilhouetteSegmenter(), EllipsoidReconstructor(), HeadlessBlender()))
