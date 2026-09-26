from __future__ import annotations

import os
from io import BytesIO

import numpy as np
from PIL import Image

from dollforge.adapters.baseline import png
from dollforge.contracts import MaskProposal, SegmentationRequest
from dollforge.domain.models import PartClass, Provenance, Side
from dollforge.errors import DependencyUnavailable, InvalidInput


PROMPT_LABELS: dict[str, tuple[PartClass, Side]] = {
    "head": (PartClass.HEAD, Side.CENTER),
    "face": (PartClass.FACE, Side.CENTER),
    "hair": (PartClass.HAIR, Side.CENTER),
    "torso": (PartClass.TORSO, Side.CENTER),
    "pelvis": (PartClass.PELVIS, Side.CENTER),
    "left upper arm": (PartClass.UPPER_ARM, Side.LEFT),
    "right upper arm": (PartClass.UPPER_ARM, Side.RIGHT),
    "left forearm": (PartClass.FOREARM, Side.LEFT),
    "right forearm": (PartClass.FOREARM, Side.RIGHT),
    "left hand": (PartClass.HAND, Side.LEFT),
    "right hand": (PartClass.HAND, Side.RIGHT),
    "left thigh": (PartClass.THIGH, Side.LEFT),
    "right thigh": (PartClass.THIGH, Side.RIGHT),
    "left shin": (PartClass.SHIN, Side.LEFT),
    "right shin": (PartClass.SHIN, Side.RIGHT),
    "left foot": (PartClass.FOOT, Side.LEFT),
    "right foot": (PartClass.FOOT, Side.RIGHT),
    "left shoe": (PartClass.FOOTWEAR, Side.LEFT),
    "right shoe": (PartClass.FOOTWEAR, Side.RIGHT),
    "shirt": (PartClass.TOP, Side.CENTER),
    "pants": (PartClass.BOTTOM, Side.CENTER),
}

PROMPT = " . ".join(PROMPT_LABELS)


def normalize_phrase(value: str) -> str:
    phrase = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
    aliases = {
        "upper arm left": "left upper arm",
        "upper arm right": "right upper arm",
        "forearm left": "left forearm",
        "forearm right": "right forearm",
        "hand left": "left hand",
        "hand right": "right hand",
        "thigh left": "left thigh",
        "thigh right": "right thigh",
        "shin left": "left shin",
        "shin right": "right shin",
        "foot left": "left foot",
        "foot right": "right foot",
        "shoe left": "left shoe",
        "shoe right": "right shoe",
    }
    return aliases.get(phrase, phrase)


class GroundedSam2Segmenter:
    """Grounding DINO proposals refined by SAM 2 masks.

    Heavy dependencies stay optional. This adapter is instantiated only when its
    local model configuration is present.
    """

    model_id = "grounded_sam2_v1"
    model_version = "1.0.0"

    def __init__(
        self,
        dino_config: str,
        dino_checkpoint: str,
        sam_model_id: str = "facebook/sam2.1-hiera-base-plus",
        device: str = "cuda",
        box_threshold: float = 0.28,
        text_threshold: float = 0.22,
    ):
        try:
            import torch
            from groundingdino.util.inference import Model
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise DependencyUnavailable(
                "Grounded SAM2 requer PyTorch, GroundingDINO e SAM 2 instalados no ambiente de visão."
            ) from exc

        self.torch = torch
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.dino = Model(
            model_config_path=dino_config,
            model_checkpoint_path=dino_checkpoint,
            device=device,
        )
        self.sam = SAM2ImagePredictor.from_pretrained(sam_model_id, device=device)
        self.model_version = f"groundingdino+{sam_model_id}"

    @classmethod
    def from_environment(cls) -> "GroundedSam2Segmenter":
        config = os.environ.get("DOLLFORGE_GDINO_CONFIG")
        checkpoint = os.environ.get("DOLLFORGE_GDINO_CHECKPOINT")
        if not config or not checkpoint:
            raise DependencyUnavailable(
                "Configure DOLLFORGE_GDINO_CONFIG e DOLLFORGE_GDINO_CHECKPOINT."
            )
        return cls(
            dino_config=config,
            dino_checkpoint=checkpoint,
            sam_model_id=os.environ.get(
                "DOLLFORGE_SAM2_MODEL", "facebook/sam2.1-hiera-base-plus"
            ),
            device=os.environ.get("DOLLFORGE_VISION_DEVICE", "cuda"),
            box_threshold=float(os.environ.get("DOLLFORGE_GDINO_BOX_THRESHOLD", "0.28")),
            text_threshold=float(os.environ.get("DOLLFORGE_GDINO_TEXT_THRESHOLD", "0.22")),
        )

    def predict(self, request: SegmentationRequest) -> list[MaskProposal]:
        rgb = np.asarray(Image.open(BytesIO(request.image_png)).convert("RGB"))
        # GroundingDINO's high-level Model API expects BGR numpy input.
        bgr = rgb[:, :, ::-1].copy()
        detections, phrases = self.dino.predict_with_caption(
            image=bgr,
            caption=PROMPT,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        boxes = np.asarray(detections.xyxy)
        confidences = np.asarray(detections.confidence)
        if len(boxes) != len(phrases):
            raise InvalidInput("Grounding DINO retornou boxes e labels incompatíveis.")

        proposals: list[MaskProposal] = []
        self.sam.set_image(rgb)
        inference = self.torch.inference_mode()
        with inference:
            for box, confidence, phrase in zip(boxes, confidences, phrases, strict=True):
                semantic = PROMPT_LABELS.get(normalize_phrase(phrase))
                if semantic is None:
                    continue
                masks, scores, _ = self.sam.predict(
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=False,
                )
                if len(masks) == 0:
                    continue
                index = int(np.argmax(scores))
                mask = np.asarray(masks[index], dtype=bool)
                ys, xs = np.where(mask)
                if len(xs) < 4:
                    continue
                part_class, side = semantic
                combined_confidence = float(
                    max(0.0, min(1.0, float(confidence) * float(scores[index])))
                )
                mask_png = png(Image.fromarray((mask * 255).astype("uint8")))
                proposals.append(
                    MaskProposal(
                        part_class=part_class.value,
                        side=side.value,
                        bbox_xyxy=(
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()) + 1,
                            int(ys.max()) + 1,
                        ),
                        confidence=combined_confidence,
                        provenance=Provenance(
                            type="model_inferred",
                            source=self.model_id,
                            evidence=[request.view.normalized_artifact_id],
                            note=f"Grounding label '{phrase}' refinado por SAM 2.",
                        ),
                        mask_png=mask_png,
                    )
                )
        return proposals
