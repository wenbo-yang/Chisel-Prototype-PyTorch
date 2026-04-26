"""
src/yolo/segmentor.py
YOLOv8 instance segmentation library.
Prototype / test purpose only — not for production use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Colour palette (BGR) cycled per class index
_PALETTE: List[Tuple[int, int, int]] = [
    (56,  56,  255),
    (56,  255, 56),
    (255, 56,  56),
    (56,  255, 206),
    (255, 206, 56),
    (206, 56,  255),
]

DEFAULT_CONF       = 0.25
DEFAULT_BORDER_COLOR     = (0, 255, 0)   # BGR green
DEFAULT_BORDER_THICKNESS = 2
DEFAULT_MASK_ALPHA        = 0.35


@dataclass
class DetectedObject:
    """A single detected instance from YOLOv8 segmentation."""
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    mask:       np.ndarray                   # binary mask, same size as input image


@dataclass
class SegmentationResult:
    """Full result for one image inference run."""
    image_path:  Path
    image_shape: Tuple[int, int, int]        # (H, W, C)
    objects:     List[DetectedObject] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.objects)


class YoloSegmentor:
    """
    Thin wrapper around a YOLOv8-seg model for instance segmentation.

    Usage::

        seg = YoloSegmentor("models/yolov8n-seg.pt")
        result = seg.run("image.png", conf=0.25)
        overlay = draw_segmentation(cv2.imread("image.png"), result)
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model: YOLO | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    @property
    def model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(str(self.model_path))
        return self._model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def run(
        self,
        image_path: str | Path,
        conf: float = DEFAULT_CONF,
    ) -> SegmentationResult:
        """
        Run segmentation on *image_path* and return a :class:`SegmentationResult`.

        Args:
            image_path: Path to input image.
            conf:       Minimum confidence threshold for detections.

        Returns:
            SegmentationResult with all detected objects and their masks.

        Raises:
            FileNotFoundError: If *image_path* cannot be read.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]
        raw_results = self.model(image, conf=conf, verbose=False)
        raw = raw_results[0]

        result = SegmentationResult(
            image_path=image_path,
            image_shape=image.shape,
        )

        if raw.masks is None or raw.boxes is None:
            return result

        masks_data = raw.masks.data.cpu().numpy()   # (N, H_feat, W_feat)

        for i, mask_feat in enumerate(masks_data):
            mask_full = cv2.resize(
                mask_feat, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

            class_id = int(raw.boxes.cls[i].item())
            conf_val = float(raw.boxes.conf[i].item())
            x1, y1, x2, y2 = (int(v.item()) for v in raw.boxes.xyxy[i])

            result.objects.append(DetectedObject(
                class_id=class_id,
                class_name=raw.names[class_id],
                confidence=conf_val,
                bbox_xyxy=(x1, y1, x2, y2),
                mask=mask_full,
            ))

        return result


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def draw_segmentation(
    image: np.ndarray,
    result: SegmentationResult,
    border_color: Tuple[int, int, int] = DEFAULT_BORDER_COLOR,
    border_thickness: int = DEFAULT_BORDER_THICKNESS,
    mask_alpha: float = DEFAULT_MASK_ALPHA,
) -> np.ndarray:
    """
    Render segmentation masks and contour borders onto *image*.

    Args:
        image:            BGR image as numpy array (not modified in place).
        result:           :class:`SegmentationResult` from :meth:`YoloSegmentor.run`.
        border_color:     BGR colour for contour borders.
        border_thickness: Pixel thickness of contour lines.
        mask_alpha:       Opacity of filled mask overlay (0=transparent, 1=opaque).

    Returns:
        New numpy array with masks and borders drawn.
    """
    if not result.objects:
        return image.copy()

    base   = image.copy()
    overlay = image.copy()

    for obj in result.objects:
        palette_color = _PALETTE[obj.class_id % len(_PALETTE)]
        mask_bool = obj.mask.astype(bool)

        # Filled mask on overlay
        overlay[mask_bool] = palette_color

        # Contour border on base
        contours, _ = cv2.findContours(
            obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(base, contours, -1, border_color, border_thickness)

        # Label
        x1, y1 = obj.bbox_xyxy[0], obj.bbox_xyxy[1]
        label = f"{obj.class_name} {obj.confidence:.2f}"
        cv2.putText(
            base, label, (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1, cv2.LINE_AA,
        )

    blended = cv2.addWeighted(overlay, mask_alpha, base, 1 - mask_alpha, 0)
    return blended
