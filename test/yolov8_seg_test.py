"""
test/yolov8_seg_test.py
Prototype / test purpose only — not for production use.

Run with:
    pytest test/yolov8_seg_test.py -v
"""

import sys
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from yolo import YoloSegmentor, draw_segmentation

MODEL_PATH = REPO_ROOT / "models/yolov8n-seg.pt"
DOG_IMAGE  = REPO_ROOT / "test/resources/object_extraction_dog.png"
OUTPUT_DIR = REPO_ROOT / "test/resources/output"


def test_yolov8_seg_outputs_result_image():
    segmentor = YoloSegmentor(MODEL_PATH)
    result = segmentor.run(DOG_IMAGE, conf=0.25)

    image = cv2.imread(str(DOG_IMAGE))
    output = draw_segmentation(image, result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{DOG_IMAGE.stem}_yolov8seg.png"
    cv2.imwrite(str(output_path), output)

    assert output_path.exists(), f"Output image was not saved: {output_path}"
