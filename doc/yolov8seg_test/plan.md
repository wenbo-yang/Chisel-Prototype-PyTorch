# YOLOv8 Segmentation — Prototype Test Plan

> **Purpose:** Evaluate YOLOv8 instance segmentation as a preprocessing step for object boundary detection in Project Chisel.

---

## Goal

Test whether YOLOv8-seg can detect and segment objects in our test imagery, and produce a visualisation showing segmentation masks with contour borders superimposed on the original image.

---

## Steps

### 1. Install Dependencies

Package required: `ultralytics` (bundles YOLOv8 + all model utilities).

```bash
pip install ultralytics
```

Dependencies pulled in automatically: `torch`, `torchvision`, `opencv-python`, `scipy`, `numpy`, `matplotlib`, `pyyaml`.

---

### 2. Model Setup — `yolov8n-seg`

- **Model:** `yolov8n-seg.pt` (nano segmentation variant, ~6.7 MB)
- **Trained on:** COCO dataset (80 classes including `person`, `cat`, `car`, etc.)
- **Download:** Automatic on first `YOLO("yolov8n-seg.pt")` call via the Ultralytics hub.
- **Inference API:**

```python
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
results = model(image, conf=0.01)
```

Each result provides:
- `result.boxes`  — bounding boxes, class IDs, confidence scores
- `result.masks`  — per-instance binary segmentation masks (N × H × W)
- `result.names`  — class name lookup dict

---

### 3. Test Image

| Property | Value |
|---|---|
| File | `src/test_data/running_man/running_man_image_5_preprocessed_mirror.png` |
| Content | Silhouette of a running person (preprocessed/greyscale) |
| Size | ~130 × 130 px |

**Note:** This is a heavily preprocessed silhouette image, not a natural photograph. YOLOv8 (COCO-trained) is optimised for natural images, so detections at the standard `conf=0.25` threshold return 0 objects. A lowered threshold of `conf=0.01` produces low-confidence detections (including `person`) sufficient to demonstrate the segmentation pipeline.

---

### 4. Inference & Output Rendering

Script: [`src/yolov8_seg_test.py`](../../src/yolov8_seg_test.py)

Pipeline:
1. Load `yolov8n-seg.pt`
2. Read test image with OpenCV
3. Run inference (`conf=0.01`)
4. For each detected instance:
   - Resize binary mask to original image dimensions
   - Fill mask region with a per-class colour on an overlay layer
   - Draw contour borders (`cv2.findContours` + `cv2.drawContours`) in green
   - Add class label + confidence text above the bounding box
5. Blend overlay onto original image (`alpha=0.35`)
6. Save output image

Output saved to: `src/test_data/running_man/output/running_man_image_5_preprocessed_mirror_yolov8seg.png`

---

### 5. Results

- **Detections at conf=0.01:** 10 objects (includes `person` @ 0.03, `cat` @ 0.12, etc.)
- Low confidence scores are expected — the silhouette image does not match COCO natural image statistics
- Segmentation masks and green contour borders are correctly rendered on the output image

---

## Observations & Next Steps

| Observation | Implication |
|---|---|
| COCO-trained model struggles with preprocessed silhouettes | For production use, consider fine-tuning on domain-specific data or using a model trained on silhouettes/binary images |
| Mask quality is coarse at nano model scale | Try `yolov8s-seg.pt` or larger for better boundary fidelity |
| Contour extraction from masks (`findContours`) works cleanly | This approach is viable for extracting object borders for downstream chisel path generation |
| Pipeline is end-to-end functional | Ready to test on natural/unprocessed input images |

---

## Useful References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLOv8 Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [COCO Dataset Classes](https://cocodataset.org/#explore)
