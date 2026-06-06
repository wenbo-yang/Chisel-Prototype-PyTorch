# Chisel-Prototype-PyTorch

> **Note:** This repository is for prototyping and testing purposes only. It is not intended for production use.

## Overview

This project contains PyTorch-based prototype models and tests for the Chisel image processing system, with a focus on YOLOv8 instance segmentation models.

---

## Quick Start

### 1. Setup Environment & Download Models

**Windows:**
```bash
.\scripts\setup.ps1
```

**macOS/Linux:**
```bash
bash scripts/setup.sh
```

This will:
- ✅ Install Python 3.11 (if needed)
- ✅ Install dependencies (PyTorch, OpenCV, YOLOv8, etc.)
- ✅ Download both segmentation models:
  - `yolov8n-seg.pt` (Nano, 7 MB) - Fast, suitable for real-time applications
  - `yolov8x-seg.pt` (Extra-Large, 137 MB) - High accuracy

### 2. Run Tests

```bash
# Run all tests
pytest test/ -v

# Run specific test with output
pytest test/yolov8_seg_test.py::test_yolov8_seg_cloudy_sky_panorama_large_model -v -s

# Run all segmentation tests with output capture disabled
pytest test/yolov8_seg_test.py -v -s
```

---

## Available Models

### YOLOv8 Nano (yolov8n-seg.pt)
- **Size**: 7 MB
- **Parameters**: ~3.2M
- **Speed**: Fastest ⚡
- **Use Case**: Edge devices, real-time processing, resource-constrained environments
- **Best For**: Quick inference, mobile/embedded deployment

### YOLOv8 Extra-Large (yolov8x-seg.pt)
- **Size**: 137 MB
- **Parameters**: ~71.8M
- **Speed**: Slower but more accurate 🎯
- **Use Case**: High-precision detection, server-side processing
- **Best For**: Maximum accuracy requirements

### Model Training
Both models are trained on the COCO dataset with 80 object classes including:
- People, animals (dog, cat, bird, etc.)
- Vehicles (car, truck, airplane, boat, etc.)
- Sports equipment, household items, furniture, etc.

**Limitations**: Cannot detect landscape features (grass, lawn, sky, water) - not part of COCO classes.

---

## Project Structure

```
Chisel-Prototype-PyTorch/
├── models/                  # YOLOv8 model weights
│   ├── yolov8n-seg.pt      # Nano model (7 MB)
│   └── yolov8x-seg.pt      # Extra-Large model (137 MB)
├── scripts/                 # Setup scripts
│   ├── setup.ps1           # Windows setup
│   └── setup.sh            # macOS/Linux setup
├── src/
│   ├── train_model.py      # Model training script
│   ├── yolo/               # YOLO segmentation library
│   │   ├── __init__.py
│   │   └── segmentor.py    # YoloSegmentor class
│   ├── test_data/          # Test images
│   └── training_data/      # Training dataset
├── test/
│   ├── yolov8_seg_test.py  # YOLOv8 segmentation tests
│   ├── resources/          # Test resources and images
│   │   ├── object_extraction_dog.png
│   │   ├── cloudy_sky_panorama_stockcake.png
│   │   └── output/         # Test output images
│   └── yolo8seg_test/      # Detailed test documentation
│       └── yolo_seg_object_classification_findings_cannot_determine_background.md
├── doc/
│   └── yolov8seg_test/     # Technical documentation
│       └── yolo_seg_object_classification_findings_cannot_determine_background.md
└── README.md               # This file
```

---

## Test Cases

### Test 1: Dog Image with Nano Model
- **File**: `test/yolov8_seg_test.py::test_yolov8_seg_outputs_result_image()`
- **Input**: `test/resources/object_extraction_dog.png`
- **Model**: YOLOv8 Nano
- **Expected**: Dog detection and segmentation
- **Output**: `test/resources/output/object_extraction_dog_yolov8seg.png`
- **Status**: ✅ PASSING

### Test 2: Cloudy Sky with Nano Model
- **File**: `test/yolov8_seg_test.py::test_yolov8_seg_cloudy_sky_panorama()`
- **Input**: `test/resources/cloudy_sky_panorama_stockcake.png`
- **Model**: YOLOv8 Nano
- **Detections**: 0 objects (sky/clouds not in COCO classes)
- **Output**: `test/resources/output/cloudy_sky_panorama_stockcake_yolov8seg.png`
- **Execution Time**: 3.44s

### Test 3: Cloudy Sky with Extra-Large Model
- **File**: `test/yolov8_seg_test.py::test_yolov8_seg_cloudy_sky_panorama_large_model()`
- **Input**: `test/resources/cloudy_sky_panorama_stockcake.png`
- **Model**: YOLOv8 Extra-Large
- **Detections**: 0 objects (sky/clouds not in COCO classes)
- **Output**: `test/resources/output/cloudy_sky_panorama_stockcake_yolov8x_seg.png`
- **Execution Time**: 4.57s

---

## Performance Benchmarks

### Inference Speed (CPU)
| Model | Cloudy Sky | GPU Recommended |
|-------|-----------|-----------------|
| Nano | 3.44s | Yes, for 30+ FPS real-time |
| Extra-Large | 4.57s | Recommended for production |

### Memory Usage
| Model | Weight File | Loaded Size | RAM Needed |
|-------|------------|-------------|-----------|
| Nano | 7 MB | ~14 MB | ~50 MB |
| Extra-Large | 137 MB | ~270 MB | ~500 MB |

---

## Dependencies

### Core Dependencies
- Python 3.11+
- PyTorch (cpu or cuda)
- torchvision
- torchaudio
- Ultralytics YOLO
- OpenCV (cv2)
- NumPy
- Matplotlib
- pytest

### Installation
All dependencies are installed automatically via setup scripts. Manual installation:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy matplotlib pytest
```

---

## Configuration

### Test Configuration
- **Confidence Threshold**: 0.25 (25% minimum detection confidence)
- **Input Format**: PNG, RGB/RGBA
- **Output Format**: PNG with segmentation overlays

### Model Confidence Tuning
In `test/yolov8_seg_test.py`, adjust confidence threshold:

```python
result = segmentor.run(image_path, conf=0.25)  # Adjust conf value (0.0-1.0)
```

Lower confidence = more detections (may include false positives)
Higher confidence = fewer detections (more precise)

---

## Documentation

### Comprehensive Testing Report
See [doc/yolov8seg_test/yolo_seg_object_classification_findings_cannot_determine_background.md](doc/yolov8seg_test/yolo_seg_object_classification_findings_cannot_determine_background.md) for:
- Detailed test results and analysis
- Model comparison and performance metrics
- COCO dataset classes reference
- Key findings and limitations
- Recommendations for alternative models
- Full test execution guide

---

## Known Limitations

### 1. Background/Landscape Detection
⚠️ **YOLOv8 cannot detect or classify background/landscape features** including:
- Grass, lawn, vegetation
- Sky, clouds, water
- Terrain features
- Other landscape elements

This is a **fundamental limitation** of the COCO training dataset (80 discrete object classes only).

### 2. Recommended Alternatives for Background Detection
For landscape/terrain classification, use:
- **Semantic Segmentation Models**: DeepLabv3, SegNet, FCN
- **Specialized Models**: DeepLabv3+ trained on ADE20K or Cityscapes
- **Agricultural AI**: Grass/weed detection models
- **Custom Training**: Fine-tune on domain-specific datasets

### 3. Other Limitations
- Inference requires adequate memory (500 MB+ for large model)
- No GPU support configured (can be added for GPU acceleration)
- COCO classes only (no custom object classes out-of-the-box)

---

## Troubleshooting

### Model Download Fails
**Issue**: Setup script fails to download models  
**Solution**: Manually download from Ultralytics:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt'); YOLO('yolov8x-seg.pt')"
```

### Tests Fail with Model Not Found
**Issue**: `FileNotFoundError` for model files  
**Solution**: Run setup script first:
```bash
# Windows
.\scripts\setup.ps1

# macOS/Linux
bash scripts/setup.sh
```

### Python Version Mismatch
**Issue**: Setup script reports Python version incompatibility  
**Solution**: Install Python 3.11 from https://www.python.org/downloads/

### Slow Inference Speed
**Issue**: Tests take too long (>10 seconds per image)  
**Solution**: GPU acceleration recommended - configure CUDA if available

---

## References

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **COCO Dataset**: https://cocodataset.org/
- **COCO Object Classes**: https://cocodataset.org/#explore
- **PyTorch Documentation**: https://pytorch.org/docs/
- **OpenCV Documentation**: https://docs.opencv.org/

---

## Development Status

- ✅ YOLOv8 nano model integration
- ✅ YOLOv8 extra-large model integration
- ✅ Instance segmentation tests
- ✅ Setup automation (Windows & macOS/Linux)
- ✅ Comprehensive documentation
- ⚠️ GPU support (not yet configured)
- ⚠️ Custom model training (separate branch)

---

## License & Attribution

This project uses:
- **YOLOv8**: Ultralytics YOLOv8 (AGPL-3.0)
- **COCO Dataset**: Copyright (c) 2014, COCO Consortium
- **PyTorch**: Meta Platforms, Inc.

---

**Last Updated**: June 6, 2026  
**Status**: Prototype/Testing  
**Python Version**: 3.11+  
**PyTorch Version**: Latest