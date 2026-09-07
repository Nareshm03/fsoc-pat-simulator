# FSOC Beacon Detection Dataset

Auto-generated synthetic dataset for training YOLO beacon detection models.

## Dataset Statistics

- **Total Valid Images**: 10
- **Rejected Samples**: 0
- **Train Split**: 6 (60.0%)
- **Validation Split**: 2 (20.0%)
- **Test Split**: 2 (20.0%)

## Quality Metrics

- **Detector Detection Rate**: 100.0% (10/10 detected)
- **Detector Failures**: 0 (still labeled with ground truth)
- **Large Detection Errors**: 0
- **Invalid Labels**: 0

## Beacon Size Range

- **Minimum Beacon Size**: 10.2 pixels
- **Maximum Beacon Size**: 24.4 pixels

## Class Distribution

- **Class 0 (beacon)**: 10 positive samples
- **Negative samples**: 0 (all images contain beacon)

## Dataset Structure

```
dataset/
├── images/
│   ├── train/       # Training images (6 images)
│   ├── val/         # Validation images (2 images)
│   └── test/        # Test images (2 images)
├── labels/
│   ├── train/       # Training labels (YOLO format)
│   ├── val/         # Validation labels
│   └── test/        # Test labels
├── dataset.yaml     # YOLO configuration file
├── contact_sheet.png # Visual sample of generated images
└── README.md        # This file
```

## Important Notes

### Ground Truth Labels

**All labels use ground truth beacon position**, not detector output.

This ensures:
- ✅ Every image has accurate annotation
- ✅ Detector failures don't create mislabeled data
- ✅ Labels remain valid even for difficult samples
- ✅ Training data is 100% accurate

### Validation

Samples were validated for:
- ✅ Beacon position within frame bounds
- ✅ Label coordinates in valid range [0, 1]
- ✅ Minimum beacon size for visibility
- ✅ Minimum brightness for detectability

Rejected samples: 0 (out of 10 attempts)

## Randomization Parameters

### Beacon Properties
- **Position X**: 250 - 1030 pixels
- **Position Y**: 150 - 570 pixels
- **Size**: 8.0 - 25.0 pixels radius
- **Brightness**: 0.5 - 1.0 multiplier

### Environmental Conditions
- **Turbulence**: 0.0 - 1.5
- **Sensor Noise**: 0.0 - 1.5
- **Camera Vibration**: 0.0 - 1.5
- **Motion Blur**: 0 - 5 kernel size

## Label Format

YOLO format: `<class> <x_center> <y_center> <width> <height>`

- All coordinates normalized to [0, 1]
- Class 0 = beacon
- Bounding box is 2.5x beacon diameter

Example: `0 0.500000 0.450000 0.039063 0.069444`

## Training with YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='beacon_detection'
)

# Validate
metrics = model.val()

# Test
model.predict('dataset/images/test', save=True)
```

## Visual Inspection

See `contact_sheet.png` for a visual sample of generated images with bounding boxes.

## Notes

- All images are PNG format (lossless)
- Labels are in YOLO TXT format (one per image)
- Dataset is fully synthetic - no manual labeling required
- Beacon is always present and correctly labeled
- Ground truth labeling ensures 100% accuracy

Generated: 10 samples with automated ground-truth labeling.
