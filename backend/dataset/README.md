# FSOC Beacon Detection Dataset

Auto-generated synthetic dataset for training YOLO beacon detection models.

## Validation Status: ✓ PASSED

### ✓ All validation checks passed!

**Per-Split Counts:**
- TRAIN: 6670 images, 6670 labels ✓
- VAL: 1670 images, 1670 labels ✓
- TEST: 1660 images, 1660 labels ✓

**Validation Checks:**
- Duplicate filenames: 0
- Split overlaps: 0
- Invalid labels: 0
- BBox outside image: 0
- Missing labels: 0
- Missing images: 0


## Dataset Statistics

- **Total Valid Images**: 10000
- **Rejected Samples**: 0
- **Train Split**: 6670 (66.7%)
- **Validation Split**: 1670 (16.7%)
- **Test Split**: 1660 (16.7%)

## Quality Metrics

- **Detector Detection Rate**: 100.0% (10000/10000 detected)
- **Detector Failures**: 0 (still labeled with ground truth)
- **Large Detection Errors**: 0
- **Invalid Labels**: 0

## Beacon Size Range

- **Minimum Beacon Size**: 8.0 pixels
- **Maximum Beacon Size**: 25.0 pixels

## Class Distribution

- **Class 0 (beacon)**: 10000 positive samples
- **Negative samples**: 0 (all images contain beacon)

## Dataset Structure

```
dataset/
├── images/
│   ├── train/       # Training images (6670 images)
│   ├── val/         # Validation images (1670 images)
│   └── test/        # Test images (1660 images)
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

### Unique Sample IDs

**Each sample has a globally unique ID** to prevent split overlap.

This ensures:
- ✅ No duplicate samples across train/val/test
- ✅ No data leakage between splits
- ✅ Valid evaluation metrics

### Validation

Samples were validated for:
- ✅ Beacon position within frame bounds
- ✅ Label coordinates in valid range [0, 1]
- ✅ Minimum beacon size for visibility
- ✅ Minimum brightness for detectability
- ✅ Unique filenames across all splits
- ✅ Image count matches label count per split
- ✅ No sample ID overlap between splits
- ✅ Beacon bbox fully inside image bounds

Rejected samples: 0 (out of 10000 attempts)

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

See `contact_sheet.png` for a visual sample of generated images with ground truth bounding boxes.

## Notes

- All images are PNG format (lossless)
- Labels are in YOLO TXT format (one per image)
- Dataset is fully synthetic - no manual labeling required
- Beacon is always present and correctly labeled with ground truth
- Ground truth labeling ensures 100% accuracy
- Unique sample IDs prevent split contamination

Generated: 10000 samples with automated ground-truth labeling.
