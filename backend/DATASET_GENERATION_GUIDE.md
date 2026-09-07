# Dataset Generation Guide

## Quick Start

### Generate a 10-Image Pilot Dataset
```bash
cd D:\Projects\SIH
python backend/generate_dataset_cli.py 10
```

### Verify Dataset Integrity
```bash
python backend/verify_dataset.py dataset
```

### Run Comprehensive Tests
```bash
python backend/test_dataset_integrity.py dataset
```

## Commands

### 1. Generate Dataset
```bash
python backend/generate_dataset_cli.py [num_images]
```

**Arguments:**
- `num_images` (optional): Number of images to generate (default: 10)

**Output:**
- `dataset/images/{train,val,test}/*.png` - Generated images
- `dataset/labels/{train,val,test}/*.txt` - YOLO format labels
- `dataset/dataset.yaml` - YOLO configuration file
- `dataset/README.md` - Dataset documentation
- `dataset/contact_sheet.png` - Visual QA grid

**Example:**
```bash
# Generate 10-image pilot
python backend/generate_dataset_cli.py 10

# Generate 1000-image dataset
python backend/generate_dataset_cli.py 1000

# Generate 20k production dataset
python backend/generate_dataset_cli.py 20000
```

### 2. Verify Dataset
```bash
python backend/verify_dataset.py [dataset_dir]
```

**Arguments:**
- `dataset_dir` (optional): Path to dataset directory (default: 'dataset')

**Checks:**
- ✓ Unique filenames across all splits
- ✓ Image count == label count per split
- ✓ No sample ID overlap between splits
- ✓ Valid YOLO label format
- ✓ Bounding boxes inside image bounds

**Exit Code:**
- `0` if validation passes
- `1` if validation fails

### 3. Comprehensive Test
```bash
python backend/test_dataset_integrity.py [dataset_dir]
```

**Arguments:**
- `dataset_dir` (optional): Path to dataset directory (default: 'dataset')

**Tests:**
1. Globally Unique Sample IDs
2. No Sample ID Overlap Between Splits
3. Image-Label Count Match Per Split
4. YOLO Label Format Validation
5. Bounding Box Inside Image Bounds
6. Complete File Structure

**Exit Code:**
- `0` if all tests pass
- `1` if any test fails

## Split Configuration

Default split ratios (configurable in `DatasetConfig`):
- **Train:** 67% (e.g., 6 out of 10, or 13,400 out of 20k)
- **Val:** 17% (e.g., 1 out of 10, or 3,400 out of 20k)
- **Test:** 16% (e.g., 3 out of 10, or 3,200 out of 20k)

To modify splits, edit `dataset_generator.py`:
```python
@dataclass
class DatasetConfig:
    train_split: float = 0.67  # 67% train
    val_split: float = 0.17    # 17% val
    test_split: float = 0.17   # 17% test (auto-adjusted to reach 100%)
```

## Dataset Structure

```
dataset/
├── images/
│   ├── train/           # Training images
│   │   ├── beacon_000000.png
│   │   ├── beacon_000001.png
│   │   └── ...
│   ├── val/             # Validation images
│   │   └── beacon_000006.png
│   └── test/            # Test images
│       ├── beacon_000007.png
│       └── ...
├── labels/
│   ├── train/           # Training labels (YOLO format)
│   │   ├── beacon_000000.txt
│   │   ├── beacon_000001.txt
│   │   └── ...
│   ├── val/             # Validation labels
│   │   └── beacon_000006.txt
│   └── test/            # Test labels
│       ├── beacon_000007.txt
│       └── ...
├── dataset.yaml         # YOLO config
├── README.md            # Documentation
└── contact_sheet.png    # Visual QA
```

## YOLO Label Format

Each `.txt` file contains one line per image:
```
<class> <x_center> <y_center> <width> <height>
```

Where:
- `class`: 0 (beacon)
- `x_center`: Normalized center X coordinate [0, 1]
- `y_center`: Normalized center Y coordinate [0, 1]
- `width`: Normalized box width [0, 1]
- `height`: Normalized box height [0, 1]

**Example:**
```
0 0.500000 0.450000 0.039063 0.069444
```

This represents a beacon centered at (50%, 45%) with width 3.9% and height 6.9%.

## Ground Truth Labeling

**IMPORTANT:** All labels use ground truth beacon positions, NOT detector output.

**Why?**
- Ensures 100% accurate annotations
- Works even when detector fails to detect beacon
- No mislabeled training data
- Detector output only used for quality checks

## Randomization

The generator randomizes:

### Beacon Properties
- Position X: 250-1030 pixels
- Position Y: 150-570 pixels
- Size: 8-25 pixels radius
- Brightness: 0.5-1.0 multiplier

### Environmental Conditions
- Turbulence: 0.0-1.5
- Sensor Noise: 0.0-1.5
- Camera Vibration: 0.0-1.5
- Motion Blur: 0-5 kernel size

## Validation Features

### Built-in Validation
The generator automatically validates:
1. Beacon within frame bounds
2. Label coordinates in [0, 1]
3. Minimum size/brightness for detectability
4. YOLO format correctness
5. No split overlaps

### Post-Generation Validation
Run verification scripts to confirm:
- Zero duplicate sample IDs
- Zero split overlaps
- All labels valid
- Image-label count matches

## Training with YOLOv8

Once dataset is generated and validated:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on your dataset
results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='beacon_detection'
)

# Validate
metrics = model.val()

# Inference on test set
model.predict('dataset/images/test', save=True)
```

## Troubleshooting

### Dataset Generation Fails
1. Check Python environment has all dependencies
2. Ensure sufficient disk space
3. Check logs in `logs/simulation_*.log`

### Validation Fails
1. Check error output for specific issues
2. Regenerate dataset if integrity issues found
3. Review `dataset/README.md` for generation stats

### Split Overlaps Detected
This should NOT happen with the fixed generator. If it does:
1. Report the bug
2. Delete dataset directory
3. Regenerate with fixed version

## Performance

### 10-Image Pilot
- Generation time: ~1-2 seconds
- Disk space: ~200 KB

### 1000-Image Dataset
- Generation time: ~1-2 minutes
- Disk space: ~20 MB

### 20k-Image Dataset
- Generation time: ~20-40 minutes
- Disk space: ~400 MB

## Contact Sheet

The contact sheet (`contact_sheet.png`) shows:
- Random samples from each split
- Ground truth bounding boxes (GREEN)
- Center points (RED)
- Split labels (colored borders)

Use this for quick visual QA of generated samples.

## Best Practices

1. **Start Small:** Generate 10-image pilot first
2. **Verify:** Run integrity tests before scaling up
3. **Review Contact Sheet:** Visually inspect samples
4. **Check README:** Review generation statistics
5. **Scale Up:** Generate full dataset once validated

## Exit Codes

All scripts return:
- `0` on success
- `1` on failure

Use in CI/CD pipelines:
```bash
python backend/generate_dataset_cli.py 1000 && \
python backend/test_dataset_integrity.py && \
echo "Dataset ready for training!"
```
