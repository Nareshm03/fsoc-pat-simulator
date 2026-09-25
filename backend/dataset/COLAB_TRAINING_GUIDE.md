# YOLO11 Training on Google Colab - Complete Guide

## Dataset Ready ✅

**10,000 Images Generated and Validated:**
- Train: 6,670 images (67%)
- Val: 1,670 images (17%)
- Test: 1,660 images (17%)
- All labels: Valid YOLO format
- Zero split overlaps
- 100% detector success rate

---

## Step 1: Prepare Dataset for Upload

### Option A: ZIP Dataset (Recommended)

**Windows PowerShell (from the repo root):**
```powershell
cd fsoc-pat-simulator\backend
Compress-Archive -Path dataset\* -DestinationPath dataset.zip -Force
```

**File size:** ~20 MB (10K images)

**What's included:**
```
dataset.zip
├── images/
│   ├── train/ (6,670 images)
│   ├── val/ (1,670 images)
│   └── test/ (1,660 images)
├── labels/
│   ├── train/ (6,670 labels)
│   ├── val/ (1,670 labels)
│   └── test/ (1,660 labels)
├── dataset.yaml
├── README.md
└── contact_sheet.png
```

### Option B: Upload to Google Drive

1. Create folder in Google Drive: `/MyDrive/FSOC/`
2. Upload `dataset.zip` to that folder
3. Use Drive mount in Colab (slower but persistent)

---

## Step 2: Upload to Google Colab

### Method 1: Direct Upload (Faster)

1. Open Google Colab: https://colab.research.google.com/
2. Upload notebook: `train_yolo_colab.ipynb`
3. Click **Runtime → Change runtime type → GPU (T4)**
4. Run **Step 1** (Setup Environment)
5. Use Colab file browser (left sidebar) to upload `dataset.zip`
6. Run **Step 2** to extract dataset

**Pros:** Faster upload  
**Cons:** Files deleted when runtime ends

### Method 2: Google Drive (Persistent)

1. Upload `dataset.zip` to Google Drive
2. In Colab, run:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/dataset.zip .
!unzip -q dataset.zip -d .
```

**Pros:** Persistent storage  
**Cons:** Slower access during training

---

## Step 3: Training Configuration

### Default Settings (train_yolo_colab.ipynb)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | YOLO11n | Nano (fastest, smallest) |
| Epochs | 100 | Can increase to 150-200 |
| Image Size | 640×640 | Standard YOLO size |
| Batch Size | 16 | Adjust based on GPU memory |
| Optimizer | AdamW | Adaptive learning rate |
| Early Stopping | 20 epochs | Stop if no improvement |
| GPU | T4 (free tier) | ~15GB VRAM |

### Expected Training Time

| Hardware | Time per Epoch | Total (100 epochs) |
|----------|----------------|-------------------|
| T4 GPU (Colab free) | ~2-3 min | ~3-5 hours |
| A100 GPU (Colab Pro) | ~30-45 sec | ~1 hour |
| V100 GPU | ~45-60 sec | ~1.5 hours |

### Model Variants

Change model size by editing:
```python
model = YOLO('yolo11n.pt')  # Nano (current)
```

Options:
- `yolo11n.pt` - Nano (fastest)
- `yolo11s.pt` - Small (balanced)
- `yolo11m.pt` - Medium (more accurate)
- `yolo11l.pt` - Large (high accuracy)
- `yolo11x.pt` - Extra Large (highest accuracy)

**Recommendation:** Start with `yolo11n` for speed, upgrade to `yolo11s` if accuracy insufficient.

---

## Step 4: Run Training

Execute cells in order:

1. **Setup Environment** - Install Ultralytics, check GPU
2. **Upload Dataset** - Extract dataset.zip
3. **Verify Dataset** - Confirm 10K images loaded
4. **Visualize Samples** - View images with GT boxes
5. **Train Model** - Start training (100 epochs)
6. **Evaluate** - Test set metrics (mAP, precision, recall)
7. **Visualize Results** - Training curves, confusion matrix
8. **Test Inference** - Run on test images
9. **Export Model** - Convert to ONNX for deployment
10. **Download** - Save trained weights

### Monitoring Training

Watch these metrics:
- **mAP50** - Target: >0.95 (95%)
- **mAP50-95** - Target: >0.80 (80%)
- **Precision** - Target: >0.95
- **Recall** - Target: >0.95
- **Box Loss** - Should decrease steadily

**Good training signs:**
- Train loss decreasing
- Val loss decreasing (not increasing = no overfitting)
- mAP increasing
- Precision/Recall high

**Warning signs:**
- Val loss increasing (overfitting)
- mAP plateaued early (<0.90)
- Large gap between train/val loss

---

## Step 5: Expected Results

### Target Metrics (10K dataset)

| Metric | Target | Excellent |
|--------|--------|-----------|
| mAP50 | >0.95 | >0.98 |
| mAP50-95 | >0.80 | >0.90 |
| Precision | >0.95 | >0.98 |
| Recall | >0.95 | >0.98 |
| Inference Speed | <10ms | <5ms |

### Why These Are Achievable

1. **Single class** - Only detecting beacons (simpler than multi-class)
2. **Clean data** - Ground truth labels (100% accurate)
3. **Large dataset** - 10K images with good variety
4. **Controlled environment** - Synthetic data, consistent lighting
5. **Clear target** - Bright beacons on dark background

---

## Step 6: Download Trained Model

After training completes:

1. Run **Step 9** (Export Model)
2. Downloads two files:
   - `best.pt` - PyTorch format (~5 MB for YOLO11n)
   - `best.onnx` - ONNX format

**Which to use:**
- **best.pt** - Use with Ultralytics library (easiest integration)
- **best.onnx** - Use for deployment (faster, smaller)

---

## Step 7: Integrate into Simulation

YOLO is already integrated in this repo (`backend/simulation/yolo_detector.py`
as `YOLO11Detector`, switchable live via `set_detector`, inference at
`imgsz=416` through the newest-frame-only async pipeline). The sketch below
shows the interface that integration follows:

**Integration code:**

```python
# backend/simulation/yolo_detector.py
from ultralytics import YOLO
import numpy as np
from typing import Optional, Dict

class YOLO11Detector:
    """YOLO11 beacon detector (see backend/simulation/yolo_detector.py)."""
    
    def __init__(self, model_path='best.pt', conf_threshold=0.25):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect beacon in frame.
        
        Args:
            frame: Input image (H, W, 3) BGR format
        
        Returns:
            Detection dict with x, y, confidence, or None if not detected
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Check if any detections
        if len(results[0].boxes) == 0:
            return None
        
        # Get first (highest confidence) detection
        box = results[0].boxes[0]
        
        # Extract center coordinates
        xyxy = box.xyxy[0].cpu().numpy()
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        confidence = float(box.conf[0])
        
        return {
            'x': float(x_center),
            'y': float(y_center),
            'confidence': confidence,
            'bbox': xyxy.tolist()
        }
```

### Update main.py

```python
# Replace current detector
from simulation.yolo_detector import YOLO11Detector

# In SimulationManager.__init__():
self.detector = YOLO11Detector(model_path='models/best.pt')
```

### Test Integration

```python
# backend/test_yolo_detector.py
import cv2
from simulation.yolo_detector import YOLO11Detector

# Initialize detector
detector = YOLO11Detector('models/best.pt')

# Test on sample image
frame = cv2.imread('dataset/images/test/beacon_008000.png')
result = detector.detect(frame)

if result:
    print(f"✓ Detected at ({result['x']:.1f}, {result['y']:.1f})")
    print(f"  Confidence: {result['confidence']:.2f}")
else:
    print("✗ No detection")
```

---

## Step 8: Performance Benchmarking

### Inference Speed Test

```python
import time
import numpy as np

# Generate 100 random frames
detector = YOLO11Detector('best.pt')
times = []

for _ in range(100):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    start = time.time()
    result = detector.detect(frame)
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000  # Convert to ms
fps = 1000 / avg_time

print(f"Average inference time: {avg_time:.2f} ms")
print(f"Theoretical FPS: {fps:.1f}")
```

**Target:** <10ms per frame (>100 FPS)

---

## Troubleshooting

### Issue: Colab Disconnects During Training

**Solution 1:** Keep browser tab active
**Solution 2:** Use Colab Pro ($10/month, more stable)
**Solution 3:** Add this to notebook:
```python
# Prevent timeout
import time
from IPython.display import display, Javascript

while True:
    display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'))
    time.sleep(300)  # Every 5 minutes
```

### Issue: Out of Memory

**Solution:** Reduce batch size
```python
# Change from batch=16 to batch=8 or batch=4
results = model.train(..., batch=8, ...)
```

### Issue: Low mAP (<0.90)

**Possible causes:**
1. Not enough epochs (increase to 150-200)
2. Model too small (try yolo11s instead of yolo11n)
3. Learning rate too high (let Ultralytics auto-tune)
4. Data quality issues (check dataset validation)

**Solution:**
```python
# Try larger model
model = YOLO('yolo11s.pt')

# More epochs
results = model.train(..., epochs=150, ...)
```

### Issue: Upload Fails

**Solution 1:** Use Google Drive method
**Solution 2:** Split dataset into smaller ZIPs
**Solution 3:** Use Colab Pro (faster upload)

---

## Summary Checklist

### Pre-Training
- [ ] Dataset verified (10K images, 67/17/17 split)
- [ ] dataset.yaml paths corrected (relative paths)
- [ ] Dataset zipped (<250 MB)
- [ ] Colab notebook ready (train_yolo_colab.ipynb)

### Training
- [ ] GPU enabled (T4 or better)
- [ ] Dataset uploaded and extracted
- [ ] Training started (100 epochs)
- [ ] Monitoring metrics (mAP, loss)

### Post-Training
- [ ] mAP50 >0.95 achieved
- [ ] Model exported (best.pt + best.onnx)
- [ ] Models downloaded locally
- [ ] Integration code ready

### Integration
- [ ] YOLO11 detector integrated (already in repo: `YOLO11Detector` + live switching)
- [ ] Inference speed tested
- [ ] Detection accuracy verified

---

## Files Summary

| File | Location | Purpose |
|------|----------|---------|
| `dataset.zip` | `backend/dataset.zip` | Dataset for upload |
| `train_yolo_colab.ipynb` | `backend/dataset/` | Training notebook |
| `dataset.yaml` | Inside dataset.zip | YOLO config |
| `COLAB_TRAINING_GUIDE.md` | `backend/dataset/` | This file |
| `best.pt` | Download from Colab | Trained model (PyTorch) |
| `best.onnx` | Download from Colab | Trained model (ONNX) |

---

## Status: ✅ READY FOR TRAINING

**Next step:** Upload dataset.zip to Google Colab and run training notebook!

Expected completion: 3-5 hours on free T4 GPU
Expected accuracy: mAP50 >0.95, mAP50-95 >0.80
